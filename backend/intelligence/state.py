"""
state.py — Conversation state machine for the travel agent.

States track where we are in the conversation flow so the agent
can ask the right questions at the right time instead of jumping
straight to answers without understanding the query.

States:
    GREETING          → Just started, welcoming the user
    UNDERSTANDING     → Figuring out what the user needs
    GATHERING_INFO    → Have the intent, but need passport/destination
    ANSWERING         → Have enough info, providing the answer
    FOLLOW_UP         → Answer given, checking if user needs more
    WRAP_UP           → Conversation winding down
    ESCALATED         → Handed off to human
"""
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    GREETING = "greeting"
    UNDERSTANDING = "understanding"
    GATHERING_INFO = "gathering_info"
    ANSWERING = "answering"
    FOLLOW_UP = "follow_up"
    WRAP_UP = "wrap_up"
    ESCALATED = "escalated"


@dataclass
class ConversationContext:
    """
    Tracks extracted entities and conversation state across turns.
    Think of this as the agent's 'working memory' for a call.
    """
    state: ConversationState = ConversationState.GREETING
    turn_count: int = 0
    passport_country: Optional[str] = None    # e.g. "India"
    destination: Optional[str] = None         # e.g. "Japan"
    visa_type: Optional[str] = None           # e.g. "Tourist Visa"
    query_intent: Optional[str] = None        # e.g. "eligibility", "documents", "processing_time"
    tools_called: list = field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None

    def has_enough_info(self) -> bool:
        """Return True if we have both passport country and destination."""
        return self.passport_country is not None and self.destination is not None

    def missing_info_prompt(self) -> str:
        """Return a natural question for the missing piece of info."""
        if not self.passport_country and not self.destination:
            return "Which passport do you hold, and where are you planning to travel to?"
        if not self.passport_country:
            return "Which country passport do you hold?"
        if not self.destination:
            return "Which country are you planning to visit?"
        return ""

    def advance_state(self, new_state: ConversationState):
        if self.state != new_state:
            logger.debug(f"State: {self.state.value} → {new_state.value}")
            self.state = new_state

    def get_state_instructions(self) -> str:
        """
        Return state-specific instructions to inject into the system prompt.
        These guide the LLM's behavior at each stage.
        """
        instructions = {
            ConversationState.GREETING: (
                "The user has just connected. Greet them warmly and ask how you can help with their travel plans. "
                "Be brief."
            ),
            ConversationState.UNDERSTANDING: (
                "Listen carefully to understand what the user needs. "
                "If it's a visa question, identify the intent (eligibility/documents/fees/processing time). "
                "Do NOT answer yet if you're missing the passport country or destination."
            ),
            ConversationState.GATHERING_INFO: (
                f"You need more information before answering. "
                f"Ask for: {self.missing_info_prompt()} "
                f"Keep it natural and brief."
            ),
            ConversationState.ANSWERING: (
                f"You now have the full context: "
                f"Passport: {self.passport_country}, Destination: {self.destination}. "
                "Use the retrieved information and tool results to give a clear, accurate answer. "
                "Speak naturally — you're on a voice call. "
                "Summarise key points (visa type, fee, time) in 2-3 sentences max."
            ),
            ConversationState.FOLLOW_UP: (
                "You've answered the user's main question. Ask if they need anything else. "
                "Remember context from earlier in the call."
            ),
            ConversationState.WRAP_UP: (
                "The user seems satisfied. Wish them a great trip and offer to help any time."
            ),
            ConversationState.ESCALATED: (
                "This case has been escalated to a human agent. "
                "Reassure the user that a specialist will assist them shortly."
            ),
        }
        return instructions.get(self.state, "")

    def extract_entities_from_text(self, text: str):
        """
        Simple keyword-based entity extraction.
        For Phase 2 this is enough; Phase 3+ can use LLM-based extraction.
        """
        text_lower = text.lower()

        # Passport country detection
        country_keywords = {
            "india": "India", "indian": "India",
            "usa": "USA", "american": "USA", "united states": "USA",
            "uk": "UK", "british": "UK", "england": "UK",
            "australia": "Australia", "australian": "Australia",
            "canada": "Canada", "canadian": "Canada",
            "germany": "Germany", "german": "Germany",
            "china": "China", "chinese": "China",
            "pakistan": "Pakistan", "pakistani": "Pakistan",
            "singapore": "Singapore",
        }
        for keyword, country in country_keywords.items():
            if keyword in text_lower:
                # Heuristic: "I'm Indian" or "Indian passport" → passport
                if self.passport_country is None and any(
                    phrase in text_lower for phrase in
                    ["i'm", "i am", "my passport", "hold", "citizen", "from"]
                ):
                    self.passport_country = country
                    logger.debug(f"Extracted passport: {country}")
                break

        # Destination detection
        destination_keywords = {
            "japan": "Japan", "japanese": "Japan",
            "thailand": "Thailand", "bangkok": "Thailand",
            "usa": "USA", "america": "USA", "united states": "USA",
            "uk": "UK", "london": "UK", "england": "UK",
            "europe": "Schengen (Europe)", "schengen": "Schengen (Europe)",
            "australia": "Australia", "sydney": "Australia",
            "canada": "Canada", "toronto": "Canada",
            "singapore": "Singapore",
            "dubai": "Dubai (UAE)", "uae": "Dubai (UAE)",
            "malaysia": "Malaysia", "kuala lumpur": "Malaysia",
        }
        for keyword, dest in destination_keywords.items():
            if keyword in text_lower and self.destination is None:
                # Heuristic: "travel to", "visit", "go to" → destination
                if any(phrase in text_lower for phrase in
                       ["travel to", "visit", "go to", "trip to", "visa for", "flying to"]):
                    self.destination = dest
                    logger.debug(f"Extracted destination: {dest}")
                    break
