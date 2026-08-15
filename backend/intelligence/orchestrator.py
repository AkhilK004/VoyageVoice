"""
orchestrator.py — Core agent orchestration (Phase 2: RAG + Tool Calling + State Machine).

Pipeline per turn:
  1. Extract entities (passport, destination) from user text
  2. Advance conversation state
  3. Run RAG retrieval to fetch relevant visa knowledge
  4. Call LLM with: system prompt + state instructions + RAG context + tool schemas
  5. Handle tool_call responses → execute tool → feed result back to LLM
  6. Stream final response sentence-by-sentence for TTS
"""
import asyncio
import json
import time
import logging
import re
from typing import AsyncGenerator

from openai import AsyncOpenAI
from backend.config import settings
from backend.intelligence.prompts import SYSTEM_PROMPT
from backend.intelligence.state import ConversationContext, ConversationState
from backend.intelligence.tools import TOOL_SCHEMAS, execute_tool
from backend.intelligence.rag import build_rag_context

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Split on sentence boundaries (. ! ? followed by space)
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


class ConversationOrchestrator:
    """
    Manages a single call session with full Phase 2 intelligence:
    RAG retrieval, tool calling, and state machine guidance.
    """

    def __init__(self):
        self.history: list[dict] = []   # Does NOT include system prompt; built dynamically
        self.context = ConversationContext()
        self.turn_count = 0
        self._interruption_context: str = ""   # Set by Phase 3 barge-in handler

    def inject_interruption_context(self, context: str):
        """
        Called by main.py when a barge-in is detected.
        This note is injected into the next LLM call so the agent
        doesn't repeat what it was saying before being interrupted.
        """
        self._interruption_context = context

    def _build_system_prompt(self, rag_context: str) -> str:
        """Build a dynamic system prompt with state instructions and RAG context."""
        state_instructions = self.context.get_state_instructions()

        prompt = SYSTEM_PROMPT

        if state_instructions:
            prompt += f"\n\n## Current Conversation Stage\n{state_instructions}"

        if rag_context:
            prompt += (
                f"\n\n## Retrieved Visa Information\n"
                f"Use the following accurate visa data to answer. "
                f"Do NOT make up facts not listed here.\n\n"
                f"{rag_context}"
            )
        else:
            prompt += (
                "\n\n## Knowledge Base\nNo specific visa data retrieved for this query. "
                "Be honest about uncertainty and suggest the user verify with official embassy sources."
            )

        # Phase 3: Barge-in recovery context
        if self._interruption_context:
            prompt += f"\n\n## Interruption Note\n{self._interruption_context}"
            self._interruption_context = ""   # Consume it (one-shot)

        return prompt

    async def respond(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Full Phase 2 pipeline:
        entity extraction → state advance → RAG → LLM + tools → stream sentences.
        """
        self.turn_count += 1
        self.context.turn_count = self.turn_count

        # 1. Extract entities from user text
        self.context.extract_entities_from_text(user_text)

        # 2. Advance state based on context
        if self.context.state == ConversationState.GREETING:
            self.context.advance_state(ConversationState.UNDERSTANDING)
        elif self.context.state == ConversationState.UNDERSTANDING:
            if self.context.has_enough_info():
                self.context.advance_state(ConversationState.ANSWERING)
            else:
                self.context.advance_state(ConversationState.GATHERING_INFO)
        elif self.context.state == ConversationState.GATHERING_INFO:
            if self.context.has_enough_info():
                self.context.advance_state(ConversationState.ANSWERING)
        elif self.context.state in (ConversationState.ANSWERING, ConversationState.FOLLOW_UP):
            self.context.advance_state(ConversationState.FOLLOW_UP)

        # 3. RAG retrieval (only if we have a meaningful query)
        rag_context = ""
        if len(user_text.split()) > 2:
            rag_context = build_rag_context(user_text)

        # 4. Build messages for LLM
        system_msg = {"role": "system", "content": self._build_system_prompt(rag_context)}
        self.history.append({"role": "user", "content": user_text})
        messages = [system_msg] + self.history

        start_time = time.monotonic()
        logger.info(
            f"Turn {self.turn_count} | State: {self.context.state.value} | "
            f"Passport: {self.context.passport_country} | Dest: {self.context.destination}"
        )

        # 5. LLM call with tool schemas (Groq supports Llama 3 tool calling)
        full_response = ""
        try:
            async for sentence in self._call_llm_with_tools(messages, start_time):
                full_response += sentence + " "
                yield sentence
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            yield "I'm sorry, something went wrong. Could you repeat that?"
            return

        total_latency = (time.monotonic() - start_time) * 1000
        logger.info(f"Turn complete in {total_latency:.0f}ms")

        # 6. Add assistant response to history
        self.history.append({"role": "assistant", "content": full_response.strip()})

    async def _call_llm_with_tools(
        self, messages: list, start_time: float
    ) -> AsyncGenerator[str, None]:
        """
        Call LLM. If it wants to use a tool, execute it and call LLM again.
        Yields sentences from the final text response.
        """
        first_token_logged = False

        # Initial LLM call
        response = await _client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=400,
            temperature=0.6,
            stream=False,    # Non-streaming for tool call detection
        )

        choice = response.choices[0]

        # Handle tool call
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_results_text = []

            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                logger.info(f"Tool call: {tool_name}({arguments})")
                result = execute_tool(tool_name, arguments)
                self.context.tools_called.append(tool_name)

                # Handle escalation
                if tool_name == "escalate_to_human":
                    self.context.escalated = True
                    self.context.advance_state(ConversationState.ESCALATED)

                tool_results_text.append(
                    f"Tool '{tool_name}' result: {json.dumps(result, indent=2)}"
                )

            # Feed tool results back to LLM for a natural response
            messages_with_tool = messages + [
                choice.message,
                {
                    "role": "tool",
                    "tool_call_id": choice.message.tool_calls[0].id,
                    "content": "\n\n".join(tool_results_text),
                }
            ]

            # Second LLM call with tool results — stream this one
            async for sentence in self._stream_llm(messages_with_tool, start_time):
                yield sentence

        else:
            # No tool call — yield the direct response as sentences
            text = choice.message.content or ""
            for sentence in SENTENCE_END.split(text):
                s = sentence.strip()
                if s:
                    yield s

    async def _stream_llm(
        self, messages: list, start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response and yield complete sentences."""
        first_token_logged = False
        buffer = ""

        stream = await _client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.6,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue

            if not first_token_logged:
                latency_ms = (time.monotonic() - start_time) * 1000
                logger.info(f"LLM first token: {latency_ms:.0f}ms")
                first_token_logged = True

            buffer += delta
            sentences = SENTENCE_END.split(buffer)
            if len(sentences) > 1:
                for s in sentences[:-1]:
                    s = s.strip()
                    if s:
                        yield s
                buffer = sentences[-1]

        if buffer.strip():
            yield buffer.strip()

    def get_history_summary(self) -> dict:
        return {
            "turn_count": self.turn_count,
            "message_count": len(self.history),
            "state": self.context.state.value,
            "passport": self.context.passport_country,
            "destination": self.context.destination,
            "tools_called": self.context.tools_called,
            "escalated": self.context.escalated,
        }
