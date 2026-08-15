"""
interruption.py — Barge-in detection and graceful recovery logic.

When the user starts speaking while the agent is still talking (barge-in):
  1. Immediately cancel remaining TTS audio generation
  2. Signal the browser to stop audio playback
  3. Record the interruption event with context for analysis
  4. Provide a context note for the next LLM call so it recovers gracefully

Key design:
  - `is_agent_speaking` is set/cleared by main.py around TTS streaming
  - `on_speech_start` is called by VAD events — it checks for barge-in
  - Barge-in events are logged for Phase 4 evaluation metrics
"""
import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class InterruptionEvent:
    """Record of a single barge-in event."""
    timestamp: float
    agent_partial_response: str          # What agent had said so far
    interrupted_at_char: int             # How many chars into the response
    user_text_after: Optional[str] = None  # What the user said after interrupting

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "agent_said_so_far": self.agent_partial_response[:100],
            "interrupted_at_char": self.interrupted_at_char,
            "user_said": self.user_text_after,
        }


class InterruptionHandler:
    """
    Manages barge-in detection and recovery for a single call session.

    Usage in main.py:
        handler = InterruptionHandler(send_stop_audio_fn)
        handler.set_agent_speaking(True, "Hello, you need a visa for...")
        # ... stream TTS ...
        # When VAD fires speech_start:
        was_interrupted = await handler.check_barge_in(cancel_tts_fn)
    """

    def __init__(self, send_stop_fn: Callable[[], Awaitable[None]]):
        """
        Args:
            send_stop_fn: Async function that sends 'stop_audio' to browser
        """
        self.send_stop = send_stop_fn
        self.is_agent_speaking: bool = False
        self._current_response_text: str = ""
        self._chars_spoken: int = 0
        self._interruption_events: list[InterruptionEvent] = []
        self._pending_interruption: Optional[InterruptionEvent] = None

    def set_agent_speaking(self, speaking: bool, response_text: str = ""):
        """Called by main.py before/after TTS streaming."""
        self.is_agent_speaking = speaking
        if speaking:
            self._current_response_text = response_text
            self._chars_spoken = 0
            logger.debug("Agent is now speaking")
        else:
            logger.debug("Agent finished speaking")

    def on_sentence_spoken(self, sentence: str):
        """Track progress of how much the agent has said."""
        self._chars_spoken += len(sentence)

    async def check_barge_in(self) -> bool:
        """
        Called when VAD detects user speech_start.
        If agent is speaking → barge-in detected → stop TTS, log event.

        Returns:
            True if a barge-in was detected and handled.
        """
        if not self.is_agent_speaking:
            return False

        logger.info(
            f"BARGE-IN detected! Agent had spoken {self._chars_spoken} chars "
            f"of '{self._current_response_text[:60]}...'"
        )

        # Record the interruption event
        event = InterruptionEvent(
            timestamp=time.monotonic(),
            agent_partial_response=self._current_response_text[:self._chars_spoken],
            interrupted_at_char=self._chars_spoken,
        )
        self._pending_interruption = event
        self._interruption_events.append(event)

        # Signal browser to stop playback immediately
        await self.send_stop()

        # Mark agent as not speaking
        self.is_agent_speaking = False

        return True

    def get_recovery_context(self) -> str:
        """
        Build a context note for the LLM about the barge-in.
        Injected into the next orchestrator call so the agent
        doesn't repeat itself.
        """
        if not self._pending_interruption:
            return ""

        event = self._pending_interruption
        self._pending_interruption = None   # Consume it

        if event.interrupted_at_char < 20:
            return "The user interrupted before you had said much. Just listen and respond fresh."
        else:
            partial = event.agent_partial_response[:80]
            return (
                f"You were interrupted mid-response after saying: '{partial}...'. "
                f"Do NOT repeat what you already said. Just address what the user says next."
            )

    def complete_last_interruption(self, user_text: str):
        """Record what the user said after interrupting."""
        if self._interruption_events:
            self._interruption_events[-1].user_text_after = user_text

    def get_stats(self) -> dict:
        return {
            "total_interruptions": len(self._interruption_events),
            "events": [e.to_dict() for e in self._interruption_events],
        }
