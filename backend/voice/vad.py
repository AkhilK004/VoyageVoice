"""
vad.py — Server-side Voice Activity Detection state tracker.

Receives speech_start / speech_end events sent from the browser's
client-side VAD (energy-based AnalyserNode) and maintains a clean
speaking state per session.

Responsibilities:
  - Track whether the user is currently speaking
  - Compute silence duration (time since last speech_end)
  - Provide state info to the interruption detector
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VADState:
    """
    Tracks voice activity state for a single call session.
    Updated by browser events: speech_start, speech_end.
    """
    is_user_speaking: bool = False
    speech_start_time: Optional[float] = None     # monotonic timestamp
    speech_end_time: Optional[float] = None       # monotonic timestamp
    last_utterance_duration_ms: Optional[float] = None
    total_speaking_time_ms: float = 0.0
    speech_event_count: int = 0

    # Silence threshold: how long (ms) of silence before treating utterance as done
    SILENCE_THRESHOLD_MS: float = 800.0

    def on_speech_start(self):
        """Called when browser VAD detects user has started speaking."""
        if not self.is_user_speaking:
            self.is_user_speaking = True
            self.speech_start_time = time.monotonic()
            self.speech_event_count += 1
            logger.debug("VAD: speech_start")

    def on_speech_end(self):
        """Called when browser VAD detects user has stopped speaking."""
        if self.is_user_speaking:
            self.is_user_speaking = False
            self.speech_end_time = time.monotonic()
            if self.speech_start_time:
                duration = (self.speech_end_time - self.speech_start_time) * 1000
                self.last_utterance_duration_ms = duration
                self.total_speaking_time_ms += duration
                logger.debug(f"VAD: speech_end after {duration:.0f}ms")

    def silence_duration_ms(self) -> Optional[float]:
        """How long (ms) since the user last stopped speaking."""
        if self.speech_end_time and not self.is_user_speaking:
            return (time.monotonic() - self.speech_end_time) * 1000
        return None

    def is_utterance_complete(self) -> bool:
        """
        True if user has stopped speaking and silence exceeds threshold.
        Used to decide when to send the transcript to the LLM.
        """
        silence = self.silence_duration_ms()
        return silence is not None and silence >= self.SILENCE_THRESHOLD_MS

    def get_stats(self) -> dict:
        return {
            "is_speaking": self.is_user_speaking,
            "speech_events": self.speech_event_count,
            "total_speaking_ms": round(self.total_speaking_time_ms, 1),
            "last_utterance_ms": round(self.last_utterance_duration_ms or 0, 1),
        }
