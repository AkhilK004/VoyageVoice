"""
stt.py — Deepgram streaming Speech-to-Text integration.

Accepts raw audio bytes (WebM/opus from browser MediaRecorder),
streams them to Deepgram, and yields transcript results with latency info.
"""
import asyncio
import time
import logging
from typing import AsyncGenerator, Optional
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
from backend.config import settings

logger = logging.getLogger(__name__)


class STTSession:
    """
    Manages a single Deepgram streaming STT session for one WebSocket call.
    Usage:
        session = STTSession(on_transcript_callback)
        await session.start()
        await session.send_audio(chunk)
        ...
        await session.finish()
    """

    def __init__(self, on_transcript):
        """
        Args:
            on_transcript: async callable(text: str, is_final: bool, latency_ms: float)
        """
        self.on_transcript = on_transcript
        self._connection = None
        self._client = None
        self._started_at: Optional[float] = None
        self._audio_received_at: Optional[float] = None

    async def start(self):
        """Open the Deepgram live connection."""
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self._client = DeepgramClient(settings.DEEPGRAM_API_KEY, config)
        self._connection = self._client.listen.asynclive.v("1")

        # Register event handlers
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_message)
        self._connection.on(LiveTranscriptionEvents.Error, self._on_error)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            interim_results=True,          # Stream partial results
            utterance_end_ms="1000",       # Silence to mark utterance end
            vad_events=True,               # Voice activity detection events
            encoding="webm-opus",          # Browser MediaRecorder default
        )

        started = await self._connection.start(options)
        if not started:
            raise RuntimeError("Failed to start Deepgram connection")

        self._started_at = time.monotonic()
        logger.info("Deepgram STT session started")

    async def send_audio(self, audio_bytes: bytes):
        """Send a chunk of audio bytes to Deepgram."""
        if self._connection:
            self._audio_received_at = time.monotonic()
            await self._connection.send(audio_bytes)

    async def finish(self):
        """Close the Deepgram connection gracefully."""
        if self._connection:
            await self._connection.finish()
            logger.info("Deepgram STT session closed")

    async def _on_message(self, *args, **kwargs):
        """Handle incoming transcript events."""
        result = kwargs.get("result")
        if not result:
            return

        try:
            sentence = result.channel.alternatives[0].transcript
            if not sentence.strip():
                return

            is_final = result.is_final
            latency_ms = None

            if self._audio_received_at:
                latency_ms = (time.monotonic() - self._audio_received_at) * 1000

            logger.debug(f"STT {'FINAL' if is_final else 'interim'}: '{sentence}' ({latency_ms:.0f}ms)")
            await self.on_transcript(sentence, is_final, latency_ms)

        except (AttributeError, IndexError) as e:
            logger.warning(f"Error parsing transcript result: {e}")

    async def _on_error(self, *args, **kwargs):
        error = kwargs.get("error")
        logger.error(f"Deepgram error: {error}")
