"""
tts.py — Text-to-Speech integration (Edge TTS).

Uses completely free Microsoft Edge Read Aloud API.
Streams synthesised audio back as bytes chunks so the browser
can start playing before the full response is generated.
"""
import asyncio
import time
import logging
from typing import AsyncGenerator
import edge_tts
from backend.config import settings

logger = logging.getLogger(__name__)

async def synthesise_speech(text: str) -> AsyncGenerator[bytes, None]:
    """
    Convert text to speech using Edge TTS, yielding audio bytes chunks as they arrive.

    Args:
        text: The text to synthesise.

    Yields:
        Raw audio bytes (MP3 format).
    """
    if not text.strip():
        return

    start_time = time.monotonic()
    first_chunk = True

    try:
        communicate = edge_tts.Communicate(text, settings.EDGE_TTS_VOICE)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                if first_chunk:
                    latency_ms = (time.monotonic() - start_time) * 1000
                    logger.info(f"TTS first byte latency: {latency_ms:.0f}ms")
                    first_chunk = False
                yield chunk["data"]
    except Exception as e:
        logger.error(f"Edge TTS error: {e}")
