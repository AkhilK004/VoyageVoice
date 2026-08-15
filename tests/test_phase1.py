"""
test_phase1.py — Integration tests for Phase 1: Voice Pipeline

Tests the following without needing a live WebSocket:
  1. Config loads correctly from .env
  2. STTSession can be instantiated
  3. TTS (Edge TTS) actually generates audio bytes
  4. ConversationOrchestrator streams sentences correctly via Groq
  5. Sentence chunking logic works

Run: python -m pytest tests/test_phase1.py -v
"""
import asyncio
import os
import sys
import pytest

# Ensure backend package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.config import settings


# ── Test 1: Config ─────────────────────────────────────────────────────────────
class TestConfig:
    def test_groq_key_loaded(self):
        """GROQ_API_KEY must be set in .env"""
        assert settings.GROQ_API_KEY, "GROQ_API_KEY is missing from .env"

    def test_deepgram_key_loaded(self):
        """DEEPGRAM_API_KEY must be set in .env"""
        assert settings.DEEPGRAM_API_KEY, "DEEPGRAM_API_KEY is missing from .env"

    def test_groq_model_default(self):
        assert settings.GROQ_MODEL == "llama-3.1-8b-instant"

    def test_edge_tts_voice_default(self):
        assert "Neural" in settings.EDGE_TTS_VOICE, "Edge TTS voice should be a Neural voice"


# ── Test 2: STT Session ───────────────────────────────────────────────────────
class TestSTT:
    def test_stt_session_instantiates(self):
        """STTSession should instantiate without errors."""
        from backend.voice.stt import STTSession
        async def dummy_callback(text, is_final, latency_ms):
            pass
        session = STTSession(on_transcript=dummy_callback)
        assert session is not None


# ── Test 3: TTS ───────────────────────────────────────────────────────────────
class TestTTS:
    def test_tts_generates_audio(self):
        """Edge TTS should produce > 0 bytes for a short phrase."""
        from backend.voice.tts import synthesise_speech

        async def _run():
            chunks = []
            async for chunk in synthesise_speech("Hello, this is a test."):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) > 0, "TTS produced no audio chunks"
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes > 1000, f"TTS audio too small: {total_bytes} bytes"
        print(f"\n  ✓ TTS produced {total_bytes} bytes in {len(chunks)} chunks")

    def test_tts_empty_input_no_crash(self):
        """Empty text should yield nothing, not crash."""
        from backend.voice.tts import synthesise_speech

        async def _run():
            chunks = []
            async for chunk in synthesise_speech("   "):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_run())
        assert chunks == [], "Empty input should yield zero chunks"


# ── Test 4: Orchestrator (LLM via Groq) ──────────────────────────────────────
class TestOrchestrator:
    def test_orchestrator_streams_sentences(self):
        """Orchestrator should yield at least one sentence from Groq."""
        from backend.intelligence.orchestrator import ConversationOrchestrator

        async def _run():
            orch = ConversationOrchestrator()
            sentences = []
            async for s in orch.respond("Do Indians need a visa to visit Japan?"):
                sentences.append(s)
            return sentences

        sentences = asyncio.run(_run())
        assert len(sentences) >= 1, "Orchestrator yielded no sentences"
        full = " ".join(sentences)
        assert len(full) > 10, "LLM response is too short"
        print(f"\n  ✓ LLM produced {len(sentences)} sentence(s): '{full[:80]}...'")

    def test_orchestrator_maintains_history(self):
        """Orchestrator should maintain multi-turn history."""
        from backend.intelligence.orchestrator import ConversationOrchestrator

        async def _run():
            orch = ConversationOrchestrator()
            # First turn
            async for _ in orch.respond("I'm Indian."):
                pass
            # Second turn — should remember context
            sentences = []
            async for s in orch.respond("Do I need a visa for Japan?"):
                sentences.append(s)
            return orch.get_history_summary(), sentences

        summary, sentences = asyncio.run(_run())
        assert summary["turn_count"] == 2
        assert len(sentences) >= 1
        print(f"\n  ✓ Turn count: {summary['turn_count']}, history length: {summary['message_count']}")

    def test_sentence_chunking(self):
        """Sentence regex should split correctly."""
        import re
        from backend.intelligence.orchestrator import SENTENCE_END
        text = "You need a visa. It costs 30 dollars. Apply online."
        parts = SENTENCE_END.split(text)
        assert len(parts) == 3, f"Expected 3 parts, got {parts}"
        print(f"\n  ✓ Sentence split: {parts}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
