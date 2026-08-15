"""
test_phase3.py — Integration tests for Phase 3: Interruption Handling + VAD.

Also runs all Phase 1 + Phase 2 regression tests.

Tests:
  Regressions (P1 + P2):
    - Config, TTS, Orchestrator still work
    - RAG retrieval, Tools, State machine still work

  Phase 3 — VAD:
    - VADState tracks speech_start / speech_end correctly
    - Silence duration computed correctly
    - is_utterance_complete fires after threshold

  Phase 3 — Interruption Handler:
    - Barge-in detected when agent is speaking
    - Barge-in NOT detected when agent is silent
    - Recovery context generated correctly (partial vs early)
    - last interruption annotated with user text

  Phase 3 — Orchestrator:
    - inject_interruption_context stored correctly
    - Injected context is consumed (one-shot, cleared after use)
    - Barge-in context appears in built prompt

Run: python -m pytest tests/test_phase3.py -v -s
"""
import asyncio
import os
import sys
import time
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase1Regression:
    def test_config_loads(self):
        from backend.config import settings
        assert settings.GROQ_API_KEY
        assert settings.DEEPGRAM_API_KEY

    def test_tts_generates_audio(self):
        from backend.voice.tts import synthesise_speech
        async def _run():
            chunks = []
            async for c in synthesise_speech("Interruption test audio."):
                chunks.append(c)
            return sum(len(c) for c in chunks)
        total = asyncio.run(_run())
        assert total > 500, f"TTS regression: {total} bytes"
        print(f"\n  ✓ TTS: {total} bytes")

    def test_orchestrator_streams_sentences(self):
        from backend.intelligence.orchestrator import ConversationOrchestrator
        async def _run():
            orch = ConversationOrchestrator()
            sentences = []
            async for s in orch.respond("Hello"):
                sentences.append(s)
            return sentences
        sentences = asyncio.run(_run())
        assert len(sentences) >= 1
        print(f"\n  ✓ LLM: {len(sentences)} sentence(s)")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
class TestPhase2Regression:
    def test_rag_retrieves_results(self):
        from backend.intelligence.rag import build_rag_context
        ctx = build_rag_context("visa for Japan from India")
        assert len(ctx) > 50
        print(f"\n  ✓ RAG: {len(ctx)} chars retrieved")

    def test_tool_visa_eligibility(self):
        from backend.intelligence.tools import check_visa_eligibility
        r = check_visa_eligibility("India", "Japan")
        assert r["found"] is True
        print(f"\n  ✓ Tool: {r['visa_type']}")

    def test_state_machine_extracts_entities(self):
        from backend.intelligence.state import ConversationContext
        ctx = ConversationContext()
        ctx.extract_entities_from_text("I'm Indian and want to visit Japan")
        assert ctx.passport_country == "India"
        assert ctx.destination == "Japan"
        print(f"\n  ✓ State: {ctx.passport_country} → {ctx.destination}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — VAD TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestVAD:
    def test_initial_state(self):
        from backend.voice.vad import VADState
        vad = VADState()
        assert vad.is_user_speaking is False
        assert vad.speech_event_count == 0
        assert vad.silence_duration_ms() is None

    def test_speech_start(self):
        from backend.voice.vad import VADState
        vad = VADState()
        vad.on_speech_start()
        assert vad.is_user_speaking is True
        assert vad.speech_event_count == 1
        assert vad.speech_start_time is not None

    def test_speech_end(self):
        from backend.voice.vad import VADState
        vad = VADState()
        vad.on_speech_start()
        time.sleep(0.1)
        vad.on_speech_end()
        assert vad.is_user_speaking is False
        assert vad.last_utterance_duration_ms is not None
        assert vad.last_utterance_duration_ms >= 80  # at least 80ms
        print(f"\n  ✓ Utterance duration: {vad.last_utterance_duration_ms:.0f}ms")

    def test_silence_duration(self):
        from backend.voice.vad import VADState
        vad = VADState()
        vad.on_speech_start()
        vad.on_speech_end()
        time.sleep(0.1)
        silence = vad.silence_duration_ms()
        assert silence is not None
        assert silence >= 80  # at least 80ms of silence
        print(f"\n  ✓ Silence after speech_end: {silence:.0f}ms")

    def test_silence_duration_none_while_speaking(self):
        from backend.voice.vad import VADState
        vad = VADState()
        vad.on_speech_start()
        # Silence duration should be None while still speaking
        assert vad.silence_duration_ms() is None

    def test_utterance_complete_after_threshold(self):
        from backend.voice.vad import VADState
        vad = VADState()
        vad.SILENCE_THRESHOLD_MS = 50  # Lower threshold for test speed
        vad.on_speech_start()
        vad.on_speech_end()
        time.sleep(0.08)  # 80ms > 50ms threshold
        assert vad.is_utterance_complete() is True
        print(f"\n  ✓ Utterance complete after threshold silence")

    def test_not_complete_too_soon(self):
        from backend.voice.vad import VADState
        vad = VADState()
        vad.SILENCE_THRESHOLD_MS = 5000  # Very high threshold
        vad.on_speech_start()
        vad.on_speech_end()
        assert vad.is_utterance_complete() is False

    def test_double_speech_start_idempotent(self):
        """Calling speech_start twice should only count as 1 event."""
        from backend.voice.vad import VADState
        vad = VADState()
        vad.on_speech_start()
        vad.on_speech_start()
        assert vad.speech_event_count == 1

    def test_stats_returns_dict(self):
        from backend.voice.vad import VADState
        vad = VADState()
        vad.on_speech_start()
        vad.on_speech_end()
        stats = vad.get_stats()
        assert "is_speaking" in stats
        assert "speech_events" in stats
        assert "total_speaking_ms" in stats
        print(f"\n  ✓ VAD stats: {stats}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — INTERRUPTION HANDLER TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestInterruptionHandler:
    def _make_handler(self):
        """Create an InterruptionHandler with a mock stop function."""
        from backend.voice.interruption import InterruptionHandler
        stop_calls = []
        async def mock_stop():
            stop_calls.append(True)
        handler = InterruptionHandler(send_stop_fn=mock_stop)
        return handler, stop_calls

    def test_no_barge_in_when_silent(self):
        """If agent is NOT speaking, check_barge_in should return False."""
        handler, stop_calls = self._make_handler()
        async def _run():
            return await handler.check_barge_in()
        result = asyncio.run(_run())
        assert result is False
        assert len(stop_calls) == 0
        print(f"\n  ✓ No barge-in when agent is silent")

    def test_barge_in_detected_when_speaking(self):
        """If agent IS speaking, check_barge_in should detect barge-in."""
        handler, stop_calls = self._make_handler()
        handler.set_agent_speaking(True, "You need a tourist visa for Japan.")

        async def _run():
            return await handler.check_barge_in()
        result = asyncio.run(_run())
        assert result is True
        assert len(stop_calls) == 1   # stop_audio was called
        assert handler.is_agent_speaking is False
        print(f"\n  ✓ Barge-in detected, stop_audio called")

    def test_recovery_context_early_interruption(self):
        """Early barge-in (< 20 chars) → fresh start message."""
        handler, _ = self._make_handler()
        handler.set_agent_speaking(True, "You need a visa.")
        handler._chars_spoken = 5   # Very early

        async def _run():
            await handler.check_barge_in()
        asyncio.run(_run())

        context = handler.get_recovery_context()
        assert "fresh" in context.lower() or "interrupt" in context.lower()
        print(f"\n  ✓ Early recovery context: '{context}'")

    def test_recovery_context_mid_response(self):
        """Mid-response barge-in → don't repeat partial message."""
        handler, _ = self._make_handler()
        handler.set_agent_speaking(True, "You need a tourist visa for Japan which costs 30 dollars.")
        handler._chars_spoken = 40  # Mid-way

        async def _run():
            await handler.check_barge_in()
        asyncio.run(_run())

        context = handler.get_recovery_context()
        assert "interrupted" in context.lower() or "repeat" in context.lower()
        print(f"\n  ✓ Mid recovery context: '{context[:60]}'")

    def test_recovery_context_consumed_after_use(self):
        """Recovery context is one-shot — cleared after get_recovery_context()."""
        handler, _ = self._make_handler()
        handler.set_agent_speaking(True, "Test response here.")

        async def _run():
            await handler.check_barge_in()
        asyncio.run(_run())

        ctx1 = handler.get_recovery_context()
        ctx2 = handler.get_recovery_context()
        assert ctx1 != ""    # First call returns context
        assert ctx2 == ""    # Second call returns empty
        print(f"\n  ✓ Recovery context is one-shot")

    def test_interruption_annotated_with_user_text(self):
        """complete_last_interruption stores what user said after barging in."""
        handler, _ = self._make_handler()
        handler.set_agent_speaking(True, "I was about to explain visa requirements.")

        async def _run():
            await handler.check_barge_in()
        asyncio.run(_run())

        handler.complete_last_interruption("Actually, just tell me the fee.")
        stats = handler.get_stats()
        assert stats["total_interruptions"] == 1
        assert stats["events"][0]["user_said"] == "Actually, just tell me the fee."
        print(f"\n  ✓ Interruption annotated: '{stats['events'][0]['user_said']}'")

    def test_multiple_interruptions_counted(self):
        """Multiple barge-ins are all recorded."""
        handler, _ = self._make_handler()

        async def _run():
            for _ in range(3):
                handler.set_agent_speaking(True, "Speaking...")
                await handler.check_barge_in()
                handler.get_recovery_context()  # consume
        asyncio.run(_run())

        stats = handler.get_stats()
        assert stats["total_interruptions"] == 3
        print(f"\n  ✓ {stats['total_interruptions']} interruptions recorded")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — ORCHESTRATOR INTERRUPTION CONTEXT TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestOrchestratorInterruption:
    def test_inject_context_stored(self):
        from backend.intelligence.orchestrator import ConversationOrchestrator
        orch = ConversationOrchestrator()
        orch.inject_interruption_context("User interrupted mid-answer about visa fees.")
        assert orch._interruption_context != ""

    def test_context_appears_in_prompt(self):
        from backend.intelligence.orchestrator import ConversationOrchestrator
        orch = ConversationOrchestrator()
        orch.inject_interruption_context("Test interruption note.")
        prompt = orch._build_system_prompt(rag_context="")
        assert "Interruption Note" in prompt
        assert "Test interruption note" in prompt
        print(f"\n  ✓ Interruption note injected into system prompt")

    def test_context_consumed_after_prompt_build(self):
        from backend.intelligence.orchestrator import ConversationOrchestrator
        orch = ConversationOrchestrator()
        orch.inject_interruption_context("One-shot note.")
        # First call includes it
        prompt1 = orch._build_system_prompt(rag_context="")
        assert "One-shot note" in prompt1
        # Second call does NOT include it
        prompt2 = orch._build_system_prompt(rag_context="")
        assert "One-shot note" not in prompt2
        print(f"\n  ✓ Interruption context consumed after one use")

    def test_orchestrator_responds_after_barge_in(self):
        """After injecting barge-in context, orchestrator still responds normally."""
        from backend.intelligence.orchestrator import ConversationOrchestrator
        async def _run():
            orch = ConversationOrchestrator()
            orch.inject_interruption_context("User interrupted. Don't repeat previous answer.")
            sentences = []
            async for s in orch.respond("What documents do I need for Japan?"):
                sentences.append(s)
            return sentences
        sentences = asyncio.run(_run())
        assert len(sentences) >= 1
        print(f"\n  ✓ Post-barge-in response: '{' '.join(sentences)[:80]}'")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
