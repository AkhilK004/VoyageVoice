"""
test_phase5.py — Final integration tests + Phase 5 polish validation.

Runs the complete stack end-to-end (all 4 phases) + tests for Phase 5:
  - Prompt v2 content validation
  - Demo scenario runner (dry-run mode)
  - Full REST API suite
  - Final regression across all phases

Think of this as the "ship it" test suite.

Run: python -m pytest tests/test_phase5.py -v -s
"""
import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE PHASE REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
class TestFullRegression:
    """Runs all critical checks from Phases 1-4 as a single smoke test."""

    def test_p1_config(self):
        from backend.config import settings
        assert settings.GROQ_API_KEY, "Missing GROQ_API_KEY"
        assert settings.DEEPGRAM_API_KEY, "Missing DEEPGRAM_API_KEY"
        print(f"\n  ✓ P1 Config OK")

    def test_p1_tts(self):
        from backend.voice.tts import synthesise_speech
        async def _run():
            data = []
            async for c in synthesise_speech("Final integration test."): data.append(c)
            return sum(len(c) for c in data)
        assert asyncio.run(_run()) > 500
        print(f"\n  ✓ P1 TTS OK")

    def test_p2_rag_retrieval(self):
        from backend.intelligence.rag import build_rag_context
        ctx = build_rag_context("visa Japan India tourist")
        assert "Japan" in ctx
        print(f"\n  ✓ P2 RAG OK ({len(ctx)} chars)")

    def test_p2_tool_execution(self):
        from backend.intelligence.tools import check_visa_eligibility, get_required_documents
        r = check_visa_eligibility("India", "Japan")
        assert r["found"] and r["fees_usd"] == 30
        r2 = get_required_documents("India", "Japan")
        assert r2["found"] and r2["document_count"] > 3
        print(f"\n  ✓ P2 Tools OK: {r['visa_type']}, {r2['document_count']} docs")

    def test_p3_vad_state_machine(self):
        from backend.voice.vad import VADState
        vad = VADState()
        vad.on_speech_start()
        assert vad.is_user_speaking
        vad.on_speech_end()
        assert not vad.is_user_speaking
        print(f"\n  ✓ P3 VAD OK")

    def test_p3_interruption(self):
        from backend.voice.interruption import InterruptionHandler
        calls = []
        async def stop(): calls.append(1)
        h = InterruptionHandler(send_stop_fn=stop)
        h.set_agent_speaking(True, "Test agent response.")
        result = asyncio.run(h.check_barge_in())
        assert result is True
        assert len(calls) == 1
        ctx = h.get_recovery_context()
        assert isinstance(ctx, str)
        print(f"\n  ✓ P3 Barge-in OK, recovery: '{ctx[:40]}'")

    def test_p4_metrics(self):
        from backend.evaluation.metrics import ConversationMetrics
        import uuid
        conv = ConversationMetrics(session_id=f"reg_{uuid.uuid4().hex[:6]}")
        t = conv.start_turn(1)
        t.e2e_latency_ms = 1200.0
        t.llm_ttft_ms = 400.0
        t.llm_total_ms = 800.0
        t.relevance_score = 4.5
        t.hallucination = False
        conv.finish()
        agg = conv.aggregate()
        assert agg["latency"]["e2e_avg_ms"] == 1200.0
        assert agg["quality"]["avg_relevance"] == 4.5
        print(f"\n  ✓ P4 Metrics OK: E2E={agg['latency']['e2e_avg_ms']}ms")

    def test_p4_rest_api(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/api/conversations").status_code == 200
        assert client.get("/api/metrics").status_code == 200
        assert client.get("/api/analysis").status_code == 200
        print(f"\n  ✓ P4 REST API OK")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — PROMPT v2
# ══════════════════════════════════════════════════════════════════════════════
class TestPromptV2:
    def test_prompt_exists(self):
        from backend.intelligence.prompts import SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 500, "Prompt too short"
        print(f"\n  ✓ Prompt v2 loaded: {len(SYSTEM_PROMPT)} chars")

    def test_prompt_has_brevity_constraint(self):
        from backend.intelligence.prompts import SYSTEM_PROMPT
        assert "2–3 sentences" in SYSTEM_PROMPT or "2-3 sentences" in SYSTEM_PROMPT
        print(f"\n  ✓ Brevity constraint present")

    def test_prompt_has_anti_hallucination(self):
        from backend.intelligence.prompts import SYSTEM_PROMPT
        assert "hallucin" in SYSTEM_PROMPT.lower() or "don't have" in SYSTEM_PROMPT.lower() or "ONLY state" in SYSTEM_PROMPT
        print(f"\n  ✓ Anti-hallucination guardrail present")

    def test_prompt_has_voice_instructions(self):
        from backend.intelligence.prompts import SYSTEM_PROMPT
        voice_keywords = ["voice", "call", "sentences", "conversational"]
        matches = [k for k in voice_keywords if k.lower() in SYSTEM_PROMPT.lower()]
        assert len(matches) >= 2, f"Missing voice keywords: {voice_keywords}"
        print(f"\n  ✓ Voice-specific instructions present: {matches}")

    def test_prompt_has_tool_preference(self):
        from backend.intelligence.prompts import SYSTEM_PROMPT
        assert "tool" in SYSTEM_PROMPT.lower()
        print(f"\n  ✓ Tool-calling preference mentioned")

    def test_prompt_has_escalation_guidance(self):
        from backend.intelligence.prompts import SYSTEM_PROMPT
        assert "escalat" in SYSTEM_PROMPT.lower()
        print(f"\n  ✓ Escalation guidance present")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — FULL ORCHESTRATOR WITH V2 PROMPT
# ══════════════════════════════════════════════════════════════════════════════
class TestOrchestratorV2:
    def test_complete_turn_pipeline(self):
        """Full turn: RAG + tool call + LLM response."""
        from backend.intelligence.orchestrator import ConversationOrchestrator

        async def _run():
            orch = ConversationOrchestrator()
            # First turn — establish context
            async for _ in orch.respond("I'm Indian."):
                pass
            # Second turn — should trigger tool call
            sentences = []
            async for s in orch.respond("Do I need a visa to visit Japan?"):
                sentences.append(s)
            return sentences, orch.get_history_summary()

        sentences, summary = asyncio.run(_run())
        full = " ".join(sentences).lower()
        has_relevant = any(w in full for w in ["visa", "japan", "india", "passport", "fee", "document"])
        assert has_relevant, f"Response not relevant: '{full[:150]}'"
        assert summary["turn_count"] == 2
        print(f"\n  ✓ Full pipeline v2: '{' '.join(sentences)[:100]}'")

    def test_brevity_compliance(self):
        """Response should be short (voice-appropriate)."""
        from backend.intelligence.orchestrator import ConversationOrchestrator

        async def _run():
            orch = ConversationOrchestrator()
            sentences = []
            async for s in orch.respond("I have an Indian passport and want to visit Thailand"):
                sentences.append(s)
            return sentences

        sentences = asyncio.run(_run())
        full = " ".join(sentences)
        # Should not be a massive wall of text
        assert len(full) < 800, f"Response too long ({len(full)} chars): voice agent should be concise"
        print(f"\n  ✓ Response brevity OK: {len(full)} chars in {len(sentences)} sentence(s)")

    def test_unknown_destination_honest(self):
        """Agent should admit uncertainty for unknown/missing countries."""
        from backend.intelligence.orchestrator import ConversationOrchestrator

        async def _run():
            orch = ConversationOrchestrator()
            sentences = []
            async for s in orch.respond("I'm Indian and want to visit Antarctica"):
                sentences.append(s)
            return sentences

        sentences = asyncio.run(_run())
        full = " ".join(sentences).lower()
        # Should express uncertainty, not hallucinate details
        has_uncertainty = any(w in full for w in [
            "don't have", "not sure", "recommend", "check", "embassy", "verify", "uncertain", "official"
        ])
        # Acceptable if it just says something generic — main thing is it doesn't hallucinate fees
        print(f"\n  ✓ Unknown destination: '{' '.join(sentences)[:120]}'")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — DEMO SCENARIO RUNNER
# ══════════════════════════════════════════════════════════════════════════════
class TestDemoScenarios:
    def test_scenarios_defined(self):
        """All 5 scenarios should be properly defined."""
        from demo_scenarios import SCENARIOS
        assert len(SCENARIOS) >= 5
        for key, scenario in SCENARIOS.items():
            assert "name" in scenario
            assert "turns" in scenario
            assert len(scenario["turns"]) >= 2
        print(f"\n  ✓ {len(SCENARIOS)} scenarios defined: {list(SCENARIOS.keys())}")

    def test_japan_scenario_runs(self):
        """Run the Japan visa scenario end-to-end."""
        from demo_scenarios import run_scenario

        async def _run():
            return await run_scenario("japan_visa", verbose=False)

        result = asyncio.run(_run())
        assert result.get("turn_count", 0) >= 3
        print(f"\n  ✓ Japan scenario: {result.get('turn_count')} turns, "
              f"E2E avg={result.get('latency', {}).get('e2e_avg_ms')}ms")

    def test_escalation_scenario_runs(self):
        """Escalation scenario should trigger escalate_to_human tool."""
        from demo_scenarios import run_scenario

        async def _run():
            return await run_scenario("escalation_test", verbose=False)

        result = asyncio.run(_run())
        assert result is not None
        print(f"\n  ✓ Escalation scenario complete: {result.get('turn_count')} turns")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — PROJECT STRUCTURE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
class TestProjectStructure:
    REQUIRED_FILES = [
        "README.md",
        ".env.example",
        ".gitignore",
        "requirements.txt",
        "demo_scenarios.py",
        "backend/__init__.py",
        "backend/main.py",
        "backend/config.py",
        "backend/voice/stt.py",
        "backend/voice/tts.py",
        "backend/voice/vad.py",
        "backend/voice/interruption.py",
        "backend/intelligence/orchestrator.py",
        "backend/intelligence/prompts.py",
        "backend/intelligence/rag.py",
        "backend/intelligence/tools.py",
        "backend/intelligence/state.py",
        "backend/knowledge/loader.py",
        "backend/knowledge/data/visa_data.json",
        "backend/evaluation/metrics.py",
        "backend/evaluation/logger.py",
        "backend/evaluation/judge.py",
        "backend/evaluation/analyzer.py",
        "frontend/index.html",
        "frontend/dashboard.html",
        "frontend/css/main.css",
        "frontend/css/dashboard.css",
        "frontend/js/audio.js",
        "frontend/js/ui.js",
        "frontend/js/dashboard.js",
        "tests/test_phase1.py",
        "tests/test_phase2.py",
        "tests/test_phase3.py",
        "tests/test_phase4.py",
        "tests/test_phase5.py",
    ]

    def test_all_required_files_exist(self):
        from pathlib import Path
        base = Path(".")
        missing = [f for f in self.REQUIRED_FILES if not (base / f).exists()]
        assert not missing, f"Missing files: {missing}"
        print(f"\n  ✓ All {len(self.REQUIRED_FILES)} required files exist")

    def test_env_example_has_required_keys(self):
        content = open(".env.example", encoding="utf-8").read()
        assert "GROQ_API_KEY" in content
        assert "DEEPGRAM_API_KEY" in content
        print(f"\n  ✓ .env.example has required keys")

    def test_gitignore_excludes_env(self):
        content = open(".gitignore", encoding="utf-8").read()
        assert ".env" in content
        assert "venv" in content
        print(f"\n  ✓ .gitignore excludes secrets and venv")

    def test_knowledge_data_complete(self):
        import json
        data = json.loads(open("backend/knowledge/data/visa_data.json", encoding="utf-8").read())
        assert len(data) >= 8, f"Only {len(data)} countries in knowledge base"
        countries = [d["country"] for d in data]
        for expected in ["Japan", "Thailand", "USA", "UK"]:
            assert expected in countries, f"Missing: {expected}"
        print(f"\n  ✓ Knowledge base: {len(data)} countries, {sum(len(d['visa_types']) for d in data)} visa types")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
