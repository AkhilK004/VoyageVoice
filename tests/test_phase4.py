"""
test_phase4.py — Integration tests for Phase 4: Evaluation Dashboard.

Also runs Phase 1 + 2 + 3 regressions.

Tests:
  Regressions (P1–P3):
    - Config, TTS, Orchestrator, RAG, Tools, State, VAD, Interruption

  Phase 4:
    - TurnMetrics serialises correctly
    - ConversationMetrics.aggregate() computes correctly
    - Session logger saves/loads/lists
    - LLM judge returns valid scores
    - Failure analyzer detects patterns correctly
    - REST API endpoints respond correctly

Run: python -m pytest tests/test_phase4.py -v -s
"""
import asyncio
import json
import os
import sys
import time
import uuid
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1–3 REGRESSION (smoke tests)
# ══════════════════════════════════════════════════════════════════════════════
class TestRegressions:
    def test_config_loads(self):
        from backend.config import settings
        assert settings.GROQ_API_KEY and settings.DEEPGRAM_API_KEY

    def test_tts_works(self):
        from backend.voice.tts import synthesise_speech
        async def _run():
            chunks = []
            async for c in synthesise_speech("Phase four test."):
                chunks.append(c)
            return sum(len(c) for c in chunks)
        assert asyncio.run(_run()) > 500

    def test_orchestrator_works(self):
        from backend.intelligence.orchestrator import ConversationOrchestrator
        async def _run():
            orch = ConversationOrchestrator()
            out = []
            async for s in orch.respond("Hello"):
                out.append(s)
            return out
        assert len(asyncio.run(_run())) >= 1

    def test_rag_works(self):
        from backend.intelligence.rag import build_rag_context
        ctx = build_rag_context("visa Japan India")
        assert len(ctx) > 20

    def test_vad_works(self):
        from backend.voice.vad import VADState
        vad = VADState()
        vad.on_speech_start()
        assert vad.is_user_speaking

    def test_interruption_works(self):
        from backend.voice.interruption import InterruptionHandler
        calls = []
        async def mock_stop(): calls.append(1)
        h = InterruptionHandler(send_stop_fn=mock_stop)
        h.set_agent_speaking(True, "Test.")
        result = asyncio.run(h.check_barge_in())
        assert result is True
        assert len(calls) == 1


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
class TestMetrics:
    def _make_conv(self, n_turns=3) -> "ConversationMetrics":
        from backend.evaluation.metrics import ConversationMetrics, TurnMetrics
        sess_id = f"test_{uuid.uuid4().hex[:8]}"
        conv = ConversationMetrics(session_id=sess_id)
        for i in range(1, n_turns + 1):
            t = conv.start_turn(i)
            t.user_text = f"Question {i}"
            t.agent_text = f"Answer {i}"
            t.stt_latency_ms = 250 + i * 10
            t.llm_ttft_ms = 400 + i * 20
            t.llm_total_ms = 800 + i * 30
            t.tts_first_byte_ms = 300 + i * 15
            t.e2e_latency_ms = 1200 + i * 40
            t.relevance_score = 4.0
            t.naturalness_score = 3.5
            t.hallucination = False
        conv.finish()
        return conv

    def test_turn_metrics_serialises(self):
        from backend.evaluation.metrics import TurnMetrics
        t = TurnMetrics(turn_id=1)
        t.user_text = "Do I need a visa?"
        t.stt_latency_ms = 300.0
        d = t.to_dict()
        assert d["turn_id"] == 1
        assert d["latencies"]["stt_ms"] == 300.0
        assert "content" in d
        assert "quality" in d
        print(f"\n  ✓ TurnMetrics serialised: {list(d.keys())}")

    def test_conversation_aggregate(self):
        conv = self._make_conv(3)
        agg = conv.aggregate()
        assert agg["turn_count"] == 3
        assert agg["latency"]["e2e_avg_ms"] is not None
        assert agg["quality"]["avg_relevance"] == 4.0
        assert agg["quality"]["hallucination_rate"] == 0.0
        print(f"\n  ✓ Aggregate: E2E avg={agg['latency']['e2e_avg_ms']}ms, rel={agg['quality']['avg_relevance']}")

    def test_p95_computation(self):
        """P95 should be at or near the highest values."""
        conv = self._make_conv(10)
        agg = conv.aggregate()
        e2e_avg = agg["latency"]["e2e_avg_ms"]
        e2e_p95 = agg["latency"]["e2e_p95_ms"]
        assert e2e_p95 >= e2e_avg, "P95 should be >= avg"
        print(f"\n  ✓ P95 >= avg: {e2e_avg}ms avg, {e2e_p95}ms p95")

    def test_empty_session_aggregate(self):
        from backend.evaluation.metrics import ConversationMetrics
        conv = ConversationMetrics(session_id="empty")
        conv.finish()
        agg = conv.aggregate()
        assert agg == {}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — LOGGER
# ══════════════════════════════════════════════════════════════════════════════
class TestLogger:
    def _make_and_save(self):
        from backend.evaluation.metrics import ConversationMetrics
        from backend.evaluation.logger import save_session, new_session_id
        sid = new_session_id()
        conv = ConversationMetrics(session_id=sid)
        t = conv.start_turn(1)
        t.user_text = "Test question"
        t.agent_text = "Test answer"
        t.llm_total_ms = 500.0
        t.relevance_score = 4.0
        conv.finish()
        save_session(conv)
        return sid

    def test_session_id_format(self):
        from backend.evaluation.logger import new_session_id
        sid = new_session_id()
        assert len(sid) > 10
        assert "_" in sid
        print(f"\n  ✓ Session ID format: {sid}")

    def test_save_and_load_session(self):
        from backend.evaluation.logger import load_session
        sid = self._make_and_save()
        loaded = load_session(sid)
        assert loaded is not None
        assert loaded["session_id"] == sid
        assert len(loaded["turns"]) == 1
        assert loaded["turns"][0]["content"]["user_text"] == "Test question"
        print(f"\n  ✓ Session saved and loaded: {sid}")

    def test_list_all_sessions_includes_saved(self):
        from backend.evaluation.logger import load_all_sessions
        sid = self._make_and_save()
        sessions = load_all_sessions()
        ids = [s["session_id"] for s in sessions]
        assert sid in ids
        print(f"\n  ✓ Session appears in list: {len(sessions)} total")

    def test_compute_global_aggregate(self):
        from backend.evaluation.logger import load_all_sessions, compute_global_aggregate
        self._make_and_save()
        sessions = load_all_sessions()
        agg = compute_global_aggregate(sessions)
        assert "total_sessions" in agg
        assert "latency" in agg
        assert "quality" in agg
        print(f"\n  ✓ Global aggregate: {agg['total_sessions']} sessions, {agg['total_turns']} turns")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — LLM JUDGE
# ══════════════════════════════════════════════════════════════════════════════
class TestJudge:
    def test_score_turn_returns_valid_structure(self):
        from backend.evaluation.judge import score_turn
        async def _run():
            return await score_turn(
                user_text="Do Indians need a visa for Japan?",
                agent_text="Yes, Indian passport holders need a tourist visa for Japan. It costs around 30 USD and takes 5-7 business days to process.",
                rag_context="Country: Japan. Tourist Visa. Fee: USD 30. Processing time: 5-7 business days.",
            )
        result = asyncio.run(_run())
        assert "relevance" in result
        assert "naturalness" in result
        assert "hallucination" in result
        if result["relevance"] is not None:
            assert 1 <= result["relevance"] <= 5
        if result["naturalness"] is not None:
            assert 1 <= result["naturalness"] <= 5
        if result["hallucination"] is not None:
            assert isinstance(result["hallucination"], bool)
        print(f"\n  ✓ Judge scores: {result}")

    def test_score_turn_empty_no_crash(self):
        from backend.evaluation.judge import score_turn
        result = asyncio.run(score_turn("", ""))
        assert "relevance" in result
        print(f"\n  ✓ Empty turn handled gracefully")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — FAILURE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
class TestAnalyzer:
    def _make_sessions_with_failures(self) -> list[dict]:
        """Create synthetic session data with clear failure patterns."""
        turns_high_latency = [
            {"latencies": {"e2e_ms": 5000, "stt_ms": 400, "llm_ttft_ms": 500, "tts_first_byte_ms": 300},
             "content": {"user_text": "Q", "agent_text": "A", "rag_context_chars": 100},
             "quality": {"relevance_score": 4.0, "naturalness_score": 3.5, "hallucination": False},
             "state": "answering", "was_interrupted": False}
            for _ in range(8)
        ]
        turns_normal = [
            {"latencies": {"e2e_ms": 1200, "stt_ms": 300, "llm_ttft_ms": 400, "tts_first_byte_ms": 250},
             "content": {"user_text": "Q", "agent_text": "A", "rag_context_chars": 200},
             "quality": {"relevance_score": 4.5, "naturalness_score": 4.0, "hallucination": False},
             "state": "answering", "was_interrupted": False}
            for _ in range(2)
        ]
        return [{
            "session_id": "test_analysis",
            "start_time": time.time(),
            "turns": turns_high_latency + turns_normal,
            "aggregate": {"duration_seconds": 120, "turn_count": 10, "total_interruptions": 2, "escalated": False,
                          "latency": {"e2e_avg_ms": 4200, "e2e_p95_ms": 5000},
                          "quality": {"avg_relevance": 4.1, "hallucination_rate": 0.0, "scored_turns": 10}},
        }]

    def test_detects_high_latency(self):
        from backend.evaluation.analyzer import analyze_sessions
        sessions = self._make_sessions_with_failures()
        result = analyze_sessions(sessions)
        pattern_types = [p["type"] for p in result["patterns"]]
        assert "high_latency" in pattern_types, f"Expected high_latency, got: {pattern_types}"
        print(f"\n  ✓ High latency pattern detected: {pattern_types}")

    def test_detects_hallucination(self):
        from backend.evaluation.analyzer import analyze_sessions
        turns = [
            {"latencies": {"e2e_ms": 1000, "stt_ms": 200, "llm_ttft_ms": 300, "tts_first_byte_ms": 200},
             "content": {"user_text": "Q", "agent_text": "A", "rag_context_chars": 100},
             "quality": {"relevance_score": 2.0, "naturalness_score": 3.0, "hallucination": True},
             "state": "answering", "was_interrupted": False}
            for _ in range(5)
        ]
        sessions = [{"session_id": "hall_test", "start_time": time.time(), "turns": turns,
                     "aggregate": {"total_interruptions": 0, "escalated": False}}]
        result = analyze_sessions(sessions)
        pattern_types = [p["type"] for p in result["patterns"]]
        assert "hallucination" in pattern_types
        print(f"\n  ✓ Hallucination pattern detected")

    def test_healthy_system_no_critical_patterns(self):
        from backend.evaluation.analyzer import analyze_sessions
        turns = [
            {"latencies": {"e2e_ms": 900, "stt_ms": 200, "llm_ttft_ms": 350, "tts_first_byte_ms": 200},
             "content": {"user_text": "Q", "agent_text": "A", "rag_context_chars": 300},
             "quality": {"relevance_score": 4.5, "naturalness_score": 4.0, "hallucination": False},
             "state": "answering", "was_interrupted": False}
            for _ in range(10)
        ]
        sessions = [{"session_id": "healthy", "start_time": time.time(), "turns": turns,
                     "aggregate": {"total_interruptions": 0, "escalated": False}}]
        result = analyze_sessions(sessions)
        high_sev = [p for p in result["patterns"] if p.get("severity") == "high"]
        assert len(high_sev) == 0, f"Healthy system should have no high severity: {high_sev}"
        print(f"\n  ✓ Healthy system: no high-severity patterns")

    def test_empty_sessions_no_crash(self):
        from backend.evaluation.analyzer import analyze_sessions
        result = analyze_sessions([])
        assert "patterns" in result
        assert "recommendations" in result


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — REST API
# ══════════════════════════════════════════════════════════════════════════════
class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        print(f"\n  ✓ /health: {r.json()}")

    def test_conversations_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        r = client.get("/api/conversations")
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data
        assert "total" in data
        print(f"\n  ✓ /api/conversations: {data['total']} sessions")

    def test_metrics_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        r = client.get("/api/metrics")
        assert r.status_code == 200
        print(f"\n  ✓ /api/metrics: {list(r.json().keys())}")

    def test_analysis_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        r = client.get("/api/analysis")
        assert r.status_code == 200
        data = r.json()
        assert "patterns" in data
        assert "recommendations" in data
        print(f"\n  ✓ /api/analysis: {len(data['patterns'])} pattern(s)")

    def test_missing_session_404(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        r = client.get("/api/conversations/nonexistent_session_12345")
        assert r.status_code == 404
        print(f"\n  ✓ /api/conversations/nonexistent → 404")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
