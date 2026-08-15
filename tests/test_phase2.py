"""
test_phase2.py — Integration tests for Phase 2: RAG + Tool Calling + State Machine.

Also re-validates all Phase 1 components are still working.

Tests:
  Phase 1 regression:
    - TTS still works
    - Orchestrator still streams sentences
  Phase 2:
    - Knowledge base loads and returns passages
    - RAG retrieves relevant results for visa queries
    - Tools execute correctly with real data
    - State machine extracts entities and transitions correctly
    - Full orchestrator pipeline: RAG + tool call + response

Run: python -m pytest tests/test_phase2.py -v
"""
import asyncio
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.config import settings


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 REGRESSION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPhase1Regression:
    """Ensure Phase 1 components still work after Phase 2 changes."""

    def test_config_still_valid(self):
        assert settings.GROQ_API_KEY, "GROQ_API_KEY missing"
        assert settings.DEEPGRAM_API_KEY, "DEEPGRAM_API_KEY missing"

    def test_tts_still_generates_audio(self):
        from backend.voice.tts import synthesise_speech

        async def _run():
            chunks = []
            async for chunk in synthesise_speech("Visa test audio."):
                chunks.append(chunk)
            return sum(len(c) for c in chunks)

        total = asyncio.run(_run())
        assert total > 500, f"TTS regression: only {total} bytes"
        print(f"\n  ✓ TTS regression OK: {total} bytes")

    def test_orchestrator_still_streams(self):
        """Phase 1 basic LLM streaming still works (now with Phase 2 plumbing)."""
        from backend.intelligence.orchestrator import ConversationOrchestrator

        async def _run():
            orch = ConversationOrchestrator()
            sentences = []
            async for s in orch.respond("Hello, what can you help me with?"):
                sentences.append(s)
            return sentences

        sentences = asyncio.run(_run())
        assert len(sentences) >= 1
        print(f"\n  ✓ LLM regression OK: {len(sentences)} sentence(s)")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: RAG TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestKnowledgeBase:
    def test_visa_data_file_exists(self):
        from pathlib import Path
        path = Path("backend/knowledge/data/visa_data.json")
        assert path.exists(), "visa_data.json not found"
        data = json.loads(path.read_text())
        assert len(data) >= 5, f"Expected at least 5 countries, got {len(data)}"
        print(f"\n  ✓ visa_data.json loaded with {len(data)} countries")

    def test_knowledge_base_loads(self):
        """ChromaDB collection loads and has entries."""
        from backend.knowledge.loader import load_knowledge_base
        collection = load_knowledge_base()
        count = collection.count()
        assert count > 0, "Knowledge base is empty"
        print(f"\n  ✓ Knowledge base loaded: {count} passages")

    def test_rag_retrieval_japan(self):
        """RAG retrieves Japan-related content for a Japan query."""
        from backend.knowledge.loader import load_knowledge_base, retrieve
        collection = load_knowledge_base()
        results = retrieve(collection, "Do Indians need a visa for Japan?", n_results=3)
        assert len(results) > 0, "RAG returned no results"
        countries = [r["country"] for r in results]
        assert "Japan" in countries, f"Expected Japan in results, got: {countries}"
        print(f"\n  ✓ RAG retrieved: {[(r['country'], r['visa_type']) for r in results]}")

    def test_rag_context_builder(self):
        """build_rag_context returns a non-empty string for travel queries."""
        from backend.intelligence.rag import build_rag_context
        context = build_rag_context("visa requirements for Thailand from India")
        assert isinstance(context, str)
        assert len(context) > 50, "RAG context is too short"
        print(f"\n  ✓ RAG context: {len(context)} chars, starts with: '{context[:60]}...'")

    def test_rag_empty_query_no_crash(self):
        """Empty/garbage query should return empty string, not crash."""
        from backend.intelligence.rag import build_rag_context
        context = build_rag_context("xyz 123 gibberish")
        assert isinstance(context, str)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: TOOL CALLING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTools:
    def test_check_visa_eligibility_india_japan(self):
        from backend.intelligence.tools import check_visa_eligibility
        result = check_visa_eligibility("India", "Japan")
        assert result["found"] is True
        assert result["country"] == "Japan"
        assert "usd" in str(result["fees_usd"]).lower() or result["fees_usd"] is not None
        print(f"\n  ✓ Visa eligibility India→Japan: {result['visa_type']}, ${result['fees_usd']}")

    def test_get_required_documents_india_usa(self):
        from backend.intelligence.tools import get_required_documents
        result = get_required_documents("India", "USA")
        assert result["found"] is True
        assert len(result["documents"]) > 3
        print(f"\n  ✓ Documents India→USA: {result['document_count']} docs")

    def test_get_processing_time_india_schengen(self):
        from backend.intelligence.tools import get_processing_time
        result = get_processing_time("India", "Schengen")
        assert result["found"] is True
        assert result["fees_usd"] is not None
        print(f"\n  ✓ Processing time India→Schengen: {result['processing_time']} days, ${result['fees_usd']}")

    def test_escalate_to_human(self):
        from backend.intelligence.tools import escalate_to_human
        result = escalate_to_human("Complex visa issue")
        assert result["escalated"] is True
        print(f"\n  ✓ Escalation: {result['message'][:50]}")

    def test_unknown_destination_returns_not_found(self):
        from backend.intelligence.tools import check_visa_eligibility
        result = check_visa_eligibility("India", "Mars")
        assert result["found"] is False

    def test_tool_schema_structure(self):
        from backend.intelligence.tools import TOOL_SCHEMAS
        assert len(TOOL_SCHEMAS) == 4
        for schema in TOOL_SCHEMAS:
            assert "function" in schema
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]
        print(f"\n  ✓ {len(TOOL_SCHEMAS)} tool schemas valid")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: STATE MACHINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestStateMachine:
    def test_initial_state_is_greeting(self):
        from backend.intelligence.state import ConversationContext, ConversationState
        ctx = ConversationContext()
        assert ctx.state == ConversationState.GREETING

    def test_entity_extraction_passport(self):
        from backend.intelligence.state import ConversationContext
        ctx = ConversationContext()
        ctx.extract_entities_from_text("I'm Indian and I want to visit Japan")
        assert ctx.passport_country == "India", f"Got: {ctx.passport_country}"
        assert ctx.destination == "Japan", f"Got: {ctx.destination}"
        print(f"\n  ✓ Entities: passport={ctx.passport_country}, dest={ctx.destination}")

    def test_has_enough_info(self):
        from backend.intelligence.state import ConversationContext
        ctx = ConversationContext()
        assert not ctx.has_enough_info()
        ctx.passport_country = "India"
        assert not ctx.has_enough_info()
        ctx.destination = "Japan"
        assert ctx.has_enough_info()

    def test_state_transition(self):
        from backend.intelligence.state import ConversationContext, ConversationState
        ctx = ConversationContext()
        ctx.advance_state(ConversationState.UNDERSTANDING)
        assert ctx.state == ConversationState.UNDERSTANDING
        ctx.advance_state(ConversationState.ANSWERING)
        assert ctx.state == ConversationState.ANSWERING

    def test_missing_info_prompt(self):
        from backend.intelligence.state import ConversationContext
        ctx = ConversationContext()
        prompt = ctx.missing_info_prompt()
        assert "passport" in prompt.lower() or "country" in prompt.lower()
        ctx.passport_country = "India"
        prompt = ctx.missing_info_prompt()
        assert "visit" in prompt.lower() or "country" in prompt.lower()
        print(f"\n  ✓ Missing info prompt: '{prompt}'")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: FULL ORCHESTRATOR PIPELINE TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipelinePhase2:
    def test_orchestrator_answers_visa_query(self):
        """
        Full pipeline: 'I'm Indian, can I travel to Japan?' →
        Should trigger RAG + tool call + answer about visa.
        """
        from backend.intelligence.orchestrator import ConversationOrchestrator

        async def _run():
            orch = ConversationOrchestrator()
            sentences = []
            async for s in orch.respond("I'm Indian and I want to visit Japan. Do I need a visa?"):
                sentences.append(s)
            return sentences, orch.get_history_summary()

        sentences, summary = asyncio.run(_run())
        assert len(sentences) >= 1
        full_text = " ".join(sentences).lower()

        # Should mention visa, Japan, or India in the response
        has_relevant_content = any(
            keyword in full_text
            for keyword in ["visa", "japan", "india", "passport", "document", "fee", "days"]
        )
        assert has_relevant_content, f"Response doesn't seem relevant: '{full_text[:150]}'"
        print(f"\n  ✓ Full pipeline response: '{' '.join(sentences)[:120]}...'")
        print(f"  ✓ Summary: {summary}")

    def test_orchestrator_multi_turn_memory(self):
        """Multi-turn: first message sets context, second uses it."""
        from backend.intelligence.orchestrator import ConversationOrchestrator

        async def _run():
            orch = ConversationOrchestrator()
            # Turn 1: establish context
            async for _ in orch.respond("I hold an Indian passport."):
                pass
            # Turn 2: use context
            sentences = []
            async for s in orch.respond("I want to visit Thailand. Do I need a visa?"):
                sentences.append(s)
            return orch.get_history_summary(), sentences

        summary, sentences = asyncio.run(_run())
        assert summary["turn_count"] == 2
        assert len(sentences) >= 1
        print(f"\n  ✓ Multi-turn: {summary['turn_count']} turns, state={summary['state']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
