"""
demo_scenarios.py — Predefined demo scenarios for VoyageVoice.

Run this to simulate realistic conversations and generate evaluation data
without needing a microphone. Each scenario exercises a different pipeline path.

Usage:
    python demo_scenarios.py
    python demo_scenarios.py --scenario japan_visa
    python demo_scenarios.py --all

Output: Prints full conversation + metrics, saves session to evaluation/results/
"""
import asyncio
import argparse
import time
import json
from typing import Optional

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS = {
    "japan_visa": {
        "name": "India → Japan Visa Enquiry",
        "description": "Tests full RAG + tool calling pipeline for a common query",
        "turns": [
            "Hi, I'm planning a trip to Japan.",
            "I hold an Indian passport.",
            "Do I need a visa for Japan?",
            "What documents do I need to apply?",
            "How long does it take to process?",
            "Thanks, that's all I needed.",
        ],
    },
    "multi_destination": {
        "name": "Multi-destination Europe + USA",
        "description": "Tests state memory across multiple destination queries",
        "turns": [
            "Hello, I'm Indian and want to plan a big trip.",
            "First, can I travel to France visa-free?",
            "What about the UK?",
            "And lastly, what do I need for the USA?",
            "Which of these is easiest to get?",
        ],
    },
    "escalation_test": {
        "name": "Complex Case → Escalation",
        "description": "Tests escalation trigger when query is out of scope",
        "turns": [
            "Hi, my visa application was rejected last week.",
            "Application number AX-2024-19823.",
            "I need to know why it was rejected.",
        ],
    },
    "voa_query": {
        "name": "Thailand Visa-on-Arrival",
        "description": "Tests visa-on-arrival query with tool call",
        "turns": [
            "I'm travelling to Bangkok next month.",
            "I'm Indian. Can I get a visa on arrival in Thailand?",
            "How much does it cost?",
        ],
    },
    "unknown_country": {
        "name": "Knowledge Base Miss",
        "description": "Tests honest response when data is missing",
        "turns": [
            "I have an Indian passport and want to visit Bhutan.",
            "Do I need a visa?",
        ],
    },
}


# ── Simulation Engine ─────────────────────────────────────────────────────────
async def run_scenario(scenario_key: str, verbose: bool = True) -> dict:
    """
    Simulate a conversation scenario through the full pipeline.
    Returns the session metrics dict.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from backend.intelligence.orchestrator import ConversationOrchestrator
    from backend.intelligence.rag import build_rag_context
    from backend.evaluation.metrics import ConversationMetrics
    from backend.evaluation.logger import new_session_id, save_session
    from backend.evaluation.judge import score_turn_background

    scenario = SCENARIOS[scenario_key]
    session_id = f"demo_{scenario_key}_{new_session_id()}"

    print(f"\n{'='*60}")
    print(f"📋 SCENARIO: {scenario['name']}")
    print(f"   {scenario['description']}")
    print(f"   Session: {session_id}")
    print(f"{'='*60}")

    orchestrator = ConversationOrchestrator()
    conv_metrics = ConversationMetrics(session_id=session_id)
    judge_tasks = []

    for i, user_text in enumerate(scenario["turns"], 1):
        print(f"\n👤 Turn {i}: {user_text}")

        # Start turn metrics
        turn = conv_metrics.start_turn(i)
        turn.user_text = user_text
        turn.conversation_state = orchestrator.context.state.value

        # Simulate STT (immediate in demo)
        turn.stt_latency_ms = 280.0

        # RAG retrieval
        rag_ctx = build_rag_context(user_text)
        turn.rag_context_chars = len(rag_ctx)

        # LLM
        llm_start = time.monotonic()
        sentences = []
        first_token = True
        async for sentence in orchestrator.respond(user_text):
            if first_token:
                turn.llm_ttft_ms = (time.monotonic() - llm_start) * 1000
                first_token = False
            sentences.append(sentence)

        turn.llm_total_ms = (time.monotonic() - llm_start) * 1000
        turn.tts_first_byte_ms = 350.0          # Simulated
        turn.e2e_latency_ms = turn.llm_ttft_ms + turn.tts_first_byte_ms + turn.stt_latency_ms
        turn.tools_called = orchestrator.context.tools_called.copy()
        turn.agent_text = " ".join(sentences)

        agent_response = " ".join(sentences)
        print(f"🤖 Aria: {agent_response}")
        if verbose:
            print(f"   📊 STT:{turn.stt_latency_ms:.0f}ms | LLM-TTFT:{turn.llm_ttft_ms:.0f}ms | E2E:{turn.e2e_latency_ms:.0f}ms")
            if turn.tools_called:
                print(f"   🔧 Tools: {turn.tools_called}")
            if rag_ctx:
                print(f"   📚 RAG: {len(rag_ctx)} chars retrieved")

        # Fire-and-forget judge scoring
        task = asyncio.create_task(
            score_turn_background(turn, user_text, agent_response, rag_ctx)
        )
        judge_tasks.append(task)

        # Small delay between turns (realistic pacing)
        await asyncio.sleep(0.2)

    # Wait for all judge scores to complete
    print(f"\n⏳ Waiting for judge scoring...")
    await asyncio.gather(*judge_tasks, return_exceptions=True)

    conv_metrics.finish()
    conv_metrics.escalated = orchestrator.context.escalated
    save_session(conv_metrics)

    agg = conv_metrics.aggregate()

    print(f"\n{'='*60}")
    print(f"📈 SCENARIO RESULTS: {scenario['name']}")
    print(f"{'='*60}")
    print(f"   Turns:              {agg.get('turn_count', 0)}")
    print(f"   E2E Avg:            {agg.get('latency', {}).get('e2e_avg_ms', '—')}ms")
    print(f"   E2E P95:            {agg.get('latency', {}).get('e2e_p95_ms', '—')}ms")
    print(f"   LLM TTFT Avg:       {agg.get('latency', {}).get('llm_ttft_avg_ms', '—')}ms")
    print(f"   Avg Relevance:      {agg.get('quality', {}).get('avg_relevance', '—')} / 5")
    print(f"   Avg Naturalness:    {agg.get('quality', {}).get('avg_naturalness', '—')} / 5")
    print(f"   Hallucination Rate: {agg.get('quality', {}).get('hallucination_rate', '—')}")
    print(f"   Tools Called:       {agg.get('tools', {}).get('unique_tools', [])}")
    print(f"   Session saved: {session_id}")

    return agg


async def run_all_scenarios():
    results = {}
    for key in SCENARIOS:
        try:
            results[key] = await run_scenario(key)
        except Exception as e:
            print(f"\n❌ Scenario '{key}' failed: {e}")
            results[key] = {"error": str(e)}

    print(f"\n{'='*60}")
    print("🏁 ALL SCENARIOS COMPLETE")
    print(f"{'='*60}")
    for key, r in results.items():
        rel = r.get("quality", {}).get("avg_relevance", "—")
        e2e = r.get("latency", {}).get("e2e_avg_ms", "—")
        status = "✅" if isinstance(rel, float) and rel >= 3 else "⚠️"
        print(f"  {status} {SCENARIOS[key]['name']}: relevance={rel}, E2E={e2e}ms")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoyageVoice Demo Scenario Runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Run a specific scenario")
    group.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-turn metrics")
    args = parser.parse_args()

    if args.all:
        asyncio.run(run_all_scenarios())
    elif args.scenario:
        asyncio.run(run_scenario(args.scenario, verbose=not args.quiet))
    else:
        # Default: run the Japan scenario as a quick demo
        asyncio.run(run_scenario("japan_visa"))
