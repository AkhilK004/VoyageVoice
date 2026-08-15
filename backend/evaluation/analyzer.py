"""
analyzer.py — Failure pattern analysis across conversation logs.

Identifies common failure modes:
  - High latency turns (E2E > threshold)
  - Low relevance turns (score < 3)
  - Hallucinated facts
  - Turns that triggered escalation
  - Frequent user repetitions (sign the agent didn't understand)
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

HIGH_LATENCY_THRESHOLD_MS = 3000    # E2E > 3s is a "high latency" failure
LOW_RELEVANCE_THRESHOLD = 2.5       # Score < 2.5 is "poor response"


def analyze_sessions(sessions: list[dict]) -> dict:
    """
    Analyze a list of session dicts and return failure patterns.
    Each session dict is from ConversationMetrics.to_dict().
    """
    if not sessions:
        return {"patterns": [], "recommendations": []}

    all_turns = []
    for s in sessions:
        for t in s.get("turns", []):
            t["session_id"] = s.get("session_id", "unknown")
        all_turns.extend(s.get("turns", []))

    if not all_turns:
        return {"patterns": [], "recommendations": []}

    patterns = []
    recommendations = []

    # ── 1. High Latency Turns ────────────────────────────────────────────────
    high_lat = [
        t for t in all_turns
        if t["latencies"].get("e2e_ms") and t["latencies"]["e2e_ms"] > HIGH_LATENCY_THRESHOLD_MS
    ]
    if high_lat:
        pct = round(len(high_lat) / len(all_turns) * 100, 1)
        avg_lat = round(sum(t["latencies"]["e2e_ms"] for t in high_lat) / len(high_lat), 0)
        patterns.append({
            "type": "high_latency",
            "severity": "high" if pct > 20 else "medium",
            "affected_turns": len(high_lat),
            "percentage": pct,
            "avg_latency_ms": avg_lat,
            "description": f"{pct}% of turns had E2E latency > {HIGH_LATENCY_THRESHOLD_MS}ms (avg {avg_lat:.0f}ms)",
        })
        if avg_lat > 5000:
            recommendations.append("TTS latency is high — consider switching to a faster TTS voice or preloading common phrases.")
        else:
            recommendations.append("LLM latency spikes detected — consider using llama-3.1-8b-instant (already set) and reducing max_tokens.")

    # ── 2. Low Relevance Turns ───────────────────────────────────────────────
    scored = [t for t in all_turns if t["quality"].get("relevance_score") is not None]
    low_rel = [t for t in scored if t["quality"]["relevance_score"] < LOW_RELEVANCE_THRESHOLD]
    if low_rel and scored:
        pct = round(len(low_rel) / len(scored) * 100, 1)
        patterns.append({
            "type": "low_relevance",
            "severity": "high" if pct > 25 else "medium",
            "affected_turns": len(low_rel),
            "percentage": pct,
            "description": f"{pct}% of scored turns had relevance < {LOW_RELEVANCE_THRESHOLD}",
            "examples": [t["content"].get("user_text", "")[:80] for t in low_rel[:3]],
        })
        recommendations.append("Review system prompt — agent may be answering questions the user didn't ask. Strengthen 'ask clarifying questions first' instruction.")

    # ── 3. Hallucinations ────────────────────────────────────────────────────
    hallucinated = [t for t in scored if t["quality"].get("hallucination") is True]
    if hallucinated and scored:
        pct = round(len(hallucinated) / len(scored) * 100, 1)
        patterns.append({
            "type": "hallucination",
            "severity": "high" if pct > 10 else "low",
            "affected_turns": len(hallucinated),
            "percentage": pct,
            "description": f"{pct}% of scored turns had hallucinated facts",
            "examples": [t["content"].get("agent_text", "")[:100] for t in hallucinated[:3]],
        })
        recommendations.append("Hallucination detected — strengthen the 'Do NOT state facts not in the retrieved context' prompt instruction. Consider increasing RAG n_results from 3 to 5.")

    # ── 4. Zero RAG Context Turns ─────────────────────────────────────────────
    no_rag = [t for t in all_turns if t["content"].get("rag_context_chars", 0) == 0 and t["content"].get("user_text")]
    if no_rag and len(no_rag) / len(all_turns) > 0.3:
        pct = round(len(no_rag) / len(all_turns) * 100, 1)
        patterns.append({
            "type": "rag_miss",
            "severity": "medium",
            "affected_turns": len(no_rag),
            "percentage": pct,
            "description": f"{pct}% of turns had no RAG context retrieved — agent may hallucinate on these",
            "examples": [t["content"].get("user_text", "")[:80] for t in no_rag[:3]],
        })
        recommendations.append("Many turns have no RAG context — expand the knowledge base with more country data and FAQs.")

    # ── 5. Interrupted Turns ──────────────────────────────────────────────────
    interrupted = [t for t in all_turns if t.get("was_interrupted")]
    if interrupted:
        pct = round(len(interrupted) / len(all_turns) * 100, 1)
        patterns.append({
            "type": "interruptions",
            "severity": "low" if pct < 20 else "medium",
            "affected_turns": len(interrupted),
            "percentage": pct,
            "description": f"{pct}% of turns were interrupted by the user",
        })
        if pct > 30:
            recommendations.append("High interruption rate — agent responses may be too long. Reduce max_tokens or add a 'keep responses under 2 sentences' instruction.")

    if not patterns:
        patterns.append({
            "type": "healthy",
            "severity": "none",
            "description": "No significant failure patterns detected. System looks healthy!",
        })
        recommendations.append("System performing well. Consider testing with diverse accents and background noise next.")

    return {
        "total_sessions": len(sessions),
        "total_turns": len(all_turns),
        "patterns": patterns,
        "recommendations": recommendations,
    }
