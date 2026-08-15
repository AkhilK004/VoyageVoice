"""
logger.py — Conversation logger: saves full conversation data to disk as JSON.

Every call session is saved as:
  evaluation/results/<session_id>.json

This enables:
  - Post-call LLM-as-judge scoring
  - Failure pattern analysis
  - Dashboard data source
  - Conversation replay
"""
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime

from backend.evaluation.metrics import ConversationMetrics

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent.parent / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store so the dashboard can read recent sessions without disk I/O
_session_cache: dict[str, dict] = {}


def new_session_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = str(uuid.uuid4())[:8]
    return f"{ts}_{short}"


def save_session(conv_metrics: ConversationMetrics) -> Path:
    """
    Serialise and save the conversation to disk.
    Also updates the in-memory cache.
    """
    data = conv_metrics.to_dict()
    _session_cache[conv_metrics.session_id] = data

    path = RESULTS_DIR / f"{conv_metrics.session_id}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info(f"Session saved: {path.name}")
    return path


def load_all_sessions() -> list[dict]:
    """Load all saved session files + any in the cache."""
    sessions = {}

    # From cache first (most recent calls)
    for sid, data in _session_cache.items():
        sessions[sid] = data

    # From disk (previous runs)
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True)[:50]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sid = data.get("session_id", path.stem)
            if sid not in sessions:
                sessions[sid] = data
        except Exception as e:
            logger.warning(f"Could not load {path.name}: {e}")

    # Return sorted newest first
    return sorted(sessions.values(), key=lambda x: x.get("start_time", 0), reverse=True)


def load_session(session_id: str) -> dict | None:
    """Load a specific session by ID."""
    if session_id in _session_cache:
        return _session_cache[session_id]

    path = RESULTS_DIR / f"{session_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
    return None


def compute_global_aggregate(sessions: list[dict]) -> dict:
    """
    Compute aggregate metrics across ALL sessions for dashboard header cards.
    """
    if not sessions:
        return {}

    all_turns = []
    for s in sessions:
        all_turns.extend(s.get("turns", []))

    if not all_turns:
        return {}

    def avg(vals):
        valid = [v for v in vals if v is not None and v > 0]
        return round(sum(valid) / len(valid), 1) if valid else None

    def p95(vals):
        valid = sorted([v for v in vals if v is not None and v > 0])
        if not valid:
            return None
        return round(valid[int(len(valid) * 0.95)], 1)

    e2e = [t["latencies"]["e2e_ms"] for t in all_turns]
    stt = [t["latencies"]["stt_ms"] for t in all_turns]
    llm = [t["latencies"]["llm_ttft_ms"] for t in all_turns]
    tts = [t["latencies"]["tts_first_byte_ms"] for t in all_turns]
    rel = [t["quality"]["relevance_score"] for t in all_turns if t["quality"]["relevance_score"]]
    nat = [t["quality"]["naturalness_score"] for t in all_turns if t["quality"]["naturalness_score"]]
    hallucinated = [t for t in all_turns if t["quality"].get("hallucination")]

    total_interruptions = sum(s.get("aggregate", {}).get("total_interruptions", 0) for s in sessions)
    total_escalations = sum(1 for s in sessions if s.get("aggregate", {}).get("escalated"))

    return {
        "total_sessions": len(sessions),
        "total_turns": len(all_turns),
        "total_interruptions": total_interruptions,
        "total_escalations": total_escalations,
        "latency": {
            "e2e_avg_ms": avg(e2e),
            "e2e_p95_ms": p95(e2e),
            "stt_avg_ms": avg(stt),
            "llm_ttft_avg_ms": avg(llm),
            "tts_avg_ms": avg(tts),
        },
        "quality": {
            "avg_relevance": avg(rel),
            "avg_naturalness": avg(nat),
            "hallucination_rate": round(len(hallucinated) / len(all_turns), 3) if all_turns else 0,
            "scored_turns": len(rel),
        },
        # Per-turn data for charts (last 100 turns)
        "chart_data": {
            "e2e_series": [v for v in e2e[-100:] if v and v > 0],
            "stt_series": [v for v in stt[-100:] if v and v > 0],
            "llm_series": [v for v in llm[-100:] if v and v > 0],
            "relevance_series": rel[-100:],
        },
    }
