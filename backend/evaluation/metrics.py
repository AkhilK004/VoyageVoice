"""
metrics.py — Per-turn latency and quality metric collection.

Tracks timing across the full pipeline:
  STT latency    → audio received to transcript ready
  LLM latency    → prompt sent to first token (TTFT)
  TTS latency    → text sent to first audio byte
  E2E latency    → user stops speaking to agent starts speaking
  Interruptions  → count and timing
  Tool calls     → which tools were called per turn
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnMetrics:
    """Metrics for a single conversation turn."""
    turn_id: int
    timestamp: float = field(default_factory=time.time)

    # Latencies (milliseconds)
    stt_latency_ms: Optional[float] = None     # audio → transcript
    llm_ttft_ms: Optional[float] = None        # prompt → first token
    llm_total_ms: Optional[float] = None       # prompt → last token
    tts_first_byte_ms: Optional[float] = None  # text → first audio byte
    e2e_latency_ms: Optional[float] = None     # speech_end → first audio out

    # Content
    user_text: str = ""
    agent_text: str = ""
    rag_context_chars: int = 0
    tools_called: list = field(default_factory=list)

    # Quality (set by judge after the turn)
    relevance_score: Optional[float] = None    # 1–5
    hallucination: Optional[bool] = None       # True if hallucinated
    naturalness_score: Optional[float] = None  # 1–5

    # State
    conversation_state: str = ""
    was_interrupted: bool = False

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "latencies": {
                "stt_ms": round(self.stt_latency_ms or 0, 1),
                "llm_ttft_ms": round(self.llm_ttft_ms or 0, 1),
                "llm_total_ms": round(self.llm_total_ms or 0, 1),
                "tts_first_byte_ms": round(self.tts_first_byte_ms or 0, 1),
                "e2e_ms": round(self.e2e_latency_ms or 0, 1),
            },
            "content": {
                "user_text": self.user_text,
                "agent_text": self.agent_text,
                "rag_context_chars": self.rag_context_chars,
                "tools_called": self.tools_called,
            },
            "quality": {
                "relevance_score": self.relevance_score,
                "hallucination": self.hallucination,
                "naturalness_score": self.naturalness_score,
            },
            "state": self.conversation_state,
            "was_interrupted": self.was_interrupted,
        }


@dataclass
class ConversationMetrics:
    """Aggregate metrics for a full conversation session."""
    session_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    turns: list[TurnMetrics] = field(default_factory=list)
    total_interruptions: int = 0
    escalated: bool = False

    def start_turn(self, turn_id: int) -> TurnMetrics:
        t = TurnMetrics(turn_id=turn_id)
        self.turns.append(t)
        return t

    def current_turn(self) -> Optional[TurnMetrics]:
        return self.turns[-1] if self.turns else None

    def finish(self):
        self.end_time = time.time()

    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def aggregate(self) -> dict:
        """Compute aggregate stats across all turns."""
        completed = [t for t in self.turns if t.llm_total_ms is not None]
        if not completed:
            return {}

        def avg(vals):
            valid = [v for v in vals if v is not None and v > 0]
            return round(sum(valid) / len(valid), 1) if valid else None

        def p95(vals):
            valid = sorted([v for v in vals if v is not None and v > 0])
            if not valid:
                return None
            idx = int(len(valid) * 0.95)
            return round(valid[min(idx, len(valid)-1)], 1)

        stt_vals = [t.stt_latency_ms for t in completed]
        llm_vals = [t.llm_ttft_ms for t in completed]
        tts_vals = [t.tts_first_byte_ms for t in completed]
        e2e_vals = [t.e2e_latency_ms for t in completed]

        scored = [t for t in completed if t.relevance_score is not None]
        avg_relevance = avg([t.relevance_score for t in scored])
        avg_naturalness = avg([t.naturalness_score for t in scored])
        hallucination_rate = (
            sum(1 for t in scored if t.hallucination) / len(scored)
            if scored else None
        )
        tools_all = [tool for t in completed for tool in t.tools_called]

        return {
            "session_id": self.session_id,
            "duration_seconds": round(self.duration_seconds(), 1),
            "turn_count": len(completed),
            "total_interruptions": self.total_interruptions,
            "escalated": self.escalated,
            "latency": {
                "stt_avg_ms": avg(stt_vals),
                "stt_p95_ms": p95(stt_vals),
                "llm_ttft_avg_ms": avg(llm_vals),
                "llm_ttft_p95_ms": p95(llm_vals),
                "tts_avg_ms": avg(tts_vals),
                "tts_p95_ms": p95(tts_vals),
                "e2e_avg_ms": avg(e2e_vals),
                "e2e_p95_ms": p95(e2e_vals),
            },
            "quality": {
                "avg_relevance": avg_relevance,
                "avg_naturalness": avg_naturalness,
                "hallucination_rate": round(hallucination_rate, 3) if hallucination_rate is not None else None,
                "scored_turns": len(scored),
            },
            "tools": {
                "total_calls": len(tools_all),
                "unique_tools": list(set(tools_all)),
            },
        }

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "turns": [t.to_dict() for t in self.turns],
            "aggregate": self.aggregate(),
        }
