"""
judge.py — LLM-as-Judge evaluation for VoyageVoice.

Uses Groq (Llama 3, free) to score each agent turn on:
  - Relevance  (1–5): Does the response address the user's question?
  - Naturalness (1–5): Does it sound like natural speech, not text?
  - Hallucination (bool): Did the agent state something not in the RAG context?

Judge runs asynchronously after each turn so it never adds to E2E latency.
"""
import asyncio
import json
import logging
from openai import AsyncOpenAI
from backend.config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

JUDGE_PROMPT = """You are an expert evaluator for a voice AI travel assistant.
Evaluate the following conversation turn and return a JSON object with these exact fields:
{{
  "relevance": <integer 1-5>,
  "naturalness": <integer 1-5>,
  "hallucination": <true or false>,
  "reasoning": "<one sentence explanation>"
}}

Scoring rubrics:
- relevance: 1=completely off-topic, 3=partially answers, 5=fully addresses the question
- naturalness: 1=reads like a document, 3=somewhat natural, 5=sounds like a human voice response (short, conversational)
- hallucination: true if the agent stated specific facts (visa fees, document lists, processing times) that are NOT in the provided knowledge context

User question: {user_text}
Agent response: {agent_text}
Knowledge context used: {rag_context}

Return ONLY the JSON object, no other text."""


async def score_turn(
    user_text: str,
    agent_text: str,
    rag_context: str = "",
) -> dict:
    """
    Score a single conversation turn using LLM-as-judge.

    Returns dict with: relevance, naturalness, hallucination, reasoning
    Falls back to None values on error (never blocks the main pipeline).
    """
    if not user_text or not agent_text:
        return {"relevance": None, "naturalness": None, "hallucination": None, "reasoning": "empty turn"}

    prompt = JUDGE_PROMPT.format(
        user_text=user_text[:300],
        agent_text=agent_text[:400],
        rag_context=rag_context[:500] if rag_context else "No specific context retrieved."
    )

    try:
        response = await _client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        scores = json.loads(raw)

        return {
            "relevance": float(scores.get("relevance", 3)),
            "naturalness": float(scores.get("naturalness", 3)),
            "hallucination": bool(scores.get("hallucination", False)),
            "reasoning": str(scores.get("reasoning", "")),
        }

    except Exception as e:
        logger.warning(f"Judge scoring failed (non-fatal): {e}")
        return {"relevance": None, "naturalness": None, "hallucination": None, "reasoning": f"error: {e}"}


async def score_turn_background(turn_metrics, user_text: str, agent_text: str, rag_context: str = ""):
    """
    Score a turn in the background and update the TurnMetrics object.
    Call this with asyncio.create_task() so it doesn't block the main pipeline.
    """
    scores = await score_turn(user_text, agent_text, rag_context)
    turn_metrics.relevance_score = scores.get("relevance")
    turn_metrics.naturalness_score = scores.get("naturalness")
    turn_metrics.hallucination = scores.get("hallucination")
    logger.debug(
        f"Judge scored turn: relevance={scores['relevance']}, "
        f"naturalness={scores['naturalness']}, "
        f"hallucination={scores['hallucination']}"
    )
