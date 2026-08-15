"""
rag.py — RAG retrieval module.

Wraps ChromaDB collection access and formats context
for injection into the LLM prompt.
"""
import logging
from typing import Optional
import chromadb
from backend.knowledge.loader import load_knowledge_base, retrieve

logger = logging.getLogger(__name__)

# Module-level collection — loaded once on first import
_collection: Optional[chromadb.Collection] = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        _collection = load_knowledge_base()
    return _collection


def build_rag_context(query: str, n_results: int = 3) -> str:
    """
    Retrieve relevant passages and format them as a context block
    to inject into the LLM system prompt.

    Args:
        query: The user's question
        n_results: How many passages to retrieve

    Returns:
        Formatted context string ready to inject into prompt
    """
    try:
        collection = get_collection()
        passages = retrieve(collection, query, n_results=n_results)

        if not passages:
            return ""

        # Only include passages with decent relevance (distance < 0.8)
        relevant = [p for p in passages if p["distance"] is None or p["distance"] < 0.8]

        if not relevant:
            return ""

        context_parts = []
        for p in relevant:
            context_parts.append(
                f"[{p['country']} — {p['visa_type']}]\n{p['text']}"
            )

        context = "\n\n".join(context_parts)
        logger.debug(f"RAG retrieved {len(relevant)} passages for: '{query[:60]}'")
        return context

    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return ""
