"""
loader.py — Knowledge base ingestion into ChromaDB.

Loads visa_data.json, converts each visa type into text passages,
embeds them using sentence-transformers (free, local), and stores
in a ChromaDB collection for fast retrieval.
"""
import json
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "visa_data.json"
CHROMA_DIR = Path(__file__).parent.parent.parent / "evaluation" / "chroma_store"
COLLECTION_NAME = "visa_knowledge"


def _get_collection():
    """Get or create ChromaDB collection with local sentence-transformer embeddings."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def build_passages(data: list) -> tuple[list[str], list[dict], list[str]]:
    """
    Convert raw visa JSON into text passages suitable for embedding.
    Returns: (documents, metadatas, ids)
    """
    documents, metadatas, ids = [], [], []
    for entry in data:
        country = entry["country"]
        for vt in entry["visa_types"]:
            # Build a rich natural-language passage for embedding
            eligible = ", ".join(vt["eligible_passports"][:6])
            docs_list = "; ".join(vt["required_documents"])

            passage = (
                f"Country: {country}. "
                f"Visa Type: {vt['type']}. "
                f"Eligible passport holders: {eligible}. "
                f"Visa-on-arrival: {'Yes' if vt['visa_on_arrival'] else 'No'}. "
                f"E-Visa available: {'Yes' if vt['e_visa'] else 'No'}. "
                f"Duration of stay: {vt['duration']}. "
                f"Processing time: {vt['processing_time_days']} days. "
                f"Fee: USD {vt['fees_usd']} (INR {vt['fees_inr']}). "
                f"Required documents: {docs_list}. "
                f"Notes: {vt['notes']}"
            )

            doc_id = f"{country}_{vt['type']}".replace(" ", "_").replace("/", "_")
            documents.append(passage)
            metadatas.append({
                "country": country,
                "visa_type": vt["type"],
                "fees_usd": vt["fees_usd"],
                "processing_time": vt["processing_time_days"],
                "eligible_passports": ", ".join(vt["eligible_passports"]),
            })
            ids.append(doc_id)

    return documents, metadatas, ids


def load_knowledge_base() -> chromadb.Collection:
    """
    Load visa data into ChromaDB. Skips if already loaded.
    Returns the ChromaDB collection.
    """
    collection = _get_collection()

    # Check if already populated
    if collection.count() > 0:
        logger.info(f"Knowledge base already loaded: {collection.count()} passages")
        return collection

    logger.info("Loading knowledge base for the first time (this takes ~30s)...")
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    documents, metadatas, ids = build_passages(data)

    # Batch upsert
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info(f"Knowledge base loaded: {collection.count()} passages")
    return collection


def retrieve(collection: chromadb.Collection, query: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve the top-k most relevant visa passages for a given query.

    Args:
        collection: ChromaDB collection
        query: User's question
        n_results: Number of passages to retrieve

    Returns:
        List of dicts with 'text', 'country', 'visa_type', 'distance'
    """
    results = collection.query(query_texts=[query], n_results=n_results)

    passages = []
    for i, doc in enumerate(results["documents"][0]):
        passages.append({
            "text": doc,
            "country": results["metadatas"][0][i].get("country", ""),
            "visa_type": results["metadatas"][0][i].get("visa_type", ""),
            "distance": round(results["distances"][0][i], 3) if results.get("distances") else None,
        })
    return passages
