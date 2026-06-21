import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


# This model runs locally on your laptop — no API key needed!
# It downloads once (~300MB) the first time you run it
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Global model instance — loaded once, reused every time
_reranker_model = None


def get_reranker():
    """
    Loads the cross-encoder model.
    We use a global variable so the model loads only ONCE
    and stays in memory — avoids slow reloading every query.
    """
    global _reranker_model

    if _reranker_model is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        print("(First time only — downloads ~300MB, please wait...)")
        _reranker_model = CrossEncoder(RERANKER_MODEL)
        print("Reranker model loaded!")

    return _reranker_model


def rerank(query: str, documents: list, top_k: int = 3) -> list:
    """
    Reranks documents by how relevant they are to the query.

    How it works:
    1. Takes each document from hybrid search results
    2. Pairs it with the query: (query, document_text)
    3. Cross-encoder scores each pair from 0 to 1
    4. Returns top_k documents with highest scores

    Args:
        query: the user's question
        documents: list of Document objects from hybrid search
        top_k: how many final documents to return (default 3)

    Returns:
        List of top_k most relevant Document objects
    """
    if not documents:
        print("No documents to rerank")
        return []

    print(f"Reranking {len(documents)} documents...")

    model = get_reranker()

    # Create (query, document_text) pairs for the cross-encoder
    # The model reads BOTH together to understand relevance
    pairs = [(query, doc.page_content) for doc in documents]

    # Get relevance scores — higher = more relevant
    scores = model.predict(pairs)

    # Attach score to each document's metadata so we can see it
    for doc, score in zip(documents, scores):
        doc.metadata["rerank_score"] = round(float(score), 4)

    # Sort documents by rerank score, highest first
    ranked_docs = sorted(documents, key=lambda d: d.metadata["rerank_score"], reverse=True)

    # Keep only top_k most relevant
    final_docs = ranked_docs[:top_k]

    print(f"Reranking done — kept top {len(final_docs)} documents")

    # Show scores for transparency
    for i, doc in enumerate(final_docs):
        print(f"  [{i+1}] Score: {doc.metadata['rerank_score']} | "
              f"Source: {doc.metadata.get('source', 'unknown')}")

    return final_docs


def rerank_with_scores(query: str, documents: list, top_k: int = 3) -> list:
    """
    Same as rerank() but returns a list of tuples (document, score)
    Useful when you want to display scores in the UI.
    """
    reranked = rerank(query, documents, top_k)
    return [(doc, doc.metadata["rerank_score"]) for doc in reranked]


# Test this file directly
if __name__ == "__main__":
    from ingestion.loader import load_and_split
    from retrieval.vector_store import build_vector_store, load_vector_store, vector_store_exists
    from retrieval.bm25_retriever import build_bm25_index, load_bm25_index, bm25_index_exists
    from retrieval.hybrid_retriever import hybrid_search

    chunks = load_and_split("data/")

    if chunks:
        # Build or load indexes
        if vector_store_exists():
            vector_store = load_vector_store()
        else:
            vector_store = build_vector_store(chunks)

        if bm25_index_exists():
            bm25, chunks = load_bm25_index()
        else:
            bm25, chunks = build_bm25_index(chunks)

        # Run full pipeline: hybrid search → rerank
        query = "what are the main findings?"

        hybrid_results = hybrid_search(query, vector_store, bm25, chunks, k=10)
        final_results = rerank(query, hybrid_results, top_k=3)

        print(f"\n--- Final Top {len(final_results)} Results ---")
        for i, doc in enumerate(final_results):
            print(f"\n[{i+1}] Rerank Score: {doc.metadata['rerank_score']}")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
    else:
        print("Add PDF files to data/ folder first, then run again.")