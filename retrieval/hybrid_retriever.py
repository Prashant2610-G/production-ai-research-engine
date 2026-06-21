import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bm25_retriever import bm25_search
from langchain_core.documents import Document


def reciprocal_rank_fusion(bm25_results: list, vector_results: list, k: int = 60) -> list:
    """
    Combines BM25 and vector search results using Reciprocal Rank Fusion.

    How RRF score is calculated:
    - Each document gets a score = 1 / (rank + k) from each list
    - k=60 is a standard constant that smooths the scores
    - Final score = sum of scores from both lists
    - Higher score = more relevant

    Example:
    - doc7 is rank 1 in BM25 → score = 1/(1+60) = 0.016
    - doc7 is rank 2 in vector → score = 1/(2+60) = 0.016
    - doc7 total RRF score = 0.032  ← appears in both, gets boosted!
    """
    scores = {}  # doc_id → total RRF score
    docs = {}    # doc_id → Document object

    # Score BM25 results
    for rank, doc in enumerate(bm25_results):
        # Use content as unique ID
        doc_id = doc.page_content[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + 1 + k)
        docs[doc_id] = doc

    # Score vector results
    for rank, doc in enumerate(vector_results):
        doc_id = doc.page_content[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + 1 + k)
        docs[doc_id] = doc

    # Sort by combined RRF score (highest first)
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    # Add RRF score to metadata so we can see it later
    final_results = []
    for doc_id in sorted_ids:
        doc = docs[doc_id]
        doc.metadata["rrf_score"] = round(scores[doc_id], 6)
        final_results.append(doc)

    return final_results


def hybrid_search(query: str, vector_store, bm25, chunks: list, k: int = 5) -> list:
    """
    Main function — runs both searches and merges results.

    Steps:
    1. Run BM25 keyword search → get top 10 results
    2. Run vector similarity search → get top 10 results
    3. Merge using RRF → get best combined top-k results

    We fetch more (10) from each and then trim to k after merging
    so we have enough candidates to rank properly.
    """
    print(f"Running hybrid search for: '{query}'")

    # Step 1 — BM25 keyword search
    bm25_results = bm25_search(query, bm25, chunks, k=10)
    print(f"  BM25 found: {len(bm25_results)} results")

    # Step 2 — Vector similarity search
    vector_results = vector_store.similarity_search(query, k=10)
    print(f"  Vector found: {len(vector_results)} results")

    # Step 3 — Merge with RRF
    merged = reciprocal_rank_fusion(bm25_results, vector_results)
    print(f"  After merging: {len(merged)} unique results")

    # Return top-k final results
    return merged[:k]


# Test this file directly
if __name__ == "__main__":
    from ingestion.loader import load_and_split
    from retrieval.vector_store import build_vector_store, load_vector_store, vector_store_exists
    from retrieval.bm25_retriever import build_bm25_index, load_bm25_index, bm25_index_exists

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

        # Test hybrid search
        results = hybrid_search("what are the main findings?", vector_store, bm25, chunks, k=5)

        print(f"\nTop {len(results)} hybrid results:")
        for i, doc in enumerate(results):
            print(f"\n[{i+1}] RRF Score: {doc.metadata.get('rrf_score', 'N/A')}")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Content: {doc.page_content[:150]}...")
    else:
        print("Add PDF files to data/ folder first, then run again.")