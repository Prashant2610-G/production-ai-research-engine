import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import pickle


# File to save BM25 index on disk
BM25_INDEX_PATH = "bm25_index.pkl"


def tokenize(text: str) -> list:
    """
    Splits text into individual words (tokens).
    BM25 works on word level, not sentence level.
    Example: "hello world" → ["hello", "world"]
    """
    return text.lower().split()


def build_bm25_index(chunks: list):
    """
    Builds a BM25 keyword index from document chunks.
    Saves it to disk so we don't rebuild every time.
    """
    print("Building BM25 index...")

    # Tokenize all chunks
    tokenized_chunks = [tokenize(chunk.page_content) for chunk in chunks]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)

    # Save index + original chunks to disk
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"BM25 index built with {len(chunks)} chunks")
    print(f"Saved to: {BM25_INDEX_PATH}")
    return bm25, chunks


def load_bm25_index():
    """
    Loads existing BM25 index from disk.
    """
    if not os.path.exists(BM25_INDEX_PATH):
        raise FileNotFoundError("BM25 index not found. Run build_bm25_index() first.")

    print("Loading BM25 index...")
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    print("BM25 index loaded successfully")
    return data["bm25"], data["chunks"]


def bm25_search(query: str, bm25, chunks: list, k: int = 5) -> list:
    """
    Searches the BM25 index for a query.
    Returns top-k most relevant chunks.
    """
    # Tokenize the query
    tokenized_query = tokenize(query)

    # Get relevance scores for all chunks
    scores = bm25.get_scores(tokenized_query)

    # Get top-k chunk indices sorted by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    # Return top chunks as Document objects with scores
    results = []
    for idx in top_indices:
        doc = chunks[idx]
        doc.metadata["bm25_score"] = round(scores[idx], 4)
        results.append(doc)

    return results


def bm25_index_exists():
    """
    Check if BM25 index already exists on disk.
    """
    return os.path.exists(BM25_INDEX_PATH)


# Test this file directly
if __name__ == "__main__":
    from ingestion.loader import load_and_split

    chunks = load_and_split("data/")

    if chunks:
        bm25, chunks = build_bm25_index(chunks)
        results = bm25_search("infrastructure cost", bm25, chunks, k=3)
        print(f"\nTop {len(results)} BM25 results:")
        for i, doc in enumerate(results):
            print(f"\n[{i+1}] Score: {doc.metadata['bm25_score']}")
            print(f"Content: {doc.page_content[:150]}...")
    else:
        print("Add PDF files to data/ folder first, then run again.")