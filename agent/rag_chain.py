"""
RAG pipeline orchestrator — connects retrieval, reranking, and LLM generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass

from retrieval.vector_store import load_vector_store, build_vector_store, vector_store_exists
from retrieval.bm25_retriever import load_bm25_index, build_bm25_index, bm25_index_exists
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank
from llm.openai_client import get_answer, get_answer_streaming
from ingestion.loader import load_and_split


@dataclass
class RAGResponse:
    """Holds everything returned after a query — answer, sources, and metrics."""
    answer: str
    sources: list
    latency_ms: float
    retrieval_ms: float
    llm_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    num_chunks_retrieved: int
    query: str


class RAGChain:
    """Main pipeline class — loads indexes once, handles all queries."""

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.vector_store = None
        self.bm25 = None
        self.chunks = None
        self.is_ready = False
        print("RAGChain initialized (call .setup() to load indexes)")

    def setup(self):
        """Loads or builds vector store and BM25 index from documents."""
        print("\n=== Setting up RAG Pipeline ===")
        start = time.time()

        chunks = load_and_split(self.data_dir)
        if not chunks:
            print("No documents found. Add PDF or text files to data/ folder.")
            return False

        self.chunks = chunks

        # load from disk if exists, otherwise build fresh
        if vector_store_exists():
            self.vector_store = load_vector_store()
        else:
            self.vector_store = build_vector_store(chunks)

        if bm25_index_exists():
            self.bm25, self.chunks = load_bm25_index()
        else:
            self.bm25, self.chunks = build_bm25_index(chunks)

        elapsed = round((time.time() - start) * 1000)
        self.is_ready = True
        print(f"Pipeline ready in {elapsed}ms | {len(self.chunks)} chunks indexed")
        return True

    def query(self, question: str, top_k: int = 3) -> RAGResponse:
        """
        Runs a question through the full pipeline.
        Fetches 10 candidates then reranks down to top_k for the LLM.
        """
        if not self.is_ready:
            raise RuntimeError("Call .setup() first.")

        print(f"\nQuery: {question}")
        total_start = time.time()

        # retrieval — hybrid search + rerank
        retrieval_start = time.time()
        hybrid_results = hybrid_search(question, self.vector_store, self.bm25, self.chunks, k=10)
        final_chunks = rerank(question, hybrid_results, top_k=top_k)
        retrieval_ms = round((time.time() - retrieval_start) * 1000)

        # generation
        llm_start = time.time()
        llm_result = get_answer(question, final_chunks)
        llm_ms = round((time.time() - llm_start) * 1000)

        total_ms = round((time.time() - total_start) * 1000)

        print(f"Done — total: {total_ms}ms | retrieval: {retrieval_ms}ms | llm: {llm_ms}ms")
        print(f"Tokens: {llm_result['input_tokens']} in / {llm_result['output_tokens']} out | cost: ${llm_result['total_cost']}")

        return RAGResponse(
            answer=llm_result["answer"],
            sources=final_chunks,
            latency_ms=total_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            input_tokens=llm_result["input_tokens"],
            output_tokens=llm_result["output_tokens"],
            cost_usd=llm_result["total_cost"],
            num_chunks_retrieved=len(final_chunks),
            query=question
        )

    def query_streaming(self, question: str, top_k: int = 3):
        """Streaming version — yields tokens one by one for real-time UI."""
        if not self.is_ready:
            raise RuntimeError("Call .setup() first.")

        hybrid_results = hybrid_search(question, self.vector_store, self.bm25, self.chunks, k=10)
        final_chunks = rerank(question, hybrid_results, top_k=top_k)

        for token in get_answer_streaming(question, final_chunks):
            yield token


if __name__ == "__main__":
    chain = RAGChain(data_dir="data/")
    ready = chain.setup()

    if ready:
        response = chain.query("What is the attention model architecture??")
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nSources:")
        for doc in response.sources:
            src = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "N/A")
            print(f"  - {src} | page {page} | rerank score: {doc.metadata.get('rerank_score')}")
    else:
        print("Add PDF files to data/ folder to test the full pipeline.")