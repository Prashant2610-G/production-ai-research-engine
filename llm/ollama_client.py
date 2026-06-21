"""
Local LLM client using Ollama — runs Llama 3 entirely offline.
Mirrors the same interface as openai_client.py so they're swappable.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

SYSTEM_PROMPT = """You are an expert research assistant. Answer ONLY from the 
provided context. Cite sources like [source: filename, page X]. If the answer 
isn't in the context, say so clearly."""


def format_context(documents: list) -> str:
    """Same formatting as openai_client — keeps both clients consistent."""
    parts = []
    for i, doc in enumerate(documents):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "N/A")
        parts.append(f"[Chunk {i+1}]\nSource: {source} | Page: {page}\nContent: {doc.page_content}")
    return "\n---\n".join(parts)


def get_answer_local(query: str, documents: list) -> dict:
    """
    Sends question + context to local Llama 3 via Ollama.
    No API cost — runs entirely on your machine.
    """
    context = format_context(documents)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    start = time.time()

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })

    elapsed_ms = round((time.time() - start) * 1000)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")

    data = response.json()

    return {
        "answer": data["response"].strip(),
        "latency_ms": elapsed_ms,
        "cost_usd": 0.0,  # local = always free
        "model": MODEL
    }


if __name__ == "__main__":
    from langchain_core.documents import Document

    test_docs = [
        Document(
            page_content="Infrastructure costs rose by 23% in Q3 2024, driven by cloud compute expenses.",
            metadata={"source": "report_2024.pdf", "page": 14}
        )
    ]

    print("Make sure 'ollama serve' is running in another terminal!")
    result = get_answer_local("What happened to infrastructure costs?", test_docs)

    print(f"\nAnswer: {result['answer']}")
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Cost: ${result['cost_usd']} (local model, always free)")