# Production AI Research Engine

A production-grade RAG (Retrieval Augmented Generation) system that answers questions over your documents using hybrid search, cross-encoder reranking, and dual LLM backends — with built-in observability and automated quality evaluation.

## What it does

Upload PDF documents, ask questions in plain English, and get accurate answers with page-level citations — pulled only from your documents, not general knowledge. Switch between cloud (GPT-4o) and fully offline local inference (Llama 3 via Ollama) depending on cost, privacy, or latency needs.

## Architecture


Query

├── BM25 keyword search ─┐

└── Vector search ────────┼─→ RRF merge → Cross-encoder rerank → LLM (GPT-4o / Llama 3) → Answer + citations

- **Hybrid retrieval**: combines BM25 (exact keyword matching) with vector similarity search, merged using Reciprocal Rank Fusion — catches both literal terms and semantic meaning
- **Cross-encoder reranking**: `ms-marco-MiniLM-L-6-v2` re-scores retrieved chunks for relevance before passing to the LLM
- **Dual LLM backends**: OpenAI GPT-4o (cloud) and Llama 3.2 via Ollama (local, offline, free)
- **Observability**: every query logs latency (p50/p95), token usage, and cost — visible in a live dashboard
- **Automated evaluation**: RAGAS scores (faithfulness, answer relevancy, context precision) gated in CI — failing scores block deployment

## Tech stack

| Layer | Tools |
|---|---|
| Retrieval | ChromaDB, BM25 (rank-bm25), sentence-transformers |
| LLM | OpenAI GPT-4o, Ollama (Llama 3.2) |
| Orchestration | LangChain |
| Evaluation | RAGAS |
| UI | Streamlit |
| CI/CD | GitHub Actions |

## Results

On a test set evaluated with RAGAS:

| Metric | Score |
|---|---|
| Faithfulness | 0.93 |
| Answer relevancy | 0.97 |
| Context precision | 0.79 |

## Running locally

```bash
git clone https://github.com/Prashant2610-G/production-ai-research-engine.git
cd production-ai-research-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI key to `.env`:


Drop PDF files into `data/`, then run:
```bash
streamlit run app.py
```

For local/offline mode, install [Ollama](https://ollama.ai) and pull a model:
```bash
ollama serve
ollama pull llama3.2
```

## Project structure

├── ingestion/         # PDF loading and chunking

├── retrieval/          # vector store, BM25, hybrid search, reranker

├── llm/                    # GPT-4o and Ollama clients

├── agent/               # main RAG pipeline orchestrator

├── monitoring/      # latency/cost/usage tracking

├── evaluation/       # RAGAS quality evaluation

├── app.py                  # Streamlit UI

└── .github/workflows/    # CI quality gate


## Design decisions

- **Hybrid over pure vector search**: BM25 catches exact technical terms (model names, numbers) that embeddings sometimes miss, improving recall on technical documents
- **Reranking before generation**: cross-encoders score query and document together, giving more accurate relevance than embedding similarity alone — keeps the LLM's context focused and reduces hallucination risk
- **Separate retrieval/LLM latency tracking**: pinpoints whether slowness comes from search or generation, critical for production debugging
- **Local mode**: demonstrates the privacy/cost/latency tradeoffs real teams face when choosing between API-based and self-hosted models
- **CI quality gate**: automated RAGAS evaluation runs on every push, blocking merges if answer quality drops below threshold