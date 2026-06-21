import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from agent.rag_chain import RAGChain
from llm.ollama_client import get_answer_local
from monitoring.tracer import save_metric, get_summary, get_recent_queries

st.set_page_config(page_title="Research Engine", page_icon="🔬", layout="wide")

# load the pipeline once and keep it in session state across reruns
if "chain" not in st.session_state:
    with st.spinner("Loading indexes..."):
        chain = RAGChain(data_dir="data/")
        ready = chain.setup()
        st.session_state.chain = chain
        st.session_state.ready = ready

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔬 Production AI Research Engine")

if not st.session_state.ready:
    st.warning("No documents found. Add PDF files to the `data/` folder and restart.")
    st.stop()

# sidebar — mode toggle

with st.sidebar:
    st.header("Settings")
    mode = st.radio("LLM mode", ["Cloud (GPT-4o)", "Local (Llama 3)"])
    st.divider()
    st.caption(f"{len(st.session_state.chain.chunks)} chunks indexed")

    st.divider()
    st.header("Live metrics")

    summary = get_summary()

    if not summary:
        st.caption("No queries yet — ask something to see metrics.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Total queries", summary["total_queries"])
        col2.metric("Avg cost/query", f"${summary['avg_cost_per_query']}")

        col3, col4 = st.columns(2)
        col3.metric("p50 latency", f"{summary['p50_latency_ms']}ms")
        col4.metric("p95 latency", f"{summary['p95_latency_ms']}ms")

        st.metric("Total cost so far", f"${summary['total_cost_usd']}")

        st.divider()
        st.subheader("Recent queries")
        for q in get_recent_queries(5):
            st.caption(f"**{q['model']}** · {q['latency_ms']}ms · ${q['cost_usd']}")
            st.caption(q["query"])
            st.divider()

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(f"📄 {src}")

# chat input
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if mode == "Cloud (GPT-4o)":
                response = st.session_state.chain.query(question)
                answer = response.answer
                sources = [f"{os.path.basename(d.metadata.get('source',''))} (p.{d.metadata.get('page','?')})" for d in response.sources]

                save_metric(
                    query=question,
                    latency_ms=response.latency_ms,
                    retrieval_ms=response.retrieval_ms,
                    llm_ms=response.llm_ms,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    num_chunks=response.num_chunks_retrieved,
                    model="gpt-4o"
                )
            else:
                from retrieval.hybrid_retriever import hybrid_search
                from retrieval.reranker import rerank

                chain = st.session_state.chain
                hybrid_results = hybrid_search(question, chain.vector_store, chain.bm25, chain.chunks, k=10)
                final_chunks = rerank(question, hybrid_results, top_k=3)
                result = get_answer_local(question, final_chunks)
                answer = result["answer"]
                sources = [f"{os.path.basename(d.metadata.get('source',''))} (p.{d.metadata.get('page','?')})" for d in final_chunks]

                save_metric(
                    query=question,
                    latency_ms=result["latency_ms"],
                    retrieval_ms=0,
                    llm_ms=result["latency_ms"],
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    num_chunks=len(final_chunks),
                    model="llama3.2-local"
                )

            st.write(answer)
            with st.expander("Sources"):
                for src in sources:
                    st.caption(f"📄 {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})