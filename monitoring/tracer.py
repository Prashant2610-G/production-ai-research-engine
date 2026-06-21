"""
Tracks query metrics — latency, cost, tokens, and quality scores.
Persists data to JSON for dashboard display.
"""

import json
import os
import time
from datetime import datetime


METRICS_FILE = "metrics_log.json"


def load_metrics() -> list:
    """Loads existing metrics from disk."""
    if not os.path.exists(METRICS_FILE):
        return []
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


def save_metric(
    query: str,
    latency_ms: float,
    retrieval_ms: float,
    llm_ms: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    num_chunks: int,
    model: str = "gpt-4o"
):
    """
    Saves a single query's metrics to the log file.
    Called automatically after every query in the pipeline.
    """
    metrics = load_metrics()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query[:100],  # truncate long queries
        "model": model,
        "latency_ms": latency_ms,
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "num_chunks": num_chunks
    }

    metrics.append(entry)

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)


def get_summary() -> dict:
    """
    Computes summary stats across all logged queries.
    Used by the dashboard to show p50/p95 latency, total cost etc.
    """
    metrics = load_metrics()

    if not metrics:
        return {}

    latencies = sorted([m["latency_ms"] for m in metrics])
    costs = [m["cost_usd"] for m in metrics]
    tokens_in = [m["input_tokens"] for m in metrics]
    tokens_out = [m["output_tokens"] for m in metrics]

    # p50 = median, p95 = 95th percentile
    p50_idx = int(len(latencies) * 0.50)
    p95_idx = int(len(latencies) * 0.95)

    return {
        "total_queries": len(metrics),
        "p50_latency_ms": latencies[p50_idx],
        "p95_latency_ms": latencies[min(p95_idx, len(latencies) - 1)],
        "avg_latency_ms": round(sum(latencies) / len(latencies)),
        "total_cost_usd": round(sum(costs), 6),
        "avg_cost_per_query": round(sum(costs) / len(costs), 6),
        "total_tokens_in": sum(tokens_in),
        "total_tokens_out": sum(tokens_out),
        "avg_tokens_per_query": round(sum(tokens_in) / len(tokens_in)),
        "last_query_time": metrics[-1]["timestamp"]
    }


def get_recent_queries(n: int = 10) -> list:
    """Returns the last n queries for display in dashboard."""
    metrics = load_metrics()
    return metrics[-n:][::-1]  # most recent first


if __name__ == "__main__":
    # simulate saving a few test metrics
    print("Saving test metrics...")

    save_metric(
        query="What is the Transformer architecture?",
        latency_ms=2400,
        retrieval_ms=800,
        llm_ms=1600,
        input_tokens=530,
        output_tokens=120,
        cost_usd=0.0045,
        num_chunks=3
    )

    save_metric(
        query="How does multi-head attention work?",
        latency_ms=1900,
        retrieval_ms=650,
        llm_ms=1250,
        input_tokens=480,
        output_tokens=95,
        cost_usd=0.0038,
        num_chunks=3
    )

    summary = get_summary()
    print("\n--- Metrics Summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n--- Recent Queries ---")
    for q in get_recent_queries():
        print(f"  [{q['timestamp']}] {q['query']} | {q['latency_ms']}ms | ${q['cost_usd']}")