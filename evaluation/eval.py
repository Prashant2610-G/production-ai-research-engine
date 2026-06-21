"""
RAGAS-based evaluation — scores faithfulness, relevancy, and precision
of the RAG pipeline against a test question set.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from agent.rag_chain import RAGChain


# test questions with known-good answers (used to score quality)
TEST_SET = [
    {
        "question": "What is the Transformer model architecture?",
        "ground_truth": "The Transformer relies entirely on attention mechanisms, removing recurrence and convolutions, enabling parallelization."
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Multi-head attention runs several attention operations in parallel and combines their outputs, letting the model attend to different parts of the input simultaneously."
    },
]


def run_evaluation(chain: RAGChain) -> dict:
    """
    Runs the test set through the RAG pipeline, collects answers
    and retrieved contexts, then scores them with RAGAS metrics.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"Running {len(TEST_SET)} test questions through the pipeline...\n")

    for item in TEST_SET:
        question = item["question"]
        print(f"  Q: {question}")

        response = chain.query(question)

        questions.append(question)
        answers.append(response.answer)
        contexts.append([doc.page_content for doc in response.sources])
        ground_truths.append(item["ground_truth"])

    # RAGAS expects a HuggingFace Dataset format
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\nScoring with RAGAS metrics...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )

    scores = result.to_pandas()[["faithfulness", "answer_relevancy", "context_precision"]].mean().to_dict()
    scores = {k: round(v, 3) for k, v in scores.items()}

    return scores


def check_quality_gate(scores: dict, thresholds: dict = None) -> bool:
    """
    Compares scores against minimum thresholds.
    Used in CI — if scores drop below threshold, the pipeline fails
    and blocks deployment. This is "regression gating."
    """
    if thresholds is None:
        thresholds = {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "context_precision": 0.6
        }

    passed = True
    print("\n--- Quality Gate Check ---")
    for metric, threshold in thresholds.items():
        score = scores.get(metric, 0)
        status = "PASS" if score >= threshold else "FAIL"
        if score < threshold:
            passed = False
        print(f"  {metric}: {score} (min {threshold}) — {status}")

    return passed


if __name__ == "__main__":
    chain = RAGChain(data_dir="data/")
    ready = chain.setup()

    if not ready:
        print("Add PDF files to data/ folder first.")
        sys.exit(1)

    scores = run_evaluation(chain)

    print("\n--- RAGAS Scores ---")
    for metric, score in scores.items():
        print(f"  {metric}: {score}")

    # save scores for CI pipeline to read later
    with open("eval_results.json", "w") as f:
        json.dump(scores, f, indent=2)

    passed = check_quality_gate(scores)

    if not passed:
        print("\nEvaluation FAILED — scores below threshold.")
        sys.exit(1)
    else:
        print("\nEvaluation PASSED — quality gate satisfied.")