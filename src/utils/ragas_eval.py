"""
RAGAS Benchmarking
-------------------
Evaluates the RAG pipeline using the RAGAS framework.
Metrics:
  • faithfulness       — is the answer grounded in context?
  • answer_relevancy   — how relevant is the answer to the question?
  • context_precision  — are retrieved chunks precise?
  • context_recall     — are all relevant chunks retrieved?

Expected test file format (JSON):
[
  {
    "question": "What is the total invoice amount?",
    "ground_truth": "The total invoice amount is $4,320.00",
    "contexts": ["...relevant chunk text..."]
  }
]
"""

import json
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.retrieval.qa_chain import answer_question

logger = logging.getLogger(__name__)


def build_ragas_dataset(test_file: str) -> Dataset:
    """Load test questions, run the pipeline, and build a RAGAS-compatible dataset."""
    test_path = Path(test_file)
    assert test_path.exists(), f"Test file not found: {test_file}"

    with open(test_path) as f:
        test_cases = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    for case in test_cases:
        question = case["question"]
        result = answer_question(question, top_k=6)

        questions.append(question)
        answers.append(result.answer)
        contexts.append([c["content"] for c in result.sources])  # list of strings
        ground_truths.append(case.get("ground_truth", ""))

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def run_ragas_benchmark(test_file: str) -> dict:
    """Run full RAGAS evaluation and return metric scores."""
    logger.info(f"Building RAGAS dataset from {test_file}")
    dataset = build_ragas_dataset(test_file)

    logger.info("Running RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "faithfulness": round(result["faithfulness"], 4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
        "context_precision": round(result["context_precision"], 4),
        "context_recall": round(result["context_recall"], 4),
        "overall": round(
            (result["faithfulness"] + result["answer_relevancy"]
             + result["context_precision"] + result["context_recall"]) / 4, 4
        ),
    }

    logger.info(f"RAGAS Results: {scores}")
    return scores
