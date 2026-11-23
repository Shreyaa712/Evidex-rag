"""
Evaluate retrieval methods with Precision@K, Recall@K, and MRR.

Usage:
    python -m src.retrieval.evaluate
    python -m src.retrieval.evaluate bm25
"""
import json
import sys

from src.config import EVAL_DIR, PROCESSED_DIR
from src.retrieval.bm25 import bm25_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import reranked_hybrid_search
from src.retrieval.vector_search import search as dense_search

QUERIES_PATH = EVAL_DIR / "mini_queries.json"

METHODS = {
    "dense": dense_search,
    "bm25": bm25_search,
    "hybrid": hybrid_search,
    "rerank": reranked_hybrid_search,
}


def load_relevant_chunks():
    """Treat every chunk from an expected paper as relevant for that paper."""
    relevant_chunks = {}

    for path in sorted(PROCESSED_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as file:
            record = json.load(file)

        relevant_chunks[record["paper_id"]] = {
            chunk["chunk_id"] for chunk in record["chunks"]
        }

    return relevant_chunks


def precision_recall_and_reciprocal_rank(results, relevant_ids, k):
    """Compute chunk-level metrics for one query."""
    top_results = results[:k]
    retrieved_ids = [hit["chunk_id"] for hit in top_results]
    retrieved_relevant = [
        chunk_id for chunk_id in retrieved_ids if chunk_id in relevant_ids
    ]

    precision = len(retrieved_relevant) / k if k else 0.0
    recall = (
        len(retrieved_relevant) / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    return precision, recall, reciprocal_rank


def evaluate(k=5, search_fn=dense_search, queries_path=QUERIES_PATH):
    with open(queries_path, encoding="utf-8") as file:
        queries = json.load(file)

    relevant_chunks = load_relevant_chunks()
    precision_values = []
    recall_values = []
    reciprocal_ranks = []

    print(f"\nK = {k}")

    for query in queries:
        paper_id = query["expected_paper_id"]
        relevant_ids = relevant_chunks.get(paper_id, set())
        results = search_fn(query["question"], k=k)

        precision, recall, reciprocal_rank = (
            precision_recall_and_reciprocal_rank(results, relevant_ids, k)
        )
        precision_values.append(precision)
        recall_values.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        rr_label = f"{reciprocal_rank:.3f}" if reciprocal_rank else "-"
        print(
            f"[P@{k}={precision:.3f} R@{k}={recall:.3f} RR={rr_label}] "
            f"{query['question']}"
        )

    count = len(queries)
    metrics = {
        f"precision@{k}": sum(precision_values) / count if count else 0.0,
        f"recall@{k}": sum(recall_values) / count if count else 0.0,
        "mrr": sum(reciprocal_ranks) / count if count else 0.0,
    }

    print(
        f"Average: P@{k}={metrics[f'precision@{k}']: .3f}, "
        f"R@{k}={metrics[f'recall@{k}']: .3f}, "
        f"MRR={metrics['mrr']: .3f}"
    )

    return metrics


def main():
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if choice != "all" and choice not in METHODS:
        sys.exit(f"Unknown method '{choice}'. Use: all, {', '.join(METHODS)}")
    if k < 1:
        sys.exit("K must be at least 1")

    names = list(METHODS) if choice == "all" else [choice]

    print("Evidex retrieval evaluation")
    summary = {}
    for name in names:
        print(f"\n=== {name} ===")
        summary[name] = evaluate(k=k, search_fn=METHODS[name])

    print("\nSummary:")
    for name, metrics in summary.items():
        print(
            f"{name}: P@{k}={metrics[f'precision@{k}']:.3f}, "
            f"R@{k}={metrics[f'recall@{k}']:.3f}, "
            f"MRR={metrics['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
