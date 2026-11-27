"""
Evaluate retrieval methods with Precision@K, Recall@K, and MRR.

Usage:
    python -m src.retrieval.evaluate
    python -m src.retrieval.evaluate bm25
"""
import json
import sys
from functools import lru_cache


from src.config import EVAL_DIR, PROCESSED_DIR
from src.retrieval.bm25 import bm25_search
from src.retrieval.build_index import SKIP_SECTIONS
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

@lru_cache(maxsize=None)
def _load_records():
    """Read every processed paper file once; return {paper_id: record}."""
    records = {}
    for path in sorted(PROCESSED_DIR.glob("*.json")):
        if path.name.startswith("_"):  # log / flat chunk list, not a paper
            continue
        with open(path, encoding="utf-8") as file:
            record = json.load(file)
        records[record["paper_id"]] = record
    return records

def load_relevant_chunks(paper_id, expected_pages):
    """Return relevant chunk IDs from the expected pages of a paper."""
    record = _load_records().get(paper_id)
    if record is None:
        return set()  # No relevant chunks found for the paper_id
           
    return {
            c["chunk_id"]
             for c in record["chunks"]
            if c["page_number"] in expected_pages
            and c.get("section_heading") not in SKIP_SECTIONS  # never indexed
    }

  


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

    precision_values = []
    recall_values = []
    reciprocal_ranks = []
    hit_values = []

    print(f"\nK = {k}")

    for query in queries:
        paper_id = query["expected_paper_id"]
        expected_pages = query.get("expected_pages", [])
        relevant_ids = load_relevant_chunks(paper_id, expected_pages)
        results = search_fn(query["question"], k=k)

        precision, recall, reciprocal_rank = (
            precision_recall_and_reciprocal_rank(results, relevant_ids, k)
        )
        precision_values.append(precision)
        recall_values.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        hit_values.append(1.0 if reciprocal_rank > 0 else 0.0)

        rr_label = f"{reciprocal_rank:.3f}" if reciprocal_rank else "-"
        print(
            f"[P@{k}={precision:.3f} R@{k}={recall:.3f} RR={rr_label}] "
            f"{query['question']}"
        )

    count = len(queries)
    metrics = {
        f"precision@{k}": sum(precision_values) / count if count else 0.0,
        f"recall@{k}": sum(recall_values) / count if count else 0.0,
        f"hit@{k}": sum(hit_values) / count if count else 0.0,
        "mrr": sum(reciprocal_ranks) / count if count else 0.0,
    }

    print(
        f"Average: P@{k}={metrics[f'precision@{k}']: .3f}, "
        f"R@{k}={metrics[f'recall@{k}']: .3f}, "
        f"Hit@{k}={metrics[f'hit@{k}']: .3f}, "
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
            f"Hit@{k}={metrics[f'hit@{k}']:.3f}, "
            f"MRR={metrics['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
