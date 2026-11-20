"""
Quick sanity check: does the right paper show up in the top-k for each test query?
This is NOT the final evaluation (Precision@K, Recall@K, MRR come later).

Usage:
    python -m src.retrieval.mini_check            # compares dense, bm25 and hybrid
    python -m src.retrieval.mini_check hybrid     # one method only
"""
import json
import sys

from src.config import EVAL_DIR
from src.retrieval.bm25 import bm25_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.vector_search import search

QUERIES_PATH = EVAL_DIR / "mini_queries.json"

METHODS = {"dense": search, "bm25": bm25_search, "hybrid": hybrid_search}


def run(k=5, search_fn=search, queries_path=QUERIES_PATH):
    with open(queries_path, encoding="utf-8") as file:
        queries = json.load(file)

    hits_found = 0

    for query in queries:

        results = search_fn(query["question"], k=k)
        found = any(r["paper_id"] == query["expected_paper_id"] for r in results)
        hits_found += found

        print(f"[{'HIT ' if found else 'MISS'}] {query['question']}")

        if not found:
            print("       got:", [(r["paper_id"], r["page_number"]) for r in results])

    rate = hits_found / len(queries)
    print(f"\nhit rate @{k}: {hits_found}/{len(queries)} = {rate:.0%}")

    return rate


if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"

    if choice != "all" and choice not in METHODS:
        sys.exit(f"Unknown method '{choice}'. Use: all, {', '.join(METHODS)}")

    names = list(METHODS) if choice == "all" else [choice]
    summary = {}

    for name in names:
        print(f"\n=== {name} ===")
        summary[name] = run(search_fn=METHODS[name])

    print("\nSummary:", {name: f"{rate:.0%}" for name, rate in summary.items()})