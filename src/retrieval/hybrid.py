"""
Hybrid search: run dense search and BM25 search, then merge the two rankings
with Reciprocal Rank Fusion (RRF).
"""
from src.retrieval.bm25 import bm25_search
from src.retrieval.vector_search import search as dense_search

RRF_K = 60  # standard constant; larger values flatten the difference between ranks


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """
    Merge several ranked lists into one.

    Args:
        rankings: dict like {"dense": [hit, ...], "bm25": [hit, ...]}, each list best first.
        k: RRF constant.

    A chunk's fused score is the sum over lists of 1 / (k + rank in that list).
    Only ranks are used, so the different score scales never need comparing.
    """

    scores = {}
    first_hit = {}
    ranks = {}

    for name, ranking in rankings.items():
        for rank, hit in enumerate(ranking, start=1):

            chunk_id = hit["chunk_id"]

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            first_hit.setdefault(chunk_id, hit)
            ranks.setdefault(chunk_id, {})[name] = rank

    fused = []

    for chunk_id in sorted(scores, key=lambda cid: scores[cid], reverse=True):
        hit = {key: value for key, value in first_hit[chunk_id].items() if key != "score"}
        hit["rrf_score"] = round(scores[chunk_id], 6)
        hit["ranks"] = ranks[chunk_id]
        fused.append(hit)

    return fused


def hybrid_search(
    query,
    k=5,
    paper_id=None,
    candidates=20,
    rrf_k=RRF_K,
    dense_fn=dense_search,
    bm25_fn=bm25_search,
):
    """Top-k chunks after fusing the top `candidates` of dense search and BM25."""
    rankings = {
        "dense": dense_fn(query, k=candidates, paper_id=paper_id),
        "bm25": bm25_fn(query, k=candidates, paper_id=paper_id),
    }

    return reciprocal_rank_fusion(rankings, k=rrf_k)[:k]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"

    for hit in hybrid_search(question):
        found_by = ", ".join(f"{name} #{rank}" for name, rank in hit["ranks"].items())
        print(
            f"{hit['rrf_score']:.4f}  {hit['paper_id']}  p.{hit['page_number']}"
            f"  [{hit['section_heading']}]  ({found_by})"
        )
        print(f"    {hit['text'][:160]!r}\n")
