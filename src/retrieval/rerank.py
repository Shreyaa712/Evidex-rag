"""
Cross-encoder reranking for retrieval results.
"""
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)

    return _reranker


def rerank_hits(query, hits, k=5, reranker=None):
    """Score query/chunk pairs with a cross-encoder and return top-k hits."""
    if not hits:
        return []

    model = reranker or get_reranker()
    scores = model.predict(
        [(query, hit["text"]) for hit in hits],
        convert_to_numpy=True,
    )

    reranked = []
    for hit, score in zip(hits, scores):
        ranked_hit = dict(hit)
        ranked_hit["rerank_score"] = round(float(score), 6)
        reranked.append(ranked_hit)

    reranked.sort(
        key=lambda hit: hit["rerank_score"],
        reverse=True,
    )
    return reranked[:k]


def reranked_hybrid_search(
    query,
    k=5,
    paper_id=None,
    candidates=20,
    reranker=None,
):
    """Run hybrid retrieval, then rerank the fused candidates."""
    from src.retrieval.hybrid import hybrid_search

    candidates_hits = hybrid_search(
        query,
        k=candidates,
        paper_id=paper_id,
        candidates=candidates,
    )
    return rerank_hits(
        query,
        candidates_hits,
        k=k,
        reranker=reranker,
    )


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"

    for hit in reranked_hybrid_search(question):
        print(
            f"{hit['rerank_score']:.4f}  {hit['paper_id']}  "
            f"p.{hit['page_number']}  [{hit['section_heading']}]"
        )
        print(f"    {hit['text'][:160]!r}\n")
