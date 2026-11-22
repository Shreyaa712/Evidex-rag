from src.retrieval.embeddings import embed_query
from src.retrieval.vector_store import get_collection


def search(query, k=5, paper_id=None):
    """
    Return the k chunks most similar to the query.
    paper_id optionally restricts the search to one paper.
    """

    collection = get_collection()
    total = collection.count()

    if total == 0:
        raise RuntimeError("The index is empty - run: python -m src.retrieval.build_index")

    kwargs = {"where": {"paper_id": paper_id}} if paper_id else {}

    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(k, total),
        **kwargs,
    )

    hits = []

    for chunk_id, text, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({
            "chunk_id": chunk_id,
            "paper_id": meta["paper_id"],
            "page_number": meta["page_number"],
            "section_heading": meta.get("section_heading", ""),
            "score": round(1 - distance, 4),  # cosine distance -> cosine similarity
            "text": text,
        })

    return hits


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"

    for hit in search(question):
        print(
            f"{hit['score']:.3f}  {hit['paper_id']}  p.{hit['page_number']}"
            f"  [{hit['section_heading']}]"
        )
        print(f"    {hit['text'][:160]!r}\n")
