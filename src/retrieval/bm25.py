"""
BM25: keyword search that scores chunks by the exact words they contain.

Written from scratch (no extra library) so you can see how it works.
Dense search finds chunks with similar MEANING; BM25 finds chunks that contain
the query's actual WORDS (names, acronyms, numbers) - the two complement each other.
"""
import math
import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with",
}


def tokenize(text):
    """Lowercase, split into letters/digits, drop stop words and 1-character tokens."""
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


class BM25Index:
    """BM25 over a list of chunk dicts (each needs "chunk_text")."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1  # how quickly repeating a word stops adding score
        self.b = b    # how strongly long chunks are penalised

        self.doc_lens = []
        self.postings = {}  # term -> list of (chunk position, term count in that chunk)

        for position, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk["chunk_text"]))
            self.doc_lens.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings.setdefault(term, []).append((position, count))

        total = len(chunks)
        self.avg_len = sum(self.doc_lens) / total if total else 0.0

        # idf: rare words (in few chunks) are worth more than common ones
        self.idf = {
            term: math.log((total - len(docs) + 0.5) / (len(docs) + 0.5) + 1)
            for term, docs in self.postings.items()
        }

    def scores(self, query):
        """BM25 score of every chunk for the query (list aligned with self.chunks)."""
        scores = [0.0] * len(self.chunks)

        for term in set(tokenize(query)):

            if term not in self.postings:
                continue

            for position, count in self.postings[term]:
                length_norm = 1 - self.b + self.b * self.doc_lens[position] / self.avg_len
                scores[position] += self.idf[term] * (
                    count * (self.k1 + 1) / (count + self.k1 * length_norm)
                )

        return scores

    def search(self, query, k=5, paper_id=None):
        """Top-k chunks, best first, in the same format as vector search."""
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        hits = []

        for position in ranked:

            if scores[position] <= 0:
                break  # the rest contain none of the query words

            chunk = self.chunks[position]

            if paper_id and chunk["paper_id"] != paper_id:
                continue

            hits.append({
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "page_number": chunk["page_number"],
                "section_heading": chunk.get("section_heading") or "",
                "score": round(scores[position], 4),
                "text": chunk["chunk_text"],
            })

            if len(hits) == k:
                break

        return hits


_index = None


def get_index(rebuild=False):
    """Build the index once from the same chunks the vector index uses."""
    global _index

    if _index is None or rebuild:
        from src.retrieval.build_index import load_all_chunks

        _index = BM25Index(load_all_chunks())

    return _index


def bm25_search(query, k=5, paper_id=None):
    return get_index().search(query, k=k, paper_id=paper_id)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"

    for hit in bm25_search(question):
        print(
            f"{hit['score']:.3f}  {hit['paper_id']}  p.{hit['page_number']}"
            f"  [{hit['section_heading']}]"
        )
        print(f"    {hit['text'][:160]!r}\n")
