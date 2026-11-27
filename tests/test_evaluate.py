import json

from src.retrieval.evaluate import (
     _load_records,
    load_relevant_chunks,
    precision_recall_and_reciprocal_rank,
)
from src.retrieval.rerank import rerank_hits


class FakeReranker:
    def predict(self, pairs, convert_to_numpy=True):
        return [float(text.count("relevant")) for _, text in pairs]


def test_rerank_hits_orders_by_cross_encoder_score():
    hits = [
        {"chunk_id": "a", "text": "low relevance"},
        {"chunk_id": "b", "text": "relevant relevant"},
    ]

    reranked = rerank_hits("question", hits, k=1, reranker=FakeReranker())

    assert reranked[0]["chunk_id"] == "b"
    assert reranked[0]["rerank_score"] == 2.0


def test_precision_recall_and_mrr_for_first_result():
    results = [
        {"chunk_id": "a"},
        {"chunk_id": "b"},
        {"chunk_id": "c"},
    ]

    precision, recall, reciprocal_rank = (
        precision_recall_and_reciprocal_rank(results, {"a", "b"}, k=2)
    )

    assert precision == 1.0
    assert recall == 1.0
    assert reciprocal_rank == 1.0


def test_recall_uses_only_all_relevant_chunks():
    results = [{"chunk_id": "a"}]

    precision, recall, reciprocal_rank = (
        precision_recall_and_reciprocal_rank(results, {"a", "b", "c"}, k=3)
    )

    assert precision == 1 / 3
    assert recall == 1 / 3
    assert reciprocal_rank == 1.0


def test_no_relevant_result_has_zero_mrr():
    results = [{"chunk_id": "x"}, {"chunk_id": "y"}]

    precision, recall, reciprocal_rank = (
        precision_recall_and_reciprocal_rank(results, {"a"}, k=2)
    )

    assert precision == 0.0
    assert recall == 0.0
    assert reciprocal_rank == 0.0

def test_load_relevant_chunks_returns_only_expected_pages(tmp_path, monkeypatch):
    paper1 = {
        "paper_id": "paper-1",
        "chunks": [
            {"chunk_id": "chunk-1", "page_number": 1},
            {"chunk_id": "chunk-2", "page_number": 2},
            {"chunk_id": "chunk-3", "page_number": 3},
            {"chunk_id": "chunk-4", "page_number": 4},
        ],
    }

    paper2 = {
        "paper_id": "paper-2",
        "chunks": [
            {"chunk_id": "chunk-5", "page_number": 2},
        ],
    }

    (tmp_path / "paper1.json").write_text(
        json.dumps(paper1), encoding="utf-8"
    )
    (tmp_path / "paper2.json").write_text(
        json.dumps(paper2), encoding="utf-8"
    )

    monkeypatch.setattr(
        "src.retrieval.evaluate.PROCESSED_DIR", tmp_path
    )
    _load_records.cache_clear()  # the loader caches; drop results from other tests
    relevant_ids = load_relevant_chunks("paper-1", [2, 3])

    assert relevant_ids == {"chunk-2", "chunk-3"}

def test_hit_is_one_when_any_relevant_chunk_is_in_top_k():
    results = [{"chunk_id": "x"}, {"chunk_id": "a"}, {"chunk_id": "y"}]

    _, _, reciprocal_rank = precision_recall_and_reciprocal_rank(
        results, {"a", "b", "c", "d"}, k=3
    )

    assert reciprocal_rank == 0.5          # first relevant chunk is at rank 2
    assert (1.0 if reciprocal_rank > 0 else 0.0) == 1.0   # so Hit@3 = 1


def test_hit_is_zero_when_no_relevant_chunk_is_in_top_k():
    results = [{"chunk_id": "x"}, {"chunk_id": "y"}]

    _, _, reciprocal_rank = precision_recall_and_reciprocal_rank(
        results, {"a"}, k=2
    )

    assert reciprocal_rank == 0.0
