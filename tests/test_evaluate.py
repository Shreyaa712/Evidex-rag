from src.retrieval.evaluate import precision_recall_and_reciprocal_rank


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
