import pytest
from chromadb.errors import NotFoundError

from src.retrieval import vector_store


def test_reset_ignores_only_missing_collection(monkeypatch):
    collection = object()

    class Client:
        def delete_collection(self, name):
            raise NotFoundError("collection does not exist")

        def get_or_create_collection(self, **kwargs):
            return collection

    monkeypatch.setattr(
        vector_store.chromadb,
        "PersistentClient",
        lambda **kwargs: Client(),
    )

    assert vector_store.get_collection(reset=True) is collection


def test_reset_propagates_unexpected_delete_errors(monkeypatch):
    class Client:
        def delete_collection(self, name):
            raise RuntimeError("database is unavailable")

        def get_or_create_collection(self, **kwargs):
            return object()

    monkeypatch.setattr(
        vector_store.chromadb,
        "PersistentClient",
        lambda **kwargs: Client(),
    )

    with pytest.raises(RuntimeError, match="database is unavailable"):
        vector_store.get_collection(reset=True)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_index_chunks_rejects_non_positive_batch_size(batch_size):
    class Collection:
        def count(self):
            return 0

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        vector_store.index_chunks([{}], Collection(), batch_size=batch_size)
