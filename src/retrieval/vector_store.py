import chromadb
from chromadb.errors import NotFoundError

from src.config import ROOT_DIR
from src.retrieval.embeddings import MODEL_NAME, embed_texts

CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "evidex_chunks"


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},  # Chroma's default is L2, so set cosine explicitly
    )


def to_metadata(chunk: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool - never None."""
    return {
        "paper_id": chunk["paper_id"],
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"],
        "section_heading": chunk.get("section_heading") or "",
        "embedding_model": MODEL_NAME,
    }


def index_chunks(chunks: list[dict], collection, batch_size: int = 128) -> int:
    """Embed and store chunks. upsert = safe to re-run (same chunk_id overwrites)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["chunk_text"] for c in batch]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=texts,
            embeddings=embed_texts(texts),
            metadatas=[to_metadata(c) for c in batch],
        )
    return collection.count()