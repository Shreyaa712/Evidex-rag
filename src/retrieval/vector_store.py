from src.config import CHROMA_DIR
from src.retrieval.embeddings import MODEL_NAME, embed_texts

COLLECTION_NAME = "evidex_chunks"


def _get_or_create(client):
    """Create the collection with cosine distance (Chroma's default is L2)."""
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )
    except TypeError:  # older Chroma versions use metadata instead
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def get_collection(reset=False):
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:  # the collection did not exist yet
            pass

    return _get_or_create(client)


def to_metadata(chunk):
    """Chroma metadata values must be str/int/float/bool - never None."""
    return {
        "paper_id": chunk["paper_id"],
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"],
        "section_heading": chunk.get("section_heading") or "",
        "embedding_model": MODEL_NAME,
    }


def index_chunks(chunks, collection, batch_size=128):
    """Embed and store chunks. upsert = safe to re-run (same chunk_id overwrites)."""
    for i in range(0, len(chunks), batch_size):

        batch = chunks[i : i + batch_size]
        texts = [chunk["chunk_text"] for chunk in batch]

        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=texts,
            embeddings=embed_texts(texts),
            metadatas=[to_metadata(chunk) for chunk in batch],
        )

    return collection.count()
