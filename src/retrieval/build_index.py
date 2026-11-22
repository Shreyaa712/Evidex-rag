import json
from pathlib import Path

from src.config import PROCESSED_DIR
from src.ingestion.validate import validate_chunks
from src.retrieval.vector_store import get_collection, index_chunks

# Reference lists and thank-you notes only add noise to retrieval
SKIP_SECTIONS = {"References", "Acknowledgements"}


def load_all_chunks(processed_dir=PROCESSED_DIR):
    """
    Read the per-paper JSON files written by the ingestion pipeline.
    Files starting with "_" (log, flat chunk list) are skipped.
    """

    chunks = []

    for path in sorted(Path(processed_dir).glob("*.json")):

        if path.name.startswith("_"):
            continue

        with open(path, encoding="utf-8") as file:
            record = json.load(file)

        chunks.extend(
            chunk for chunk in record["chunks"]
            if chunk.get("section_heading") not in SKIP_SECTIONS
        )

    return chunks


if __name__ == "__main__":
    chunks = load_all_chunks()
    validate_chunks(chunks)

    total = index_chunks(chunks, get_collection(reset=True))
    print(f"Indexed. Collection now holds {total} chunks.")
