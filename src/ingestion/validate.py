"""Validate chunk dictionaries against the Evidex-RAG chunk contract."""

import argparse
import json
import sys
from pathlib import Path


REQUIRED = {"chunk_id", "paper_id", "page_number", "chunk_index", "chunk_text"}


def validate_chunks(chunks: list[dict]) -> None:
    if not chunks:
        raise ValueError("no chunks at all")

    seen_ids = set()

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise TypeError(f"chunk at index {index} must be a dictionary")

        missing = REQUIRED - chunk.keys()
        if missing:
            raise ValueError(
                f"{chunk.get('chunk_id')}: missing fields {sorted(missing)}"
            )

        chunk_id = chunk["chunk_id"]
        if chunk_id in seen_ids:
            raise ValueError(f"duplicate chunk_id {chunk_id}")
        seen_ids.add(chunk_id)

        page_number = chunk["page_number"]
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError(f"{chunk_id}: bad page_number")

        chunk_index = chunk["chunk_index"]
        if (
            not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or chunk_index < 0
        ):
            raise ValueError(f"{chunk_id}: bad chunk_index")

        chunk_text = chunk["chunk_text"]
        if not isinstance(chunk_text, str):
            raise TypeError(f"{chunk_id}: chunk_text must be a string")
        if not chunk_text.strip():
            raise ValueError(f"{chunk_id}: empty text")

    print(
        f"OK: {len(chunks)} chunks from "
        f"{len({chunk['paper_id'] for chunk in chunks})} papers"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Evidex-RAG chunks from a JSON file."
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        help="Path to a JSON list of chunks, or '-' to read standard input.",
    )
    args = parser.parse_args()

    if not args.json_file:
        parser.print_help()
        return

    try:
        if args.json_file == "-":
            chunks = json.load(sys.stdin)
        else:
            chunks = json.loads(Path(args.json_file).read_text(encoding="utf-8"))

        if not isinstance(chunks, list):
            raise TypeError("JSON root must be a list of chunk dictionaries")

        validate_chunks(chunks)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")


if __name__ == "__main__":
    main()