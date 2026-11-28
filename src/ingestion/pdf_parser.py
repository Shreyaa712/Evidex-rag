
import json
import hashlib
import logging
from pathlib import Path

import pymupdf

from src.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    PAPERS_META_PATH,
    PROCESSED_DIR,
)
from src.ingestion.chunker import create_chunks
from src.ingestion.cleaner import clean_pages
from src.ingestion.validate import validate_chunks

logger = logging.getLogger(__name__)


def calculate_file_hash(pdf_path: Path) -> str:
    """Calculate the 64-character SHA-256 hash of a PDF file in 1MB blocks."""
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def extract_pdfs(folder_path: str) -> tuple[list[dict], list[dict]]:
    """
    Extract text pages and document metadata from all valid, non-duplicate PDFs
    in a given folder.
    """
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    pdf_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {folder_path}")

    pages: list[dict] = []
    documents_metadata: list[dict] = []
    seen_hashes: set[str] = set()

    for pdf_path in pdf_files:
        try:
            file_hash = calculate_file_hash(pdf_path)

            if file_hash in seen_hashes:
                logger.warning("Duplicate PDF skipped: %s", pdf_path.name)
                continue

            seen_hashes.add(file_hash)
            paper_id = file_hash[:12]

            with pymupdf.open(pdf_path) as document:
                if document.is_encrypted:
                    logger.warning("Encrypted PDF skipped: %s", pdf_path.name)
                    continue

                # Extract metadata directly while the document is already open
                pdf_meta = document.metadata or {}
                documents_metadata.append({
                    "paper_id": paper_id,
                    "paper_name": pdf_path.name,
                    "title": pdf_meta.get("title") or None,
                    "author": pdf_meta.get("author") or None,
                    "subject": pdf_meta.get("subject") or None,
                    "keywords": pdf_meta.get("keywords") or None,
                    "creator": pdf_meta.get("creator") or None,
                    "producer": pdf_meta.get("producer") or None,
                    "creation_date": pdf_meta.get("creationDate") or None,
                    "modification_date": pdf_meta.get("modDate") or None,
                    "page_count": len(document),
                })

                # Extract text page by page
                for page_number, page in enumerate(document, start=1):
                    text = page.get_text("text")

                    if not text.strip():
                        logger.warning(
                            "No extractable text: %s, page %d",
                            pdf_path.name,
                            page_number,
                        )
                        continue

                    pages.append({
                        "paper_id": paper_id,
                        "paper_name": pdf_path.name,
                        "page_number": page_number,
                        "text": text,
                    })

            logger.info("Processed PDF: %s", pdf_path.name)

        except Exception:
            logger.exception("Failed to process: %s", pdf_path.name)

    return pages, documents_metadata


def process_single_pdf(pdf_path: Path) -> dict:
    """Process one PDF and save its chunks and metadata."""

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("A valid PDF file is required.")

    file_hash = calculate_file_hash(pdf_path)
    paper_id = file_hash[:12]

    # Check whether this paper has already been processed.
    if PAPERS_META_PATH.exists():
        existing_metadata = json.loads(
            PAPERS_META_PATH.read_text(encoding="utf-8")
        )

        if any(item["paper_id"] == paper_id for item in existing_metadata):
            raise ValueError("This paper has already been uploaded.")

    # 1. Extract pages and metadata.
    with pymupdf.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        metadata = document.metadata or {}

        document_metadata = {
            "paper_id": paper_id,
            "paper_name": pdf_path.name,
            "title": metadata.get("title") or None,
            "author": metadata.get("author") or None,
            "subject": metadata.get("subject") or None,
            "keywords": metadata.get("keywords") or None,
            "creator": metadata.get("creator") or None,
            "producer": metadata.get("producer") or None,
            "creation_date": metadata.get("creationDate") or None,
            "modification_date": metadata.get("modDate") or None,
            "page_count": len(document),
        }

        raw_pages = []

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text")

            if text.strip():
                raw_pages.append({
                    "paper_id": paper_id,
                    "paper_name": pdf_path.name,
                    "page_number": page_number,
                    "text": text,
                })

    if not raw_pages:
        raise ValueError("No extractable text found in this PDF.")

    # 2. Clean extracted pages.
    cleaned_pages = clean_pages(raw_pages)

    # 3. Create chunks using the existing chunker.
    chunks = create_chunks(
        cleaned_pages,
        chunk_size=CHUNK_SIZE_WORDS,
        overlap=CHUNK_OVERLAP_WORDS,
    )

    # 4. Validate chunks.
    validate_chunks(chunks)

    # 5. Save this paper's processed record.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "paper_id": paper_id,
        "paper_name": pdf_path.name,
        "metadata": document_metadata,
        "chunks": chunks,
    }

    output_path = PROCESSED_DIR / f"{paper_id}.json"
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 6. Add metadata without overwriting existing papers.
    PAPERS_META_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_metadata = []
    if PAPERS_META_PATH.exists():
        existing_metadata = json.loads(
            PAPERS_META_PATH.read_text(encoding="utf-8")
        )

    existing_metadata.append(document_metadata)

    PAPERS_META_PATH.write_text(
        json.dumps(existing_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "paper_id": paper_id,
        "paper_name": pdf_path.name,
        "page_count": document_metadata["page_count"],
        "chunk_count": len(chunks),
    }

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    raw_folder = Path(__file__).resolve().parents[2] / "data" / "raw"

    # 1. Extract raw pages and document metadata
    raw_pages, metadata_list = extract_pdfs(str(raw_folder))

    if not raw_pages:
        logger.warning("No text-bearing pages were extracted from any PDF.")
        return

    # 2. Clean extracted page text
    cleaned_pages = clean_pages(raw_pages)

    # 3. Split cleaned pages into overlapping chunks
    chunks = create_chunks(
        cleaned_pages,
        chunk_size=CHUNK_SIZE_WORDS,
        overlap=CHUNK_OVERLAP_WORDS,
    )

    # 4. Validate chunks against the Evidex-RAG contract
    validate_chunks(chunks)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    chunks_by_paper: dict[str, list[dict]] = {}
    for chunk in chunks:
        chunks_by_paper.setdefault(chunk["paper_id"], []).append(chunk)

    metadata_by_paper = {
        item["paper_id"]: item
        for item in metadata_list
    }

    for paper_id, paper_chunks in chunks_by_paper.items():
        metadata = metadata_by_paper[paper_id]
        record = {
            "paper_id": paper_id,
            "paper_name": metadata["paper_name"],
            "metadata": metadata,
            "chunks": paper_chunks,
        }
        output_path = PROCESSED_DIR / f"{paper_id}.json"
        output_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    PAPERS_META_PATH.write_text(
        json.dumps(metadata_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Success: Processed {len(metadata_list)} PDFs -> "
        f"{len(cleaned_pages)} clean pages -> {len(chunks)} valid chunks.\n"
        f"Wrote records to {PROCESSED_DIR}"
    )


if __name__ == "__main__":
    main()