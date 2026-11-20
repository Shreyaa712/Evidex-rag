
import hashlib
import logging
from pathlib import Path

import pymupdf

from chunker import create_chunks
from cleaner import clean_pages
from validate import validate_chunks

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
    chunks = create_chunks(cleaned_pages)

    # 4. Validate chunks against the Evidex-RAG contract
    validate_chunks(chunks)

    print(
        f"Success: Processed {len(metadata_list)} PDFs -> "
        f"{len(cleaned_pages)} clean pages -> {len(chunks)} valid chunks."
    )


if __name__ == "__main__":
    main()