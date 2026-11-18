
from pathlib import Path
import pymupdf


def extract_metadata(pdf_path):
    """
    Extract embedded PDF metadata and basic document information.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pymupdf.open(pdf_path) as document:

        pdf_metadata = document.metadata or {}

        metadata = {
            "file_name": pdf_path.name,
            "title": pdf_metadata.get("title") or None,
            "author": pdf_metadata.get("author") or None,
            "subject": pdf_metadata.get("subject") or None,
            "keywords": pdf_metadata.get("keywords") or None,
            "creator": pdf_metadata.get("creator") or None,
            "producer": pdf_metadata.get("producer") or None,
            "creation_date": pdf_metadata.get("creationDate") or None,
            "modification_date": pdf_metadata.get("modDate") or None,
            "page_count": len(document)
        }

    return metadata
