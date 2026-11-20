
import hashlib
import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


def extract_pdfs(folder_path: str) -> list[dict]:

    pdf_files = sorted(Path(folder_path).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found.")

    pages = []
    seen_hashes = set()

    for pdf_path in pdf_files:
        try:
            file_hash = hashlib.sha256(
                pdf_path.read_bytes()
            ).hexdigest()

            if file_hash in seen_hashes:
                logger.warning("Duplicate PDF skipped: %s", pdf_path.name)
                continue

            seen_hashes.add(file_hash)

            with pymupdf.open(pdf_path) as document:

                if document.is_encrypted:
                    logger.warning("Encrypted PDF skipped: %s", pdf_path.name)
                    continue

                for page_number, page in enumerate(document, start=1):

                    text = page.get_text("text")

                    if not text.strip():
                        logger.warning(
                            "No extractable text: %s, page %d",
                            pdf_path.name, page_number
                        )
                        continue

                    pages.append({
                        "paper_id": file_hash[:12],
                        "paper_name": pdf_path.name,
                        "page_number": page_number,
                        "text": text
                    })

            logger.info("Processed PDF: %s", pdf_path.name)

        except Exception:
            logger.exception("Failed to process: %s", pdf_path.name)

    return pages


def main() -> None:
    raw_folder = Path(__file__).resolve().parents[2] / "data" / "raw"
    pages = extract_pdfs(str(raw_folder))

    paper_count = len({page["paper_id"] for page in pages})
    print(f"Extracted {len(pages)} text-bearing pages from {paper_count} PDFs.")


if __name__ == "__main__":
    main()

