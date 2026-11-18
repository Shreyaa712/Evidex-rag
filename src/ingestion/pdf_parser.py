
from pathlib import Path
import pymupdf


def extract_pdf(pdf_path):
    """
    Extract text from a PDF page by page.

    Args:
        pdf_path: Path to the research paper PDF.

    Returns:
        A dictionary containing the PDF name,
        page count, and extracted page-level text.
    """

    pdf_path = Path(pdf_path)

    # Check whether the PDF exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Not a valid file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file must be a PDF.")

    # Open the PDF and extract page-level text
    with pymupdf.open(pdf_path) as document:

        pages = []

        for page_number, page in enumerate(document, start=1):

            page_text = page.get_text("text")

            pages.append({
                "page_number": page_number,
                "text": page_text
            })

        # Return structured extraction results
        extracted_data = {
            "file_name": pdf_path.name,
            "page_count": len(document),
            "pages": pages
        }

    return extracted_data
