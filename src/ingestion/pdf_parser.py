
from pathlib import Path
import pymupdf


def extract_pdf(pdf_path):

    pdf_path = Path(pdf_path)

    # Check whether the PDF exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Open the PDF
    document = pymupdf.open(pdf_path)

    extracted_data = {
        "file_name": pdf_path.name,
        "page_count": len(document),
        "metadata": document.metadata,
        "pages": []
    }

    # Extract text page by page
    for page_number, page in enumerate(document, start=1):

        page_text = page.get_text("text")

        extracted_data["pages"].append({
            "page_number": page_number,
            "text": page_text
        })

    document.close()

    return extracted_data


if __name__ == "__main__":

    # Locate the EVIDEX project folder
    project_root = Path(__file__).resolve().parents[2]

    pdf_path = (
        project_root
        / "data"
        / "raw"
        / "01_rag_lewis_2020.pdf"
    )

    # Extract PDF information
    result = extract_pdf(pdf_path)

    # Display results
    print("\n========== EVIDEX PDF EXTRACTION ==========")

    print("\n1. PDF NAME")
    print(result["file_name"])

    print("\n2. TOTAL PAGES")
    print(result["page_count"])

    print("\n3. PDF METADATA")
    print(result["metadata"])

    print("\n4. FIRST PAGE TEXT (FIRST 500 CHARACTERS)")
    print(result["pages"][0]["text"][:500])

    print("\n========== EXTRACTION COMPLETED ==========")
