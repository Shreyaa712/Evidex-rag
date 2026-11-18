
from pathlib import Path
import hashlib


def calculate_file_hash(pdf_path):
    """
    Calculate the SHA-256 hash of a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A 64-character SHA-256 hexadecimal hash.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Not a valid file: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file must be a PDF.")

    sha256 = hashlib.sha256()

    # Read the file in blocks to avoid loading large PDFs into memory
    with open(pdf_path, "rb") as file:

        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def find_duplicate(pdf_path, existing_hashes):
    """
    Check whether a PDF has already been processed.

    Args:
        pdf_path: Path to the PDF being checked.
        existing_hashes: Dictionary mapping filenames to SHA-256 hashes.

    Returns:
        The filename of the matching document, or None.
    """

    new_hash = calculate_file_hash(pdf_path)

    for file_name, stored_hash in existing_hashes.items():

        if new_hash == stored_hash:
            return file_name

    return None
