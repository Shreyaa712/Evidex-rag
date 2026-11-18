
import re


def clean_text(text):
    """
    Clean extracted research paper text.

    Preserves meaningful paragraph boundaries while
    removing unnecessary whitespace and formatting noise.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # Normalize Windows and old-style line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null characters
    text = text.replace("\x00", "")

    # Remove unnecessary spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces at the beginning and end of lines
    text = re.sub(r" *\n *", "\n", text)

    # Reduce excessive blank lines while preserving paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text


def clean_pages(pages):
    """
    Clean extracted page text without losing page numbers.
    """

    cleaned_pages = []

    for page in pages:

        cleaned_pages.append({
            "page_number": page["page_number"],
            "text": clean_text(page["text"])
        })

    return cleaned_pages
