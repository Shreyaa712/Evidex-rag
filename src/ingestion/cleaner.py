
import re


def clean_text(text: str) -> str:

    # Remove unwanted control characters
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        "",
        text
    )

    # Standardize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Re-join words that the PDF broke across two lines: "re-\ntrieve" -> "retrieve".
    # Only lowercase-to-lowercase breaks are joined, so "Self-\nAttention" and
    # numbers like "10-\n20" are left alone.
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    # Normalize spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_pages(pages: list[dict]) -> list[dict]:

    cleaned_pages = []

    for page in pages:
        cleaned = page.copy()

        cleaned["text"] = clean_text(page["text"])

        if cleaned["text"]:
            cleaned_pages.append(cleaned)

    return cleaned_pages
