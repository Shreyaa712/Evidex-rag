
import re


# Common headings found in research papers
SECTION_PATTERN = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)?"
    r"(?P<section>"
    r"abstract|"
    r"introduction|"
    r"related work|"
    r"background|"
    r"methodology|"
    r"methods|"
    r"approach|"
    r"experimental setup|"
    r"experiments|"
    r"results|"
    r"discussion|"
    r"conclusion|"
    r"limitations|"
    r"future work|"
    r"references|"
    r"acknowledgements|"
    r"appendix"
    r")"
    r"\s*[:.]?\s*$",
    re.IGNORECASE
)


def identify_sections(pages):
    """
    Identify common research-paper section headings.
    """

    detected_sections = []

    for page in pages:

        page_number = page["page_number"]
        page_text = page["text"]

        for line in page_text.splitlines():

            match = SECTION_PATTERN.match(line)

            if match:
                detected_sections.append({
                    "page_number": page_number,
                    "section": match.group("section").title()
                })

    return detected_sections


def split_text(text, chunk_size=500, overlap=100):
    """
    Split text into chunks using word counts.

    Args:
        text: Input text.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words repeated between chunks.

    Returns:
        List of text chunks.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks



def create_chunks(pages, chunk_size=500, overlap=100, paper_id=None):
    """
    Create section-aware chunks while preserving page information.

    Detects section changes within a page instead of
    assigning one section to the entire page.
    """

    chunks = []

    current_section = "Unknown"
    paper_id = paper_id or "unknown"

    for page in pages:

        page_number = page["page_number"]
        page_text = page["text"]

        if not page_text.strip():
            continue

        current_segment = []

        def save_segment():
            segment_text = "\n".join(current_segment).strip()

            if not segment_text:
                return

            segment_chunks = split_text(
                segment_text,
                chunk_size=chunk_size,
                overlap=overlap
            )

            for chunk_text in segment_chunks:

                chunk_index = len(chunks)
                chunks.append({
                    "chunk_id": f"{paper_id}:{chunk_index}",
                    "paper_id": paper_id,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "section": current_section,
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })

        for line in page_text.splitlines():

            match = SECTION_PATTERN.match(line)

            if match:
                # Save content belonging to the previous section
                save_segment()

                # Update the current section
                current_section = match.group("section").title()

                current_segment = []

            else:
                current_segment.append(line)

        # Save the remaining content on this page
        save_segment()

    return chunks
