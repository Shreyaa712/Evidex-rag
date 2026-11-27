
import re

from src.config import MIN_CHUNK_WORDS


def detect_section_heading(line: str) -> str | None:
    """Detect section headings that should be identified during indexing."""
    heading = line.strip().lower()

    # Remove numbering such as "7 References" or "7.1 References".
    heading = re.sub(r"^\d+(?:\.\d+)*\s*", "", heading)

    # Remove surrounding punctuation.
    heading = heading.strip(" \t:.-–—")

    if heading in {"references", "bibliography", "works cited"}:
        return "References"

    if heading in {"acknowledgements", "acknowledgments"}:
        return "Acknowledgements"

    return None


def create_chunks(
    pages: list[dict],
    chunk_size: int = 150,
    overlap: int = 30,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> list[dict]:

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    if not 0 < min_chunk_words <= chunk_size:
        raise ValueError("Invalid minimum chunk size.")

    # Group pages by paper so different PDFs never get mixed.
    pages_by_paper = {}

    for page in pages:
        pages_by_paper.setdefault(page["paper_id"], []).append(page)

    chunks = []

    for paper_id, paper_pages in pages_by_paper.items():
        paper_pages.sort(key=lambda page: page["page_number"])

        words = []
        word_pages = []
        word_sections = []

        current_section = None

        # Read line by line to detect section headings.
        for page in paper_pages:
            for line in page["text"].splitlines():

                detected_heading = detect_section_heading(line)

                if detected_heading:
                    current_section = detected_heading
                    continue

                line_words = re.findall(r"\S+", line)

                words.extend(line_words)
                word_pages.extend(
                    [page["page_number"]] * len(line_words)
                )
                word_sections.extend(
                    [current_section] * len(line_words)
                )

        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            section = word_sections[start]
            cut_at_section = False

            # Do not allow a chunk to cross into another section.
            for position in range(start + 1, end):
                if word_sections[position] != section:
                    end = position
                    cut_at_section = True
                    break

            # Merge a small final remainder only if it belongs
            # to the same section.
            remaining_is_same_section = all(
                item == section for item in word_sections[end:]
            )

            if (
                end == min(start + chunk_size, len(words))
                and 0 < len(words) - end < min_chunk_words
                and remaining_is_same_section
            ):
                end = len(words)

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                page_numbers = list(
                    dict.fromkeys(word_pages[start:end])
                )

                chunks.append({
                    "chunk_id": (
                        f"{paper_id}_p{page_numbers[0]}_c{chunk_index + 1}"
                    ),
                    "paper_id": paper_id,
                    "paper_name": paper_pages[0]["paper_name"],
                    "page_number": page_numbers[0],
                    "page_numbers": page_numbers,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "section_heading": section,
                })

                chunk_index += 1

            if end >= len(words):
                break
            # After a cut at a section boundary, start fresh in the new section.
            # Overlapping here would pull the old section's words back in and the
            # same cut would repeat forever (infinite loop). max() guarantees
            # that start always moves forward.
            start = end if cut_at_section else max(end - overlap, start + 1)

    return chunks

