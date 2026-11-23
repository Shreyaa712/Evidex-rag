
import re


def create_chunks(
    pages: list[dict],
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[dict]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    chunks = []

    for page in pages:
        words = re.findall(r"\S+", page["text"])
        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": (
                        f"{page['paper_id']}_"
                        f"p{page['page_number']}_c{chunk_index + 1}"
                    ),
                    "paper_id": page["paper_id"],
                    "paper_name": page["paper_name"],
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "section_heading": None,
                })
                chunk_index += 1

            if end >= len(words):
                break

            start = end - overlap

    return chunks

