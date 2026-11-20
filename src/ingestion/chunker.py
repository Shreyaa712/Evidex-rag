
def create_chunks(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap: int = 200
) -> list[dict]:

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    chunks = []

    for page in pages:
        text = page["text"]
        start = 0
        chunk_number = 1

        while start < len(text):

            end = min(start + chunk_size, len(text))

            # Avoid cutting words unnecessarily
            if end < len(text):
                boundary = text.rfind(" ", start, end)

                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": (
                        f"{page['paper_id']}_"
                        f"p{page['page_number']}_c{chunk_number}"
                    ),
                    "paper_id": page["paper_id"],
                    "paper_name": page["paper_name"],
                    "page_number": page["page_number"],
                    "chunk_number": chunk_number,
                    "text": chunk_text
                })

                chunk_number += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks

