
from src.ingestion.chunker import create_chunks


def make_page(paper_id, paper_name, page_number, text):
    return {
        "paper_id": paper_id,
        "paper_name": paper_name,
        "page_number": page_number,
        "text": text,
    }


def test_chunk_can_contain_text_from_multiple_pages():
    pages = [
        make_page("paper1", "Paper 1", 1, "one two three"),
        make_page("paper1", "Paper 1", 2, "four five six seven eight"),
    ]

    chunks = create_chunks(
        pages, chunk_size=5, overlap=0, min_chunk_words=2
    )

    assert chunks[0]["page_numbers"] == [1, 2]
    assert chunks[0]["chunk_text"] == "one two three four five"


def test_page_numbers_are_tracked_correctly():
    pages = [
        make_page("paper1", "Paper 1", 1, "one two three"),
        make_page("paper1", "Paper 1", 2, "four five six seven eight"),
    ]

    chunks = create_chunks(
        pages, chunk_size=5, overlap=0, min_chunk_words=2
    )

    assert chunks[0]["page_numbers"] == [1, 2]
    assert chunks[1]["page_numbers"] == [2]


def test_chunks_from_different_papers_are_not_mixed():
    pages = [
        make_page("paper1", "Paper 1", 1, "apple banana"),
        make_page("paper2", "Paper 2", 1, "orange mango"),
    ]

    chunks = create_chunks(
        pages, chunk_size=10, overlap=0, min_chunk_words=2
    )

    assert len(chunks) == 2
    assert chunks[0]["paper_id"] == "paper1"
    assert chunks[1]["paper_id"] == "paper2"


def test_small_final_remainder_is_merged():
    pages = [
        make_page(
            "paper1",
            "Paper 1",
            1,
            "one two three four five six seven",
        )
    ]

    chunks = create_chunks(
        pages, chunk_size=5, overlap=0, min_chunk_words=3
    )

    assert len(chunks) == 1
    assert chunks[0]["chunk_text"] == (
        "one two three four five six seven"
    )


def test_references_heading_is_detected():
    pages = [
        make_page(
            "paper1",
            "Paper 1",
            1,
            "Introduction\nThis is the main content.\n"
            "References\nReference one\nReference two",
        )
    ]

    chunks = create_chunks(
        pages, chunk_size=20, overlap=0, min_chunk_words=2
    )

    assert chunks[0]["section_heading"] is None
    assert chunks[1]["section_heading"] == "References"


def test_chunk_does_not_mix_content_and_references():
    pages = [
        make_page(
            "paper1",
            "Paper 1",
            1,
            "main content words here\n"
            "References\nreference words here",
        )
    ]

    chunks = create_chunks(
        pages, chunk_size=20, overlap=0, min_chunk_words=2
    )

    assert len(chunks) == 2
    assert chunks[0]["section_heading"] is None
    assert chunks[1]["section_heading"] == "References"
