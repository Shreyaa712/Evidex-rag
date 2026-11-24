
import json

from src.ingestion.chunker import create_chunks
from src.retrieval.build_index import load_all_chunks


def test_build_index_skips_references_and_acknowledgements(tmp_path):
    pages = [
        {
            "paper_id": "paper1",
            "paper_name": "paper.pdf",
            "page_number": 1,
            "text": (
                "Introduction\n"
                "This is normal research content.\n"
                "References\n"
                "Reference one details.\n"
                "Acknowledgements\n"
                "Thank you to the research team."
            ),
        }
    ]

    chunks = create_chunks(
        pages,
        chunk_size=100,
        overlap=0,
    )

    record = {
        "paper_id": "paper1",
        "paper_name": "paper.pdf",
        "chunks": chunks,
    }

    json_path = tmp_path / "paper1.json"
    json_path.write_text(json.dumps(record), encoding="utf-8")

    loaded_chunks = load_all_chunks(tmp_path)

    headings = {
        chunk["section_heading"]
        for chunk in loaded_chunks
    }

    assert None in headings
    assert "References" not in headings
    assert "Acknowledgements" not in headings

    assert all(
        "Reference one details." not in chunk["chunk_text"]
        and "Thank you to the research team." not in chunk["chunk_text"]
        for chunk in loaded_chunks
    )
