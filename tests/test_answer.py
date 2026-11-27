from src.generation.answer import build_prompt, sources_for

FAKE_HITS = [
    {"paper_id": "3e67fc1a9977", "page_number": 2,
     "section_heading": "", "text": "DPR uses two BERT encoders."},
    {"paper_id": "23e3249e9a1e", "page_number": 3,
     "section_heading": "", "text": "RAG combines a retriever and a generator."},
]


def test_prompt_numbers_each_excerpt_with_its_source():
    prompt = build_prompt("What is DPR?", FAKE_HITS)

    assert "[1] (paper 3e67fc1a9977, page 2)" in prompt
    assert "[2] (paper 23e3249e9a1e, page 3)" in prompt
    assert prompt.endswith("Answer:")


def test_sources_use_the_same_numbering_as_the_prompt():
    sources = sources_for(FAKE_HITS)

    assert [s["number"] for s in sources] == [1, 2]
    assert sources[0]["paper_id"] == "3e67fc1a9977"