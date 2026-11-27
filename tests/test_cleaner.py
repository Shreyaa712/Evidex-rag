from src.ingestion.cleaner import clean_text


def test_word_split_across_lines_is_rejoined():
    assert clean_text("re-\ntrieve passages") == "retrieve passages"


def test_capitalised_or_numeric_breaks_are_left_alone():
    assert clean_text("Self-\nAttention") == "Self-\nAttention"
    assert clean_text("pages 10-\n20") == "pages 10-\n20"