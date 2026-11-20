"""
Check how many chunks are longer than the embedding model can read.
Text beyond the limit is silently ignored when embedding.

Usage: python -m src.retrieval.token_check
"""
from src.retrieval.build_index import load_all_chunks
from src.retrieval.embeddings import get_model


def main():
    model = get_model()
    limit = model.max_seq_length

    chunks = load_all_chunks()
    lengths = [len(model.tokenizer.encode(chunk["chunk_text"])) for chunk in chunks]

    over = sum(length > limit for length in lengths)

    print(f"model reads up to {limit} tokens")
    print(f"longest chunk: {max(lengths)} tokens")
    print(f"chunks over the limit: {over} of {len(lengths)}")


if __name__ == "__main__":
    main()