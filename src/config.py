from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "evaluation"
PAPERS_META_PATH = DATA_DIR / "papers.json"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# Chunking (in WORDS). 150 words is roughly 200-225 tokens, which fits the
# 256-token limit of the embedding model. We test other sizes during evaluation.
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30
MIN_CHUNK_WORDS = 20