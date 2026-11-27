"""
Generation step: retrieve chunks, then ask an LLM to answer
using only those chunks, with citations.
"""
import os
from dotenv import load_dotenv
from groq import Groq
from src.retrieval.hybrid import hybrid_search

TOP_K = 5


SYSTEM_PROMPT = (
    "You answer questions about research papers using ONLY the numbered "
    "excerpts provided. If the excerpts do not contain the answer, say "
    "'I could not find this in the provided papers.' Do not use outside "
    "knowledge. After each claim, cite the excerpt number in square "
    "brackets, like [1] or [2][3]."
)


def build_context(hits):
    """Format retrieved chunks as numbered excerpts with their sources."""
    lines = []
    for number, hit in enumerate(hits, start=1):
        lines.append(
            f"[{number}] (paper {hit['paper_id']}, page {hit['page_number']})\n"
            f"{hit['text']}"
        )
    return "\n\n".join(lines)


def build_prompt(question, hits):
    """The user-side message: excerpts first, then the question."""
    context = build_context(hits)
    return f"Excerpts:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"


def sources_for(hits):
    """Source list in the same numbering the answer uses."""
    return [
        {
            "number": number,
            "paper_id": hit["paper_id"],
            "page_number": hit["page_number"],
            "section": hit["section_heading"],
            "text": hit["text"],
        }
        for number, hit in enumerate(hits, start=1)
    ]


def retrieve(question, k=TOP_K):
    """Get the top chunks using the hybrid search you already built."""
    return hybrid_search(question, k=k)


load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"  # or "openai/gpt-oss-20b"
TEMPERATURE = 0.0  # factual answers: no randomness

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        _client = Groq(api_key=api_key)
    return _client


def call_llm(system_prompt, user_prompt):
    """Send the prompt to Groq and return the answer text."""
    response = get_client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def answer(question):
    """Return {'answer': str, 'sources': [...]} for a question."""
    hits = retrieve(question)
    if not hits:
        return {"answer": "I could not find this in the provided papers.", "sources": []}

    text = call_llm(SYSTEM_PROMPT, build_prompt(question, hits))
    return {"answer": text, "sources": sources_for(hits)}