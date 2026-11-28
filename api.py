import json
import uuid
from pathlib import Path

from src.config import RAW_DIR, PROCESSED_DIR
from src.ingestion.pdf_parser import process_single_pdf
from src.retrieval.vector_store import get_collection, index_chunks
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.generation.answer import answer

app = FastAPI(
    title="Evidex API",
    description="Ask questions across research papers. Answers cite paper and page.",
    version="0.1.0",
)

# Lets a browser frontend (React, etc.) on another port call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Source(BaseModel):
    number: int
    paper_id: str
    page_number: int
    section: str
    text: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    try:
        result = answer(request.question.strip())
    except RuntimeError as error:  # for example, a missing API key
        raise HTTPException(status_code=500, detail=str(error))
    except Exception:
        raise HTTPException(status_code=502, detail="The language model call failed.")
    return result


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Check file extension.
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # 2. Create the raw directory if it doesn't exist.
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Keep the original filename, avoiding overwrites.
    original_name = Path(file.filename).name
    saved_path = RAW_DIR / original_name

    if saved_path.exists():
        saved_path = RAW_DIR / f"{uuid.uuid4().hex}_{original_name}"

    try:
        # 4. Save the uploaded PDF.
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        saved_path.write_bytes(content)

        # 5. Run the existing ingestion pipeline.
        result = process_single_pdf(saved_path)

    except ValueError as error:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error))

    except HTTPException:
        saved_path.unlink(missing_ok=True)
        raise

    except Exception as error:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Could not process this PDF."
        ) from error

    # 6. Load the newly generated chunks.
    processed_path = PROCESSED_DIR / f"{result['paper_id']}.json"

    with processed_path.open("r", encoding="utf-8") as f:
        paper_data = json.load(f)

    chunks = [
    chunk for chunk in paper_data["chunks"]
    if (chunk.get("section_heading") or "").strip().lower()
    not in {"references", "acknowledgements"}
]

    # 7. Add only this paper's chunks to ChromaDB.
    collection = get_collection(reset=False)
    total_chunks = index_chunks(chunks, collection)

    # 8. Return upload details.
    return {
        "message": "PDF uploaded and indexed successfully.",
        "paper_id": result["paper_id"],
        "paper_name": result["paper_name"],
        "page_count": result["page_count"],
        "chunk_count": result["chunk_count"],
        "total_indexed_chunks": total_chunks,
    }
