from fastapi import FastAPI, HTTPException
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