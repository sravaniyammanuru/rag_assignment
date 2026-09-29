from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.graph import rag_graph


app = FastAPI(
    title="Agentic AI RAG API",
    description="Answers questions using the Agentic AI eBook only.",
    version="1.0.0",
)


def _jsonish(meta: dict) -> dict:
    cleaned = {}
    for key, value in (meta or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


class ChatRequest(BaseModel):
    query: str


class RetrievedChunk(BaseModel):
    content: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float
    # extra detail for debugging; assignment cares about the fields above
    retrieved_chunks: list[RetrievedChunk]


@app.get("/")
def root():
    return {"message": "Agentic AI RAG API is running. POST /chat with {\"query\": \"...\"}"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = rag_graph.invoke(
        {
            "question": request.query,
            "context": [],
            "documents": [],
            "scores": [],
            "answer": "",
            "score": 0.0,
        }
    )

    docs = result.get("documents") or []
    scores = result.get("scores") or []
    chunks = []
    for i, document in enumerate(docs):
        chunks.append(
            RetrievedChunk(
                content=document.page_content,
                score=round(float(scores[i]), 4) if i < len(scores) else None,
                metadata=_jsonish(document.metadata or {}),
            )
        )

    return ChatResponse(
        query=request.query,
        final_answer=result["answer"],
        retrieved_context_chunks=[c.content for c in chunks],
        confidence_score=float(result["score"]),
        retrieved_chunks=chunks,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
