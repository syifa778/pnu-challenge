import uuid
from fastapi import FastAPI
from pydantic import BaseModel

from core.retriever import retrieve
from core.generator import generate
from core.memory_store import get_memory, append_memory


app = FastAPI(title="Qwen RAG API")


# =========================
# Request Models
# =========================

class GenerateRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5


# =========================
# Session Management
# =========================

@app.post("/start")
def start_session():
    """
    Start a new conversation session.
    The session_id is opaque and contains no PII.
    """
    return {
        "session_id": str(uuid.uuid4())
    }


# =========================
# RAG Endpoints
# =========================

@app.post("/retrieve")
def retrieve_api(query: str, top_k: int = 5):
    """
    Stateless retrieval endpoint.
    Returns raw RAG results (no sanitization).
    """
    return retrieve(query, top_k)


@app.post("/generate")
def generate_api(req: GenerateRequest):
    """
    Memory-aware RAG generation endpoint.
    """
    # 1. Load conversation memory
    memory = get_memory(req.session_id)

    # 2. Retrieve relevant documents
    retrieved = retrieve(req.query, req.top_k)

    # 3. Generate answer with memory + RAG context
    answer = generate(
        query=req.query,
        retrieved=retrieved,
        conversation_memory=memory
    )

    # 4. Update memory (semantic + sanitized)
    append_memory(req.session_id, f"User asked: {req.query}")
    append_memory(req.session_id, f"Assistant answered: {answer}")

    return {
        "session_id": req.session_id,
        "query": req.query,
        "answer": answer,
        "evidence": retrieved
    }
