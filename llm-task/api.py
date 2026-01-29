from fastapi import FastAPI
from core.retriever import retrieve
from core.generator import generate
from core.safety import contains_pii

app = FastAPI(title="Qwen RAG API")

@app.post("/retrieve")
def retrieve_api(query: str, top_k: int = 1):
    if contains_pii(query):
        return {
            "error": "Query contains sensitive personal data."
        }

    return retrieve(query, top_k)

@app.post("/generate")
def generate_api(query: str, top_k: int = 1):
    retrieved = retrieve(query, top_k)
    answer = generate(query, retrieved)
    return {
        "query": query,
        "answer": answer,
        "evidence": retrieved
    }
