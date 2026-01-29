import ollama

EMBED_MODEL = "qwen3-embedding:0.6b"

def embed(text: str) -> list[float]:
    return ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )["embedding"]
