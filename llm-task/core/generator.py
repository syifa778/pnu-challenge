import ollama
from core.safety import redact_pii

LLM_MODEL = "qwen3:8b"

def generate(query: str, retrieved: list[dict]) -> str:
    safe_context = []

    for r in retrieved:
        if r["metadata"]["source"] == "personas":
            safe_context.append(redact_pii(r["text"]))
        else:
            safe_context.append(r["text"])

    context = "\n\n".join(safe_context)

    prompt = f"""
                You are a cybersecurity analyst.

                Use ONLY the context below.

                {context}

                Question:
                {query}

                Answer concisely and factually.
            """
            
    res = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return res["message"]["content"]
