import ollama
from core.safety import redact_pii

LLM_MODEL = "qwen3:8b"

def generate(
    query: str,
    retrieved: list[dict],
    conversation_memory: list[str] | None = None
) -> str:
    context_blocks = []

    # 1. Retrieved RAG context
    for r in retrieved:
        text = r["text"]

        if r["metadata"].get("source") == "personas":
            text = redact_pii(text)

        context_blocks.append(text)

    # 2. Conversation memory (always sanitized)
    if conversation_memory:
        memory_block = [
            redact_pii(m) for m in conversation_memory
        ]
        context_blocks.append(
            "Previous conversation:\n" + "\n".join(memory_block)
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a cybersecurity analyst specializing in CVE and CWE analysis.

Rules:
- Use ONLY the provided context.
- Do NOT reveal or reconstruct personal information.
- Focus on vulnerabilities, weaknesses, impacts, and mitigations.

Context:
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
