import faiss
import json
import numpy as np
from core.embedder import embed

INDEX_FILE = "vector_store/index.faiss"
META_FILE = "vector_store/metadata.json"

index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "r") as f:
    METADATA = json.load(f)

def retrieve(query: str, top_k: int = 5, source: str | None = None):
    q = np.array([embed(query)]).astype("float32")
    faiss.normalize_L2(q)

    scores, ids = index.search(q, top_k * 2)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        item = METADATA[idx]
        if source and item["metadata"]["source"] != source:
            continue

        results.append({
            "score": float(score),
            "text": item["text"],
            "metadata": item["metadata"]
        })

        if len(results) == top_k:
            break

    return results
