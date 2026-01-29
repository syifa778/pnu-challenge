import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from prepare_data import prepare_all_documents


# =====================
# Configuration
# =====================
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_STORE_DIR = Path("vector_store")
INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"


def main():
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare documents
    documents = prepare_all_documents()
    if not documents:
        raise RuntimeError("No documents found to ingest")

    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    # Embed texts
    print("Embedding documents...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]

    # Build FAISS index (cosine similarity via inner product)
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index + metadata
    faiss.write_index(index, str(INDEX_FILE))

    documents_for_store = [
        {
            "text": text,
            "metadata": meta
        }
        for text, meta in zip(texts, metadatas)
    ]

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(documents_for_store, f, indent=2)


    print("Ingestion complete!")
    print(f"Total vectors: {index.ntotal}")
    print(f"Index saved to: {INDEX_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()
