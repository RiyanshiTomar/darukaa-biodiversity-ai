"""
build_index.py
----------------
Reads kb_data.json (the structured knowledge layer) and builds a local FAISS
vector index over it using sentence-transformers embeddings.

Run this once (and again whenever kb_data.json changes):
    python knowledge_base/build_index.py

Output:
    knowledge_base/faiss_index/index.faiss
    knowledge_base/faiss_index/kb_metadata.json   (id -> full KB record, for retrieval)
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
KB_PATH = BASE_DIR / "kb_data.json"
INDEX_DIR = BASE_DIR / "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, free, runs locally (384-dim)


def make_embedding_text(record: dict) -> str:
    """
    Concatenate the fields that matter for semantic retrieval into one string.
    We embed topic + content + metrics + quantified_impact so a query about
    either the practice ('cover crops') or the metric ('soil organic carbon')
    can retrieve this record.
    """
    return (
        f"Topic: {record['topic']}. "
        f"Category: {record['category']}. "
        f"Details: {record['content']} "
        f"Affects metrics: {', '.join(record['metrics_affected'])}. "
        f"Evidence: {record.get('quantified_impact', '')}"
    )


def build_index():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_records = json.load(f)

    print(f"Loaded {len(kb_records)} knowledge base records.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [make_embedding_text(r) for r in kb_records]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    # Inner product on normalized vectors == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

    # Store metadata in the SAME order the vectors were added, so
    # FAISS row i <-> kb_metadata[i]
    with open(INDEX_DIR / "kb_metadata.json", "w", encoding="utf-8") as f:
        json.dump(kb_records, f, indent=2, ensure_ascii=False)

    print(f"Index built with {index.ntotal} vectors (dim={dim}).")
    print(f"Saved to: {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
