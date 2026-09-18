"""
retriever.py
-------------
Loads the FAISS index built by knowledge_base/build_index.py and exposes a
simple `retrieve(query, k)` function that returns the top-k KB records most
relevant to a query, with similarity scores.

This is the "how knowledge is retrieved" piece the hackathon rubric asks for.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.parent
INDEX_DIR = BASE_DIR / "knowledge_base" / "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class KnowledgeRetriever:
    def __init__(self):
        index_path = INDEX_DIR / "index.faiss"
        meta_path = INDEX_DIR / "kb_metadata.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run `python knowledge_base/build_index.py` first."
            )

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.kb_records = json.load(f)

        # Load the model on first retrieval rather than during API import.
        # This lets hosted platforms detect the HTTP port before model startup.
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        return self.model

    def retrieve(self, query: str, k: int = 4, min_score: float = 0.15):
        """
        Returns a list of dicts: {record, score}, sorted by relevance.
        min_score filters out weakly-related results (cosine similarity threshold).
        """
        query_vec = self._get_model().encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")

        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < min_score:
                continue
            results.append({"record": self.kb_records[idx], "score": float(score)})
        return results

    def retrieve_by_metrics(self, metric_names: list, k_per_metric: int = 2):
        """
        Direct structured lookup (not embedding-based): pull KB records that
        explicitly list one of the given metric names in metrics_affected.
        Used when we have structured JSON input (soil %, rainfall, etc.) and
        want deterministic, guaranteed-relevant retrieval rather than
        similarity search alone.
        """
        matches = []
        for record in self.kb_records:
            if any(m in record.get("metrics_affected", []) for m in metric_names):
                matches.append(record)
        return matches[: k_per_metric * len(metric_names)]


if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    test_query = "soil organic carbon is very low and rainfall is low, monoculture wheat"
    for r in retriever.retrieve(test_query, k=3):
        print(f"[{r['score']:.3f}] {r['record']['topic']}")
