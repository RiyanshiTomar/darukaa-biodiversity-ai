"""
Lightweight knowledge-base retrieval for the free hosting tier.

The original implementation used FAISS and SentenceTransformers. Those
packages pull in native libraries and PyTorch, which exceed the 512 MB RAM
limit on Render's free web service. This implementation keeps the same
retriever interface but ranks the small local knowledge base with
case-insensitive token and phrase overlap, using only the Python standard
library.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
KB_PATH = BASE_DIR / "knowledge_base" / "kb_data.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "very", "what", "with", "you",
}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(value.lower())
        if token not in _STOP_WORDS
    }


def _record_text(record: dict) -> str:
    return " ".join(
        [
            str(record.get("topic", "")),
            str(record.get("category", "")),
            str(record.get("content", "")),
            " ".join(record.get("metrics_affected", [])),
            str(record.get("quantified_impact", "")),
            str(record.get("source", "")),
        ]
    )


class KnowledgeRetriever:
    """Ranks local evidence records without loading a machine-learning model."""

    def __init__(self):
        if not KB_PATH.exists():
            raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")

        with KB_PATH.open("r", encoding="utf-8") as file:
            self.kb_records = json.load(file)

        self._record_tokens = [_tokens(_record_text(record)) for record in self.kb_records]

    def retrieve(self, query: str, k: int = 4, min_score: float = 0.05):
        """Return the highest-overlap records as {record, score} dictionaries."""
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        ranked = []
        query_lower = query.lower()
        for index, record_tokens in enumerate(self._record_tokens):
            overlap = query_tokens.intersection(record_tokens)
            if not overlap:
                continue

            # Normalize by the query size so short, focused queries still score well.
            score = len(overlap) / len(query_tokens)
            record_text = _record_text(self.kb_records[index]).lower()
            if any(
                phrase in record_text
                for phrase in (
                    "soil organic carbon",
                    "cover crop",
                    "pollinator",
                    "habitat connectivity",
                    "integrated pest management",
                )
                if phrase in query_lower
            ):
                score += 0.15

            if score >= min_score:
                ranked.append({"record": self.kb_records[index], "score": min(score, 1.0)})

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:k]

    def retrieve_by_metrics(self, metric_names: list, k_per_metric: int = 2):
        """
        Deterministic lookup for structured inputs. Records are returned when
        they explicitly mention one of the requested affected metrics.
        """
        normalized_metrics = {str(metric).lower() for metric in metric_names}
        matches = []
        for record in self.kb_records:
            record_metrics = {
                str(metric).lower() for metric in record.get("metrics_affected", [])
            }
            if normalized_metrics.intersection(record_metrics):
                matches.append(record)
        return matches[: k_per_metric * len(normalized_metrics)]


if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    test_query = "soil organic carbon is very low and rainfall is low, monoculture wheat"
    for result in retriever.retrieve(test_query, k=3):
        print(f"[{result['score']:.3f}] {result['record']['topic']}")
