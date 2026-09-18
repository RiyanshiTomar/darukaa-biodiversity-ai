"""
Validate the local knowledge base.

The free-tier deployment uses the dependency-free retriever in
``rag/retriever.py`` and does not need a FAISS index. This script is retained
for compatibility with older Render build commands.
"""

import json
from pathlib import Path

KB_PATH = Path(__file__).parent / "kb_data.json"


def build_index():
    with KB_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("kb_data.json must contain a non-empty JSON list")

    print(f"Validated {len(records)} knowledge base records.")
    print("No vector index is required for the lightweight free-tier retriever.")


if __name__ == "__main__":
    build_index()
