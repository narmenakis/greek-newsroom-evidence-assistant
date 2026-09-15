"""Command-line retrieval smoke test for the local AI core."""

from __future__ import annotations

import argparse
import json

from .config import Settings
from .retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve source chunks from the local index")
    parser.add_argument("query", help="Question to retrieve evidence for")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    results = HybridRetriever(Settings.from_env()).retrieve(args.query, limit=args.limit)
    print(
        json.dumps(
            [
                {
                    "rank": result.rank,
                    "score": round(result.score, 6),
                    "article_id": result.document.metadata.get("article_id"),
                    "url": result.document.metadata.get("url"),
                    "title": result.document.metadata.get("title"),
                    "text": result.document.page_content,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
