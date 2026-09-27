"""Command-line grounded answer tool for the selected provider."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config import Settings
from .pipeline import GroundedRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question using retrieved corpus evidence")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    result = GroundedRAG(Settings.from_env()).answer(args.query, limit=args.limit)
    generation = result.generation
    print(
        json.dumps(
            {
                "status": result.status.value,
                "answer": result.answer,
                "abstention_reason": result.abstention_reason,
                "used_source_ids": result.used_source_ids,
                "citation_validation": asdict(result.citation_validation),
                "sources": [asdict(source) for source in result.sources],
                "provider": generation.provider if generation else None,
                "model": generation.model if generation else None,
                "latency_seconds": round(generation.latency_seconds, 3) if generation else None,
                "total_tokens": generation.total_tokens if generation else None,
                "estimated_cost_usd": generation.estimated_cost_usd if generation else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
