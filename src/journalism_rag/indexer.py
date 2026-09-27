"""Resumable local Chroma index builder for the journalism RAG core."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from langchain_chroma import Chroma
from transformers import AutoTokenizer

from .chunker import chunk_documents
from .config import Settings
from .embeddings import PromptAwareEmbeddings, is_e5_model
from .loader import load_corpus


INDEX_CONTRACT_VERSION = "journalism-rag-index-v2"
INDEXER_COMMAND = "PYTHONPATH=src ./.venv/bin/python -m journalism_rag.indexer"


class IndexContractError(RuntimeError):
    """Raised when a persisted collection cannot be safely resumed or queried."""


def _collection_contract(settings: Settings) -> dict[str, str | int]:
    """Return the persisted settings that determine vector compatibility."""

    prompt_contract = "e5-instruct-v1" if is_e5_model(settings.embedding_model) else "plain-v1"
    return {
        "journalism_rag_contract": INDEX_CONTRACT_VERSION,
        "journalism_rag_embedding_model": settings.embedding_model,
        "journalism_rag_embedding_prompt": prompt_contract,
        "journalism_rag_chunk_size_tokens": settings.chunk_size_tokens,
        "journalism_rag_chunk_overlap_tokens": settings.chunk_overlap_tokens,
    }


def _validate_collection(
    collection: Chroma,
    settings: Settings,
    *,
    expected_chunk_ids: set[str] | None = None,
) -> set[str]:
    """Validate persisted metadata and return IDs currently in the collection."""

    expected_contract = _collection_contract(settings)
    actual_contract = getattr(getattr(collection, "_collection", None), "metadata", None)
    if actual_contract != expected_contract:
        raise IndexContractError(
            f"Chroma collection {settings.collection_name!r} has no compatible "
            "journalism-rag index contract. Run "
            f"`{INDEXER_COMMAND} --reset` "
            "to rebuild only this collection."
        )
    existing_ids = {str(item) for item in collection.get().get("ids", [])}
    if expected_chunk_ids is not None:
        stale_ids = sorted(existing_ids - expected_chunk_ids)
        if stale_ids:
            preview = ", ".join(stale_ids[:3])
            suffix = "..." if len(stale_ids) > 3 else ""
            raise IndexContractError(
                f"Chroma collection {settings.collection_name!r} contains "
                f"{len(stale_ids)} stale chunk(s) ({preview}{suffix}). Run "
                f"`{INDEXER_COMMAND} --reset` "
                "to rebuild only this collection; ordinary resume will not delete "
                "stale data."
            )
    return existing_ids


def _device(settings: Settings) -> str:
    if settings.embedding_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if settings.embedding_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RAG_EMBEDDING_DEVICE=cuda was requested but CUDA is unavailable")
    if settings.embedding_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("RAG_EMBEDDING_DEVICE=mps was requested but MPS is unavailable")
    return settings.embedding_device


def _embedding_function(settings: Settings) -> PromptAwareEmbeddings:
    return PromptAwareEmbeddings(
        settings.embedding_model,
        device=_device(settings),
        batch_size=settings.embedding_batch_size,
    )


def _reset_collection(settings: Settings) -> None:
    """Delete only the selected collection, preserving sibling collections."""

    if not settings.chroma_path.exists():
        return
    collection = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_path),
    )
    collection.delete_collection()


def build_index(settings: Settings, *, reset: bool = False) -> dict[str, int | str]:
    settings.validate()
    if reset:
        _reset_collection(settings)
    documents = load_corpus(settings.corpus_path)
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    chunks = chunk_documents(
        documents,
        tokenizer,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    db = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_path),
        embedding_function=_embedding_function(settings),
        collection_metadata=_collection_contract(settings),
    )
    expected_chunk_ids = {str(chunk.metadata["chunk_id"]) for chunk in chunks}
    existing_ids = _validate_collection(
        db,
        settings,
        expected_chunk_ids=expected_chunk_ids,
    )
    pending = [chunk for chunk in chunks if chunk.metadata["chunk_id"] not in existing_ids]
    for start in range(0, len(pending), settings.chroma_batch_size):
        batch = pending[start : start + settings.chroma_batch_size]
        db.add_documents(
            documents=batch,
            ids=[chunk.metadata["chunk_id"] for chunk in batch],
        )
        print(f"Indexed {min(start + len(batch), len(pending))}/{len(pending)} chunks")
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "new_chunks": len(pending),
        "collection": settings.collection_name,
        "path": str(settings.chroma_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or resume the local Chroma index")
    parser.add_argument("--reset", action="store_true", help="Delete the configured index first")
    parser.add_argument("--corpus", type=Path, help="Override RAG_CORPUS_PATH")
    parser.add_argument("--chroma-path", type=Path, help="Override RAG_CHROMA_PATH")
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--chroma-batch-size", type=int)
    args = parser.parse_args()
    settings = Settings.from_env()
    overrides = {
        "corpus_path": args.corpus or settings.corpus_path,
        "chroma_path": args.chroma_path or settings.chroma_path,
        "embedding_batch_size": args.embedding_batch_size or settings.embedding_batch_size,
        "chroma_batch_size": args.chroma_batch_size or settings.chroma_batch_size,
    }
    settings = Settings(**{**settings.__dict__, **overrides})
    print(build_index(settings, reset=args.reset))


if __name__ == "__main__":
    main()
