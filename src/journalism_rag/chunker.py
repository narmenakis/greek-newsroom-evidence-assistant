"""Token-aware chunking and deterministic chunk identifiers."""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    # Fast tokenizers can bypass the model's 512-token warning; we intentionally
    # encode long articles before slicing them into bounded model-sized chunks.
    if hasattr(tokenizer, "backend_tokenizer"):
        return list(tokenizer.backend_tokenizer.encode(text).ids)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return list(encoded)


def chunk_documents(
    documents: list[Document],
    tokenizer: Any,
    *,
    chunk_size_tokens: int = 384,
    chunk_overlap_tokens: int = 64,
) -> list[Document]:
    """Split documents by tokenizer tokens and attach stable metadata.

    The title is included in every chunk so retrieval can match headlines while
    the body remains the source text used for generation.
    """

    if chunk_size_tokens <= 0 or chunk_overlap_tokens < 0:
        raise ValueError("chunk_size_tokens must be positive and overlap non-negative")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
    chunks: list[Document] = []
    step = chunk_size_tokens - chunk_overlap_tokens
    for document in documents:
        title = str(document.metadata.get("title", "")).strip()
        source_text = f"Τίτλος: {title}\n\n{document.page_content}" if title else document.page_content
        ids = _token_ids(tokenizer, source_text)
        if not ids:
            continue
        article_id = str(document.metadata["article_id"])
        content_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16]
        for chunk_index, start in enumerate(range(0, len(ids), step)):
            token_slice = ids[start : start + chunk_size_tokens]
            if not token_slice:
                break
            text = tokenizer.decode(token_slice, skip_special_tokens=True).strip()
            if not text:
                continue
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "token_count": len(token_slice),
                    "content_hash": content_hash,
                    "chunk_id": f"{article_id}:{content_hash}:{chunk_index}",
                }
            )
            chunks.append(Document(page_content=text, metadata=metadata))
            if start + chunk_size_tokens >= len(ids):
                break
    return chunks
