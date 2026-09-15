"""Bounded text extraction for user-uploaded document formats."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


def extract_uploaded_text(payload: bytes, media_type: str) -> str:
    """Extract UTF-8 text or PDF page text from an uploaded byte payload.

    The caller enforces the raw upload byte limit before invoking this helper.
    PDF parsing errors are converted to clear ``ValueError`` exceptions so the
    Streamlit client can reject malformed or image-only PDFs without a traceback.
    """

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("uploaded file cannot be empty")
    normalized_media_type = media_type.strip().lower()
    if normalized_media_type in {"text/plain", "text/markdown"}:
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("text upload must be valid UTF-8") from exc
    if normalized_media_type != "application/pdf":
        raise ValueError(f"unsupported media_type: {normalized_media_type}")

    try:
        reader = PdfReader(BytesIO(payload), strict=False)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf exposes several parser-specific exceptions.
        raise ValueError("could not extract text from PDF") from exc
    text = "\n\n".join(page for page in pages if page.strip())
    if not text.strip():
        raise ValueError("PDF contains no extractable text")
    return text
