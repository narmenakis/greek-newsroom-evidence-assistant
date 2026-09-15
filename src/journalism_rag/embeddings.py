"""Provider-neutral Hugging Face embedding setup and model-specific prompts."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_huggingface import HuggingFaceEmbeddings


E5_QUERY_INSTRUCTION = (
    "Instruct: Given a journalistic question, retrieve relevant passages from news articles "
    "that factually and clearly answer the question\nQuery: "
)


def is_e5_model(model_name: str) -> bool:
    return model_name.lower().startswith("intfloat/multilingual-e5")


def format_document(model_name: str, text: str) -> str:
    """Format one corpus passage according to the selected model's contract."""

    return f"passage: {text}" if is_e5_model(model_name) else text


def format_query(model_name: str, query: str) -> str:
    """Format one search query according to the selected model's contract."""

    return f"{E5_QUERY_INSTRUCTION}{query}" if is_e5_model(model_name) else query


class PromptAwareEmbeddings:
    """LangChain-compatible embeddings with E5 document/query formatting."""

    def __init__(self, model_name: str, *, device: str, batch_size: int):
        self.model_name = model_name
        self._embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        formatted = [format_document(self.model_name, text) for text in texts]
        return self._embedder.embed_documents(formatted)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(format_query(self.model_name, text))
