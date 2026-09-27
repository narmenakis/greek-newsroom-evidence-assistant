"""Validated, environment-driven settings for ingestion and indexing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _nonnegative_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative")
    return parsed


LOCAL_OLLAMA_MODEL = "journalism-rag-qwen3.5:9b"
LOCAL_OLLAMA_BASE_URL = "http://localhost:11434/v1"
LOCAL_OLLAMA_MAX_TOKENS = 8192
HOSTED_DEEPSEEK_MODEL = "deepseek-flash"
HOSTED_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
HOSTED_DEEPSEEK_MAX_TOKENS = 8192


@dataclass(frozen=True)
class Settings:
    """Settings shared by the corpus loader and vector indexer.

    Values can be overridden with ``RAG_*`` environment variables.  The
    defaults are deliberately conservative for the 24 GB Apple Silicon
    demonstration machine; the older 16 GB baseline is historical only.
    """

    corpus_path: Path = Path("sample_database.csv")
    chroma_path: Path = Path("chroma_sample")
    collection_name: str = "journalism_rag__e5-large-instruct"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
    chroma_batch_size: int = 64
    # Reranking remains disabled unless explicitly enabled in configuration.
    reranker_enabled: bool = False
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_device: str = "auto"
    reranker_candidate_limit: int = 20
    reranker_evidence_limit: int = 5
    reranker_batch_size: int = 4
    reranker_max_length: int = 256
    chunk_size_tokens: int = 384
    chunk_overlap_tokens: int = 64
    enable_logging: bool = False
    show_diagnostics: bool = False
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-flash"
    llm_base_url: str = "https://api.deepseek.com"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.1
    # DeepSeek reasoning is disabled by default to preserve the visible answer budget.
    deepseek_thinking: str = "disabled"
    # Ollama reasoning is disabled by default for predictable local responses.
    ollama_reasoning_effort: str = "none"
    deepseek_api_key: str | None = field(default=None, repr=False)
    tavily_mcp_url: str = "https://mcp.tavily.com/mcp"
    tavily_timeout_seconds: float = 30.0
    tavily_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        llm_provider = env.get("RAG_LLM_PROVIDER", "deepseek").lower()
        if llm_provider == "ollama":
            default_llm_model = LOCAL_OLLAMA_MODEL
            default_llm_base_url = LOCAL_OLLAMA_BASE_URL
            default_llm_max_tokens = LOCAL_OLLAMA_MAX_TOKENS
        else:
            default_llm_model = HOSTED_DEEPSEEK_MODEL
            default_llm_base_url = HOSTED_DEEPSEEK_BASE_URL
            default_llm_max_tokens = HOSTED_DEEPSEEK_MAX_TOKENS
        settings = cls(
            corpus_path=Path(env.get("RAG_CORPUS_PATH", cls.corpus_path)),
            chroma_path=Path(env.get("RAG_CHROMA_PATH", cls.chroma_path)),
            collection_name=env.get("RAG_COLLECTION_NAME", cls.collection_name),
            embedding_model=env.get("RAG_EMBEDDING_MODEL", cls.embedding_model),
            embedding_device=env.get("RAG_EMBEDDING_DEVICE", cls.embedding_device),
            embedding_batch_size=_positive_int(
                "RAG_EMBEDDING_BATCH_SIZE",
                env.get("RAG_EMBEDDING_BATCH_SIZE", str(cls.embedding_batch_size)),
            ),
            chroma_batch_size=_positive_int(
                "RAG_CHROMA_BATCH_SIZE",
                env.get("RAG_CHROMA_BATCH_SIZE", str(cls.chroma_batch_size)),
            ),
            reranker_enabled=env.get("RAG_RERANKER_ENABLED", "0").lower()
            in {"1", "true", "yes"},
            reranker_model=env.get("RAG_RERANKER_MODEL", cls.reranker_model),
            reranker_device=env.get("RAG_RERANKER_DEVICE", cls.reranker_device).lower(),
            reranker_candidate_limit=_positive_int(
                "RAG_RERANKER_CANDIDATE_LIMIT",
                env.get("RAG_RERANKER_CANDIDATE_LIMIT", str(cls.reranker_candidate_limit)),
            ),
            reranker_evidence_limit=_positive_int(
                "RAG_RERANKER_EVIDENCE_LIMIT",
                env.get("RAG_RERANKER_EVIDENCE_LIMIT", str(cls.reranker_evidence_limit)),
            ),
            reranker_batch_size=_positive_int(
                "RAG_RERANKER_BATCH_SIZE",
                env.get("RAG_RERANKER_BATCH_SIZE", str(cls.reranker_batch_size)),
            ),
            reranker_max_length=_positive_int(
                "RAG_RERANKER_MAX_LENGTH",
                env.get("RAG_RERANKER_MAX_LENGTH", str(cls.reranker_max_length)),
            ),
            chunk_size_tokens=_positive_int(
                "RAG_CHUNK_SIZE_TOKENS",
                env.get("RAG_CHUNK_SIZE_TOKENS", str(cls.chunk_size_tokens)),
            ),
            chunk_overlap_tokens=_positive_int(
                "RAG_CHUNK_OVERLAP_TOKENS",
                env.get("RAG_CHUNK_OVERLAP_TOKENS", str(cls.chunk_overlap_tokens)),
            ),
            enable_logging=env.get("RAG_ENABLE_LOGGING", "0").lower() in {"1", "true", "yes"},
            show_diagnostics=env.get("RAG_SHOW_DIAGNOSTICS", "0").lower()
            in {"1", "true", "yes"},
            llm_provider=llm_provider,
            llm_model=env.get("RAG_LLM_MODEL", default_llm_model),
            llm_base_url=env.get("RAG_LLM_BASE_URL", default_llm_base_url).rstrip("/"),
            llm_timeout_seconds=float(env.get("RAG_LLM_TIMEOUT_SECONDS", "60")),
            llm_max_retries=_nonnegative_int(
                "RAG_LLM_MAX_RETRIES", env.get("RAG_LLM_MAX_RETRIES", "2")
            ),
            llm_max_tokens=_positive_int(
                "RAG_LLM_MAX_TOKENS",
                env.get("RAG_LLM_MAX_TOKENS", str(default_llm_max_tokens)),
            ),
            llm_temperature=float(env.get("RAG_LLM_TEMPERATURE", "0.1")),
            deepseek_thinking=env.get("RAG_DEEPSEEK_THINKING", "disabled").lower(),
            ollama_reasoning_effort=env.get(
                "RAG_OLLAMA_REASONING_EFFORT", "none"
            ).lower(),
            deepseek_api_key=env.get("DEEPSEEK_API_KEY") or None,
            tavily_mcp_url=env.get("RAG_TAVILY_MCP_URL", cls.tavily_mcp_url).rstrip("/"),
            tavily_timeout_seconds=float(env.get("RAG_TAVILY_TIMEOUT_SECONDS", "30")),
            tavily_api_key=env.get("TAVILY_API_KEY") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.collection_name.strip():
            raise ValueError("collection_name cannot be empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model cannot be empty")
        if self.embedding_device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("embedding_device must be one of cpu, cuda, mps, or auto")
        if not self.reranker_model.strip():
            raise ValueError("reranker_model cannot be empty")
        if self.reranker_device not in {"cpu", "mps", "auto"}:
            raise ValueError("reranker_device must be one of cpu, mps, or auto")
        if self.reranker_evidence_limit > self.reranker_candidate_limit:
            raise ValueError("reranker_evidence_limit cannot exceed reranker_candidate_limit")
        if self.llm_provider not in {"deepseek", "ollama"}:
            raise ValueError("llm_provider must be one of deepseek or ollama")
        if not self.llm_model.strip():
            raise ValueError("llm_model cannot be empty")
        if not self.llm_base_url.startswith(("http://", "https://")):
            raise ValueError("llm_base_url must start with http:// or https://")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be greater than zero")
        if self.llm_max_retries < 0:
            raise ValueError("llm_max_retries cannot be negative")
        if self.llm_max_tokens <= 0:
            raise ValueError("llm_max_tokens must be greater than zero")
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError("llm_temperature must be between 0 and 2")
        if self.deepseek_thinking not in {"enabled", "disabled"}:
            raise ValueError("deepseek_thinking must be enabled or disabled")
        if self.ollama_reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError(
                "ollama_reasoning_effort must be one of none, low, medium, or high"
            )
        if not self.tavily_mcp_url.startswith("https://"):
            raise ValueError("tavily_mcp_url must start with https://")
        if self.tavily_timeout_seconds <= 0:
            raise ValueError("tavily_timeout_seconds must be greater than zero")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        for name in ("embedding_batch_size", "chroma_batch_size", "chunk_size_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
