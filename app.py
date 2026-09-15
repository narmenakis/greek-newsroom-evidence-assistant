"""Thin Streamlit client for the reusable journalism RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("STREAMLIT_WATCHER_TYPE", "none")

import streamlit as st
from dotenv import load_dotenv

from journalism_rag.config import (
    HOSTED_DEEPSEEK_BASE_URL,
    HOSTED_DEEPSEEK_MAX_TOKENS,
    HOSTED_DEEPSEEK_MODEL,
    LOCAL_OLLAMA_BASE_URL,
    LOCAL_OLLAMA_MAX_TOKENS,
    LOCAL_OLLAMA_MODEL,
    Settings,
)
from journalism_rag.llm import LLMError
from journalism_rag.loader import CorpusError, load_corpus
from journalism_rag.mcp_web import TavilyMCPClient, TavilyMCPError
from journalism_rag.pipeline import GroundedAnswer, GroundedRAG
from journalism_rag.retrieval import HybridRetriever
from journalism_rag.reranker import CrossEncoderReranker, create_reranker
from journalism_rag.research_brief import (
    ResearchBriefAgent,
    ResearchBriefError,
    ResearchBriefRun,
)
from journalism_rag.user_sources import (
    InMemorySourceStore,
    IndexedSourceCatalog,
    RetrievalScope,
    ScopedRetriever,
    UploadedSourceIndex,
    UploadLimits,
    extract_uploaded_text,
)


MAX_CHAT_HISTORY = 40
# Kept as safe defaults for direct helper calls; the running app passes the
# validated Settings flags explicitly after loading .env.
ENABLE_LOGGING = False
SHOW_DIAGNOSTICS = False

PROVIDER_LABELS = {
    "Τοπικό Ollama": "ollama",
    "Hosted DeepSeek": "deepseek",
}
PROVIDER_NAMES = {value: key for key, value in PROVIDER_LABELS.items()}


def provider_privacy_notice(settings: Settings) -> str:
    """Describe provider-specific data transfer before a question is sent."""

    if settings.llm_provider == "deepseek":
        return (
            "Τα ανακτημένα αποσπάσματα αποστέλλονται στο hosted DeepSeek. "
            "Σε ερωτήσεις συνέχειας, μπορεί να αποσταλούν επίσης τα πρόσφατα "
            "μηνύματα της συνομιλίας για τη δημιουργία ερωτήματος αναζήτησης. "
            "Απαιτείται DEEPSEEK_API_KEY."
        )
    return (
        "Τα ανακτημένα αποσπάσματα αποστέλλονται στο τοπικό Ollama "
        f"({settings.llm_base_url})."
    )


def settings_for_provider(settings: Settings, provider: str) -> Settings:
    """Return runtime settings for one explicit provider selection.

    Provider-specific defaults prevent a runtime switch from accidentally
    sending a DeepSeek model name to Ollama (or vice versa). The original
    settings object is never mutated, so Streamlit can cache each pipeline by
    its complete provider/model configuration.
    """

    if provider not in PROVIDER_NAMES:
        raise ValueError("provider must be one of ollama or deepseek")
    if provider == settings.llm_provider:
        return settings
    if provider == "ollama":
        return replace(
            settings,
            llm_provider="ollama",
            llm_model=LOCAL_OLLAMA_MODEL,
            llm_base_url=LOCAL_OLLAMA_BASE_URL,
            llm_max_tokens=LOCAL_OLLAMA_MAX_TOKENS,
        )
    return replace(
        settings,
        llm_provider="deepseek",
        llm_model=HOSTED_DEEPSEEK_MODEL,
        llm_base_url=HOSTED_DEEPSEEK_BASE_URL,
        llm_max_tokens=HOSTED_DEEPSEEK_MAX_TOKENS,
    )


def retrieval_cache_key(settings: Settings) -> tuple[Any, ...]:
    """Return settings that determine heavyweight retrieval/reranking objects."""

    return (
        str(settings.corpus_path),
        str(settings.chroma_path),
        settings.collection_name,
        settings.embedding_model,
        settings.embedding_device,
        settings.embedding_batch_size,
        settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
        settings.reranker_enabled,
        settings.reranker_model,
        settings.reranker_device,
        settings.reranker_candidate_limit,
        settings.reranker_evidence_limit,
        settings.reranker_batch_size,
        settings.reranker_max_length,
    )


def pipeline_cache_key(settings: Settings) -> tuple[Any, ...]:
    """Return settings affecting generation without storing raw credentials."""

    credential_fingerprint = (
        hashlib.sha256(settings.deepseek_api_key.encode("utf-8")).hexdigest()
        if settings.deepseek_api_key
        else "unset"
    )

    return (
        *retrieval_cache_key(settings),
        settings.llm_provider,
        settings.llm_model,
        settings.llm_base_url,
        settings.llm_timeout_seconds,
        settings.llm_max_retries,
        settings.llm_max_tokens,
        settings.llm_temperature,
        settings.deepseek_thinking,
        settings.ollama_reasoning_effort,
        credential_fingerprint,
    )


@st.cache_resource
def load_retrieval_stack(
    cache_key: tuple[Any, ...],
    _settings: Settings,
) -> tuple[HybridRetriever, CrossEncoderReranker | None]:
    """Load provider-independent retrieval and optional reranking once."""

    del cache_key  # The key is consumed by Streamlit; settings build the objects.
    retriever = HybridRetriever(_settings)
    reranker = None
    if _settings.reranker_enabled:
        reranker = create_reranker(
            _settings.reranker_model,
            device=_settings.reranker_device,
            max_length=_settings.reranker_max_length,
            batch_size=_settings.reranker_batch_size,
            local_files_only=True,
        )
    return retriever, reranker


@st.cache_resource
def load_rag_pipeline(
    cache_key: tuple[Any, ...],
    _settings: Settings,
    _retriever: HybridRetriever,
    _reranker: CrossEncoderReranker | None,
) -> GroundedRAG:
    """Load one lightweight provider-specific pipeline around shared resources."""

    del cache_key  # The key is consumed by Streamlit; settings build the object.
    return GroundedRAG(
        _settings,
        retriever=_retriever,
        reranker=_reranker,
    )


def get_upload_catalog(settings: Settings) -> IndexedSourceCatalog:
    """Return the source registry stored only in the current Streamlit session."""

    catalog = st.session_state.get("upload_catalog")
    if catalog is None:
        catalog = IndexedSourceCatalog(
            InMemorySourceStore(),
            UploadedSourceIndex(settings=settings),
        )
        st.session_state.upload_catalog = catalog
    return catalog


def ingest_uploaded_file(
    upload_catalog: IndexedSourceCatalog,
    uploaded_file: Any,
    *,
    limits: UploadLimits | None = None,
) -> bool:
    """Validate, extract, and index one Streamlit upload.

    The caller handles the returned duplicate/created status and catches the
    ``ValueError`` raised for an invalid file. Keeping one-file ingestion here
    lets the UI process a batch without allowing one bad file to discard the
    valid files in that batch.
    """

    if uploaded_file is None:
        raise ValueError("Επίλεξε πρώτα ένα αρχείο.")
    source_limits = limits or UploadLimits()
    filename = str(uploaded_file.name)
    payload = uploaded_file.getvalue()
    media_type = (
        "text/markdown"
        if filename.lower().endswith((".md", ".markdown"))
        else "application/pdf"
        if filename.lower().endswith(".pdf")
        else "text/plain"
    )
    if len(payload) > source_limits.max_source_bytes:
        raise ValueError("source exceeds max_source_bytes")
    content = extract_uploaded_text(payload, media_type)
    result = upload_catalog.add_source(
        filename,
        content,
        media_type=media_type,
        original_filename=filename,
    )
    return result.created


@st.cache_data
def load_filter_options(corpus_path: str) -> tuple[list[str], list[str], list[str]]:
    """Read filter choices from validated corpus metadata without loading a model."""

    documents = load_corpus(corpus_path)
    authors = sorted({str(doc.metadata["author"]) for doc in documents if doc.metadata.get("author")})
    websites = sorted(
        {str(doc.metadata["website"]) for doc in documents if doc.metadata.get("website")}
    )
    sections = sorted(
        {str(doc.metadata["section"]) for doc in documents if doc.metadata.get("section")}
    )
    return authors, websites, sections


def date_to_timestamp(value: date, *, end_of_day: bool = False) -> float:
    """Convert a date-only filter boundary to a Unix timestamp in UTC."""

    boundary = datetime.max.time() if end_of_day else datetime.min.time()
    return datetime.combine(value, boundary, tzinfo=timezone.utc).timestamp()


def build_where_filter(
    author_filter: list[str],
    website_filter: list[str],
    section_filter: list[str],
    start_date: date | None,
    end_date: date | None,
) -> dict[str, Any] | None:
    """Translate sidebar selections into a Chroma metadata filter."""

    clauses: list[dict[str, Any]] = []
    if section_filter:
        clauses.append({"section": {"$in": [value.strip() for value in section_filter]}})
    if author_filter:
        clauses.append({"author": {"$in": [value.strip() for value in author_filter]}})
    if website_filter:
        clauses.append({"website": {"$in": [value.strip() for value in website_filter]}})
    if start_date:
        clauses.append({"datetime": {"$gte": date_to_timestamp(start_date)}})
    if end_date:
        clauses.append({"datetime": {"$lte": date_to_timestamp(end_date, end_of_day=True)}})
    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def answer_query(
    query_text: str,
    rag: GroundedRAG,
    *,
    author_filter: list[str],
    website_filter: list[str],
    section_filter: list[str],
    start_date: date | None,
    end_date: date | None,
    max_docs: int,
    scope: RetrievalScope | str = RetrievalScope.CORPUS_ONLY,
    history: Sequence[Mapping[str, str]] | None = None,
) -> GroundedAnswer:
    """Adapt Streamlit controls to the provider-neutral RAG interface."""

    where = build_where_filter(
        author_filter,
        website_filter,
        section_filter,
        start_date,
        end_date,
    )
    answer_kwargs: dict[str, Any] = {"limit": max_docs, "where": where}
    selected_scope = RetrievalScope(scope)
    if selected_scope is not RetrievalScope.CORPUS_ONLY:
        answer_kwargs["scope"] = selected_scope
    if history is not None:
        answer_kwargs["history"] = history
    return rag.answer(query_text, **answer_kwargs)


def stream_answer_query(
    query_text: str,
    rag: GroundedRAG,
    *,
    author_filter: list[str],
    website_filter: list[str],
    section_filter: list[str],
    start_date: date | None,
    end_date: date | None,
    max_docs: int,
    scope: RetrievalScope | str = RetrievalScope.CORPUS_ONLY,
    history: Sequence[Mapping[str, str]] | None = None,
    on_update: Callable[[str], None] | None = None,
) -> GroundedAnswer:
    """Adapt Streamlit controls to the validated direct-QA stream."""

    where = build_where_filter(
        author_filter,
        website_filter,
        section_filter,
        start_date,
        end_date,
    )
    answer_kwargs: dict[str, Any] = {"limit": max_docs, "where": where}
    selected_scope = RetrievalScope(scope)
    if selected_scope is not RetrievalScope.CORPUS_ONLY:
        answer_kwargs["scope"] = selected_scope
    if history is not None:
        answer_kwargs["history"] = history
    answer_kwargs["on_update"] = on_update
    return rag.stream_answer(query_text, **answer_kwargs)


def provisional_answer_text(text: str) -> str:
    """Format retrieval feedback or the currently streamed direct answer."""

    return f"{text}▌" if text else "Ανάκτηση σχετικών πηγών…"


def research_brief_query(
    query_text: str,
    agent: ResearchBriefAgent,
    *,
    author_filter: list[str],
    website_filter: list[str],
    section_filter: list[str],
    start_date: date | None,
    end_date: date | None,
    max_docs: int,
    scope: RetrievalScope | str = RetrievalScope.CORPUS_ONLY,
    history: Sequence[Mapping[str, str]] | None = None,
) -> ResearchBriefRun:
    """Adapt the same Streamlit filters to the opt-in research workflow."""

    where = build_where_filter(
        author_filter,
        website_filter,
        section_filter,
        start_date,
        end_date,
    )
    return agent.run(
        query_text,
        limit=max_docs,
        where=where,
        scope=RetrievalScope(scope),
        history=history,
    )


def _published_at_label(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="minutes")


def source_payload(result: GroundedAnswer) -> list[dict[str, Any]]:
    """Create compact, serializable source rows for display and chat history."""

    used = set(result.used_source_ids)
    return [
        {
            "source_id": source.source_id,
            "title": source.title or "Τίτλος λείπει",
            "url": source.url,
            "outlet": source.outlet,
            "source_type": source.source_type,
            "score": round(source.score, 4),
            "used": source.source_id in used,
            "published_at": _published_at_label(source.published_at),
            "retrieved_at": None,
        }
        for source in result.sources
    ]


def research_source_payload(result: ResearchBriefRun) -> list[dict[str, Any]]:
    """Create source rows while retaining live-web publication and retrieval times."""

    used = set(result.citation_validation.cited_source_ids)
    web_evidence_by_url = {evidence.url: evidence for evidence in result.web_evidence}
    rows: list[dict[str, Any]] = []
    for source in result.sources:
        evidence = web_evidence_by_url.get(source.url or "")
        rows.append(
            {
                "source_id": source.source_id,
                "title": source.title or "Τίτλος λείπει",
                "url": source.url,
                "outlet": source.outlet,
                "source_type": source.source_type,
                "score": round(source.score, 4),
                "used": source.source_id in used,
                "published_at": (
                    evidence.published_at.isoformat(timespec="minutes")
                    if evidence is not None and evidence.published_at is not None
                    else _published_at_label(source.published_at)
                ),
                "retrieved_at": (
                    evidence.retrieved_at.isoformat(timespec="minutes")
                    if evidence is not None
                    else None
                ),
            }
        )
    return rows


def generation_metrics(result: GroundedAnswer) -> dict[str, Any]:
    generation = result.generation
    if generation is None:
        return {
            "status": result.status.value,
            "abstention_reason": result.abstention_reason,
            "provider": None,
            "model": None,
            "latency_seconds": 0.0,
            "total_tokens": None,
            "estimated_cost_usd": None,
        }
    return {
        "status": result.status.value,
        "abstention_reason": result.abstention_reason,
        "provider": generation.provider,
        "model": generation.model,
        "latency_seconds": round(generation.latency_seconds, 3),
        "total_tokens": generation.total_tokens,
        "estimated_cost_usd": generation.estimated_cost_usd,
    }


def citation_diagnostics(
    result: GroundedAnswer,
    *,
    retrieval_scope: RetrievalScope | str | None = None,
    uploaded_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Return safe structural details for debugging a rejected answer.

    The rejected model answer itself is intentionally not returned here.
    """

    validation = result.citation_validation
    generation = result.generation
    return {
        "status": result.status.value,
        "abstention_reason": result.abstention_reason,
        "invalid_source_ids": list(validation.invalid_source_ids),
        "uncited_claims": list(validation.uncited_claims),
        "coverage_warning": validation.coverage_warning,
        "cited_source_ids": list(validation.cited_source_ids),
        "retrieved_source_count": len(result.sources),
        "retrieved_source_ids": [source.source_id for source in result.sources],
        "retrieval_query": result.retrieval_query,
        "retrieval_scope": getattr(retrieval_scope, "value", retrieval_scope),
        "uploaded_source_count": len(uploaded_source_ids),
        "uploaded_source_ids": list(uploaded_source_ids),
        "generation": None
        if generation is None
        else {
            "provider": generation.provider,
            "model": generation.model,
            "finish_reason": generation.finish_reason,
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": generation.total_tokens,
            "latency_seconds": round(generation.latency_seconds, 3),
            "estimated_cost_usd": generation.estimated_cost_usd,
            "output_characters": len(generation.text),
        },
    }


def failure_diagnostics(
    error: BaseException,
    *,
    query: str | None,
    settings: Settings,
    retrieval_scope: RetrievalScope | str | None = None,
    uploaded_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Return safe diagnostics for an exception without exposing credentials or evidence."""

    return {
        "error_type": type(error).__name__,
        "error": str(error),
        "query": query,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "max_tokens": settings.llm_max_tokens,
        "temperature": settings.llm_temperature,
        "deepseek_thinking": settings.deepseek_thinking,
        "retrieval_scope": getattr(retrieval_scope, "value", retrieval_scope),
        "uploaded_source_count": len(uploaded_source_ids),
        "uploaded_source_ids": list(uploaded_source_ids),
    }


def render_failure_diagnostics(
    error: BaseException,
    *,
    query: str | None,
    settings: Settings,
    retrieval_scope: RetrievalScope | str | None = None,
    uploaded_source_ids: Sequence[str] = (),
) -> None:
    """Always display safe failure details so users can diagnose a failed turn."""

    with st.expander("🛠 Τεχνικά στοιχεία αποτυχίας (debug)"):
        st.json(
            failure_diagnostics(
                error,
                query=query,
                settings=settings,
                retrieval_scope=retrieval_scope,
                uploaded_source_ids=uploaded_source_ids,
            )
        )


def debug_print_turn(
    original_query: str,
    result: GroundedAnswer,
    *,
    retrieval_scope: RetrievalScope | str | None = None,
    uploaded_source_ids: Sequence[str] = (),
    show_diagnostics: bool | None = None,
) -> None:
    """Print failure details always; include raw output only for opt-in debugging."""

    diagnostics_enabled = SHOW_DIAGNOSTICS if show_diagnostics is None else show_diagnostics
    is_failure = result.abstention_reason is not None
    if not diagnostics_enabled and not is_failure:
        return
    print("\n[RAG DEBUG] ------------------------------", flush=True)
    print(f"[RAG DEBUG] original query: {original_query}", flush=True)
    print(
        f"[RAG DEBUG] rewritten retrieval query: {result.retrieval_query or original_query}",
        flush=True,
    )
    if result.generation is None:
        print("[RAG DEBUG] raw model summary: <model was not called>", flush=True)
    elif diagnostics_enabled:
        print(
            "[RAG DEBUG] raw model summary (unvalidated): "
            f"{result.generation.text}",
            flush=True,
        )
    else:
        print("[RAG DEBUG] raw model summary: <hidden; enable RAG_SHOW_DIAGNOSTICS=1>", flush=True)
    print(f"[RAG DEBUG] displayed answer: {result.answer}", flush=True)
    print(f"[RAG DEBUG] status: {result.status.value}", flush=True)
    if result.abstention_reason:
        print(f"[RAG DEBUG] abstention reason: {result.abstention_reason}", flush=True)
        print(
            "[RAG DEBUG] validation details: "
            f"{citation_diagnostics(result, retrieval_scope=retrieval_scope, uploaded_source_ids=uploaded_source_ids)}",
            flush=True,
        )
    print("[RAG DEBUG] ------------------------------\n", flush=True)


def debug_print_failure(
    original_query: str,
    error: BaseException,
    settings: Settings,
    *,
    retrieval_scope: RetrievalScope | str | None = None,
    uploaded_source_ids: Sequence[str] = (),
) -> None:
    """Always print safe exception diagnostics for a failed question."""

    print("\n[RAG DEBUG] ------------------------------", flush=True)
    print(
        "[RAG DEBUG] failure: "
        f"{failure_diagnostics(error, query=original_query, settings=settings, retrieval_scope=retrieval_scope, uploaded_source_ids=uploaded_source_ids)}",
        flush=True,
    )
    print("[RAG DEBUG] ------------------------------\n", flush=True)


def log_interaction(user_id: str, question: str, answer: str) -> None:
    entry = {
        "user": user_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
    }
    with open("chat_logs.jsonl", "a", encoding="utf-8") as output:
        output.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_metrics(user_id: str, question: str, metrics: dict[str, Any], word_count: int) -> None:
    entry = {
        "user": user_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "metrics": metrics,
        "word_count": word_count,
    }
    with open("timings_log.jsonl", "a", encoding="utf-8") as output:
        output.write(json.dumps(entry, ensure_ascii=False) + "\n")


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("🔗 Δες τις Πηγές"):
        for source in sources:
            used_label = "χρησιμοποιήθηκε" if source["used"] else "ανακτήθηκε"
            score_label = (
                ""
                if source.get("source_type") == "web"
                else f", score {source['score']}"
            )
            st.write(
                f"{source['source_id']} — {source['title']} "
                f"({used_label}{score_label})"
            )
            if source.get("source_type"):
                st.caption(f"Τύπος: {source['source_type']}")
            if source.get("outlet"):
                st.caption(str(source["outlet"]))
            if source.get("published_at"):
                st.caption(f"Δημοσίευση: {source['published_at']}")
            if source.get("retrieved_at"):
                st.caption(f"Ανάκτηση από το web: {source['retrieved_at']}")
            if source.get("url"):
                button_label = (
                    "Άνοιγμα web πηγής"
                    if source.get("source_type") == "web"
                    else "Άνοιγμα άρθρου"
                )
                st.link_button(button_label, str(source["url"]))


def render_diagnostics(
    result: GroundedAnswer,
    *,
    retrieval_scope: RetrievalScope | str | None = None,
    uploaded_source_ids: Sequence[str] = (),
) -> None:
    """Show safe diagnostics automatically for any abstained answer."""

    if result.abstention_reason is None:
        return
    with st.expander("🛠 Τεχνικά στοιχεία αποτυχίας (debug)"):
        st.json(
            citation_diagnostics(
                result,
                retrieval_scope=retrieval_scope,
                uploaded_source_ids=uploaded_source_ids,
            )
        )


def render_history() -> None:
    for role, message, sources in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
            if role == "assistant":
                render_sources(sources)


def require_user_id() -> bool:
    if st.session_state.get("user_id"):
        return True
    st.title("Καλώς ήρθες στο Greek Journalism RAG Chatbot! 💬")
    email_input = st.text_input("Παρακαλώ γράψε το email σου για να συνεχίσεις:")
    if st.button("Υποβολή"):
        if re.match(r"^[\w.-]+@[\w.-]+\.\w+$", email_input.strip()):
            st.session_state.user_id = email_input.strip()
            st.rerun()
        else:
            st.error("Παρακαλώ εισάγετε ένα έγκυρο email.")
    return False


def main() -> None:
    st.set_page_config(page_title="Greek Journalism RAG Chatbot", page_icon="🤖")
    # Re-read the ignored local config on every Streamlit run so model changes
    # take effect after a browser refresh instead of waiting for a process restart.
    load_dotenv(override=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if not require_user_id():
        return

    base_settings = Settings.from_env()
    if base_settings.llm_provider not in PROVIDER_NAMES:
        st.error(
            "Μη υποστηριζόμενος πάροχος στο .env. Χρησιμοποίησε `ollama` ή `deepseek`."
        )
        return
    st.title("Greek Journalism RAG Chatbot 💬")
    st.sidebar.markdown(f"### Συνδεδεμένος ως:\n**{st.session_state.user_id}**")
    selected_provider_label = st.sidebar.selectbox(
        "Μοντέλο απάντησης",
        options=list(PROVIDER_LABELS),
        index=list(PROVIDER_LABELS.values()).index(base_settings.llm_provider),
        key="llm_provider_choice",
        help="Η επιλογή ισχύει αμέσως για τις επόμενες ερωτήσεις και δεν αλλάζει το .env.",
    )
    selected_provider = PROVIDER_LABELS[selected_provider_label]
    settings = settings_for_provider(base_settings, selected_provider)
    st.sidebar.caption(f"Ενεργός πάροχος: {selected_provider_label}")
    st.sidebar.caption(f"Μοντέλο: `{settings.llm_model}`")
    if selected_provider == "deepseek":
        st.sidebar.warning(provider_privacy_notice(settings))
    else:
        st.sidebar.info(provider_privacy_notice(settings))

    st.sidebar.header("🧭 Λειτουργία")
    mode_labels = {
        "Άμεση απάντηση από τις πηγές": "direct",
        "Live research brief": "research",
    }
    selected_mode_label = st.sidebar.radio(
        "Τύπος απάντησης",
        options=list(mode_labels),
        index=0,
        key="answer_mode",
    )
    answer_mode = mode_labels[selected_mode_label]
    if answer_mode == "research":
        st.sidebar.warning(
            "Η λειτουργία κάνει live εξωτερική αναζήτηση μέσω Tavily MCP. "
            "Τα αποτελέσματα του web αποστέλλονται στο επιλεγμένο μοντέλο για σύνθεση."
        )

    if settings.llm_provider == "deepseek" and not settings.deepseek_api_key:
        st.error("Λείπει το DEEPSEEK_API_KEY. Επίλεξε Τοπικό Ollama ή φόρτωσε το κλειδί στο .env.")
        return

    try:
        all_authors, all_websites, all_sections = load_filter_options(str(settings.corpus_path))
        with st.spinner("⚙️ Φόρτωση του AI pipeline..."):
            retrieval_stack = load_retrieval_stack(retrieval_cache_key(settings), settings)
            rag = load_rag_pipeline(
                pipeline_cache_key(settings),
                settings,
                *retrieval_stack,
            )
    except (CorpusError, LLMError, RuntimeError, ValueError) as exc:
        st.error(f"Το AI pipeline δεν μπόρεσε να φορτωθεί: {type(exc).__name__}: {exc}")
        render_failure_diagnostics(exc, query=None, settings=settings)
        return

    upload_catalog = get_upload_catalog(settings)
    uploaded_source_ids = tuple(source.source_id for source in upload_catalog.list_sources())
    scoped_rag = GroundedRAG(
        settings,
        retriever=ScopedRetriever(
            corpus_retriever=rag.retriever,
            uploaded_index=upload_catalog.index,
        ),
        chat_model=rag.chat_model,
        reranker=rag.reranker,
    )

    st.sidebar.header("📊 Φίλτρα")
    author_filter = st.sidebar.multiselect("Συγγραφέας", options=all_authors)
    website_filter = st.sidebar.multiselect("Ιστοσελίδα", options=all_websites)
    section_filter = st.sidebar.multiselect("Στήλη", options=all_sections)
    start_date = st.sidebar.date_input("Από Ημερομηνία", value=None)
    end_date = st.sidebar.date_input("Έως Ημερομηνία", value=None)
    max_docs = st.sidebar.slider("Μέγιστος αριθμός πηγών", 1, 20, 5)

    st.sidebar.header("📁 Προσωπικές πηγές")
    st.sidebar.caption("Οι μεταφορτώσεις είναι προσωρινές και ισχύουν μόνο για αυτή τη συνεδρία.")
    uploaded_files = st.sidebar.file_uploader(
        "Πρόσθεσε αρχείο κειμένου, Markdown ή PDF",
        type=["txt", "md", "markdown", "pdf"],
        accept_multiple_files=True,
    )
    if st.sidebar.button("Προσθήκη πηγής", key="add_upload"):
        if not uploaded_files:
            st.sidebar.error("Επίλεξε πρώτα ένα αρχείο.")
        else:
            for uploaded_file in uploaded_files:
                try:
                    created = ingest_uploaded_file(upload_catalog, uploaded_file)
                except (UnicodeDecodeError, ValueError) as exc:
                    st.sidebar.error(f"{uploaded_file.name}: Η πηγή απορρίφθηκε: {exc}")
                else:
                    message = "Η πηγή προστέθηκε." if created else "Η πηγή υπήρχε ήδη."
                    st.sidebar.success(f"{uploaded_file.name}: {message}")

    available_sources = upload_catalog.list_sources()
    if available_sources:
        st.sidebar.caption("Πηγές αυτής της συνεδρίας:")
        for source in available_sources:
            st.sidebar.write(f"• {source.display_name}")
        source_to_delete = st.sidebar.selectbox(
            "Πηγή για διαγραφή",
            options=available_sources,
            format_func=lambda source: source.display_name,
            key="source_to_delete",
        )
        if st.sidebar.button("Διαγραφή πηγής", key="delete_upload"):
            upload_catalog.delete_source(source_to_delete.source_id)
            st.sidebar.success("Η πηγή διαγράφηκε.")
            st.rerun()
    else:
        st.sidebar.caption("Δεν υπάρχουν μεταφορτωμένες πηγές.")

    scope_labels = {
        "Μόνο το corpus": RetrievalScope.CORPUS_ONLY,
        "Μόνο οι μεταφορτώσεις": RetrievalScope.UPLOADED_ONLY,
        "Corpus και μεταφορτώσεις": RetrievalScope.BOTH,
    }
    selected_scope_label = st.sidebar.radio(
        "Πηγές αναζήτησης RAG",
        options=list(scope_labels),
        key="retrieval_scope",
    )
    selected_scope = scope_labels[selected_scope_label]
    # Refresh this snapshot after upload/delete controls have run so diagnostics
    # describe the exact source set used for the upcoming question.
    uploaded_source_ids = tuple(source.source_id for source in upload_catalog.list_sources())
    render_history()
    if answer_mode == "research" and not settings.tavily_api_key:
        st.error(
            "Το Live research brief χρειάζεται TAVILY_API_KEY στο τοπικό .env. "
            "Η άμεση απάντηση παραμένει διαθέσιμη χωρίς Tavily."
        )
        return
    prompt = (
        "Δώσε ένα θέμα για live research brief..."
        if answer_mode == "research"
        else "Ρώτα κάτι σχετικό με τα δημοσιεύματα..."
    )
    user_question = st.chat_input(prompt)
    if not user_question:
        return

    with st.chat_message("user"):
        st.write(user_question)
    prior_history = [
        {"role": role, "content": message}
        for role, message, _sources in st.session_state.chat_history
    ]
    st.session_state.chat_history.append(("user", user_question, []))
    st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

    assistant_container = None
    provisional = None
    if answer_mode == "direct" and getattr(scoped_rag.chat_model, "supports_streaming", False):
        assistant_container = st.chat_message("assistant")

    try:
        if answer_mode == "research":
            with st.spinner("🌐 Αναζήτηση στο web και σύνθεση research brief..."):
                research_result = research_brief_query(
                    user_question,
                    ResearchBriefAgent(
                        scoped_rag,
                        mcp_client=TavilyMCPClient(settings),
                    ),
                    author_filter=author_filter,
                    website_filter=website_filter,
                    section_filter=section_filter,
                    start_date=start_date,
                    end_date=end_date,
                    max_docs=max_docs,
                    scope=selected_scope,
                    history=prior_history,
                )
        elif assistant_container is not None:
            with assistant_container:
                provisional = st.empty()
                provisional.write(provisional_answer_text(""))

                def render_provisional(text: str) -> None:
                    provisional.write(provisional_answer_text(text))

                result = stream_answer_query(
                    user_question,
                    scoped_rag,
                    author_filter=author_filter,
                    website_filter=website_filter,
                    section_filter=section_filter,
                    start_date=start_date,
                    end_date=end_date,
                    max_docs=max_docs,
                    scope=selected_scope,
                    history=prior_history,
                    on_update=render_provisional,
                )
                provisional.empty()
                debug_print_turn(
                    user_question,
                    result,
                    retrieval_scope=selected_scope,
                    uploaded_source_ids=uploaded_source_ids,
                    show_diagnostics=settings.show_diagnostics,
                )
        else:
            with st.spinner("💭 Ανάκτηση πηγών και δημιουργία τεκμηριωμένης απάντησης..."):
                result = answer_query(
                    user_question,
                    scoped_rag,
                    author_filter=author_filter,
                    website_filter=website_filter,
                    section_filter=section_filter,
                    start_date=start_date,
                    end_date=end_date,
                    max_docs=max_docs,
                    scope=selected_scope,
                    history=prior_history,
                )
                debug_print_turn(
                    user_question,
                    result,
                    retrieval_scope=selected_scope,
                    uploaded_source_ids=uploaded_source_ids,
                    show_diagnostics=settings.show_diagnostics,
                )
    except (
        LLMError,
        ResearchBriefError,
        TavilyMCPError,
        RuntimeError,
        ValueError,
    ) as exc:
        if answer_mode == "research":
            st.error(
                "Το live περιεχόμενο δεν μπόρεσε να επαληθευτεί, οπότε δεν δημιουργήθηκε "
                f"research brief. ({type(exc).__name__}: {exc})"
            )
        else:
            st.error(f"Η ερώτηση απέτυχε: {type(exc).__name__}: {exc}")
        debug_print_failure(
            user_question,
            exc,
            settings,
            retrieval_scope=selected_scope,
            uploaded_source_ids=uploaded_source_ids,
        )
        render_failure_diagnostics(
            exc,
            query=user_question,
            settings=settings,
            retrieval_scope=selected_scope,
            uploaded_source_ids=uploaded_source_ids,
        )
        if provisional is not None:
            provisional.empty()
        return

    if answer_mode == "research":
        answer_text = research_result.brief
        sources = research_source_payload(research_result)
        citation_validation = research_result.citation_validation
        metrics = {
            "status": "answered",
            "mode": "live_research_brief",
            "provider": research_result.generation.provider,
            "model": research_result.generation.model,
            "latency_seconds": round(research_result.generation.latency_seconds, 3),
            "total_tokens": research_result.generation.total_tokens,
            "estimated_cost_usd": research_result.generation.estimated_cost_usd,
            "mcp_calls": research_result.mcp_calls,
            "web_source_count": len(research_result.web_evidence),
        }
    else:
        answer_text = result.answer
        sources = source_payload(result)
        citation_validation = result.citation_validation
        metrics = generation_metrics(result)
    final_container = (
        assistant_container
        if answer_mode == "direct" and assistant_container is not None
        else st.chat_message("assistant")
    )
    with final_container:
        st.write(answer_text)
        if answer_mode == "research":
            st.caption(
                "Live web evidence: εξωτερικά δεδομένα μέσω Tavily MCP. "
                "Οι χρόνοι ανάκτησης εμφανίζονται στις πηγές."
            )
        if citation_validation.coverage_warning:
            st.warning("Ορισμένες προτάσεις δεν περιέχουν άμεση παραπομπή σε πηγή.")
        if answer_mode == "direct" and result.abstention_reason:
            st.caption(f"Κατάσταση: {result.status.value} — {result.abstention_reason}")
        render_sources(sources)
        if answer_mode == "direct":
            render_diagnostics(
                result,
                retrieval_scope=selected_scope,
                uploaded_source_ids=uploaded_source_ids,
            )

    st.session_state.chat_history.append(("assistant", answer_text, sources))
    if settings.enable_logging:
        user_id = str(st.session_state.user_id)
        log_interaction(user_id, user_question, answer_text)
        log_metrics(user_id, user_question, metrics, len(answer_text.split()))


if __name__ == "__main__":
    main()
