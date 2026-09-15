import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from streamlit.testing.v1 import AppTest

import app
from journalism_rag.citations import CitationValidation
from journalism_rag.config import Settings
from journalism_rag.llm import GenerationResult
from journalism_rag.pipeline import AnswerStatus, GroundedAnswer, SourceCitation
from journalism_rag.research_brief import ResearchBriefRun
from journalism_rag.user_sources import (
    InMemorySourceStore,
    IndexedSourceCatalog,
    RetrievalScope,
    UploadedSourceIndex,
)
from journalism_rag.web_sources import WebEvidence


def grounded_result():
    return GroundedAnswer(
        status=AnswerStatus.ANSWERED,
        answer="Αναφέρονται 57 νεκροί [S1].",
        sources=(
            SourceCitation(
                source_id="S1",
                article_id="a1",
                url="https://example.com/a",
                title="Τέμπη",
                rank=1,
                score=0.91234,
                chunk_id="a1:c1:0",
                text="Το άρθρο αναφέρει 57 νεκρούς.",
                author="Reporter",
                outlet="example.com",
                section="News",
                published_at=1.0,
            ),
        ),
        used_source_ids=("S1",),
        citation_validation=CitationValidation(
            valid=True,
            abstained=False,
            cited_source_ids=("S1",),
            invalid_source_ids=(),
            uncited_claims=(),
        ),
        abstention_reason=None,
        generation=GenerationResult(
            text="Αναφέρονται 57 νεκροί [S1].",
            provider="fake",
            model="fake-model",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18,
            latency_seconds=0.01,
        ),
    )


def rejected_result():
    result = grounded_result()
    return GroundedAnswer(
        status=AnswerStatus.ABSTAINED,
        answer="Δεν υπάρχουν επαρκή στοιχεία στο διαθέσιμο corpus.",
        sources=result.sources,
        used_source_ids=(),
        citation_validation=CitationValidation(
            valid=False,
            abstained=False,
            cited_source_ids=("S1",),
            invalid_source_ids=("S9",),
            uncited_claims=("Η σύγκρουση έγινε τη νύχτα.",),
            coverage_warning=True,
        ),
        abstention_reason="citation_validation_failed",
        generation=result.generation,
        retrieval_query="σύγκρουση τρένων Τέμπη",
    )


def research_result():
    generation = grounded_result().generation
    local_source = SourceCitation(
        source_id="C1",
        article_id="a1",
        url="https://archive.example/tempi",
        title="Archive report",
        rank=1,
        score=0.91,
        chunk_id="a1:c1:0",
        text="Archive evidence.",
        author="Reporter",
        outlet="archive.example",
        section="News",
        published_at=1677628800.0,
        source_type="corpus",
    )
    retrieved_at = datetime(2026, 9, 10, 9, 30, tzinfo=timezone.utc)
    published_at = datetime(2026, 9, 9, 8, 15, tzinfo=timezone.utc)
    web_evidence = WebEvidence(
        title="Current report",
        url="https://news.example/tempi",
        excerpt="Current web evidence.",
        retrieved_at=retrieved_at,
        published_at=published_at,
    )
    web_source = SourceCitation(
        source_id="W1",
        article_id=None,
        url=web_evidence.url,
        title=web_evidence.title,
        rank=1,
        score=0.0,
        chunk_id=None,
        text=web_evidence.excerpt,
        author=None,
        outlet=None,
        section=None,
        published_at=published_at.timestamp(),
        source_type="web",
    )
    validation = CitationValidation(
        valid=True,
        abstained=False,
        cited_source_ids=("C1", "W1"),
        invalid_source_ids=(),
        uncited_claims=(),
    )
    return ResearchBriefRun(
        generation=generation,
        local_sources=(local_source,),
        web_evidence=(web_evidence,),
        tool_exchanges=(),
        mcp_calls=1,
        retrieval_query="Tempi latest",
        brief="A cited brief [C1] [W1].",
        sources=(local_source, web_source),
        citation_validation=validation,
    )


class FakeRAG:
    def __init__(self):
        self.calls = []

    def answer(self, query, *, limit=5, where=None, scope=RetrievalScope.CORPUS_ONLY):
        self.calls.append({"query": query, "limit": limit, "where": where, "scope": scope})
        return grounded_result()

    def stream_answer(self, query, *, limit=5, where=None, scope=RetrievalScope.CORPUS_ONLY, on_update=None):
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "where": where,
                "scope": scope,
                "on_update": on_update,
            }
        )
        if on_update is not None:
            on_update("Προσωρινή απάντηση")
        return grounded_result()


class FakeResearchAgent:
    def __init__(self):
        self.calls = []

    def run(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return research_result()


class FakeUploadedFile:
    def __init__(self, name, payload):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload


class StreamlitIntegrationTests(unittest.TestCase):
    def test_provisional_answer_text_uses_placeholder_then_stream_cursor(self):
        self.assertEqual(app.provisional_answer_text(""), "Ανάκτηση σχετικών πηγών…")
        self.assertEqual(
            app.provisional_answer_text("Η απάντηση"),
            "Η απάντηση▌",
        )

    def test_date_filter_boundaries_are_start_and_inclusive_end_of_day(self):
        where = app.build_where_filter(
            author_filter=[],
            website_filter=[],
            section_filter=[],
            start_date=date(2023, 3, 1),
            end_date=date(2023, 3, 2),
        )

        self.assertEqual(
            where,
            {
                "$and": [
                    {
                        "datetime": {
                            "$gte": app.date_to_timestamp(date(2023, 3, 1))
                        }
                    },
                    {
                        "datetime": {
                            "$lte": app.date_to_timestamp(
                                date(2023, 3, 2), end_of_day=True
                            )
                        }
                    },
                ]
            },
        )
        self.assertEqual(
            app.date_to_timestamp(date(2023, 3, 1)),
            datetime(2023, 3, 1, tzinfo=timezone.utc).timestamp(),
        )
        self.assertEqual(
            app.date_to_timestamp(date(2023, 3, 2), end_of_day=True),
            datetime(2023, 3, 2, 23, 59, 59, 999999, tzinfo=timezone.utc).timestamp(),
        )

    def test_app_adapter_passes_filters_and_returns_structured_result(self):
        rag = FakeRAG()
        result = app.answer_query(
            "Πόσοι νεκροί αναφέρονται;",
            rag,
            author_filter=["Reporter"],
            website_filter=["example.com"],
            section_filter=["News"],
            start_date=date(2023, 3, 1),
            end_date=date(2023, 3, 2),
            max_docs=7,
        )
        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertEqual(rag.calls[0]["limit"], 7)
        clauses = rag.calls[0]["where"]["$and"]
        self.assertIn({"author": {"$in": ["Reporter"]}}, clauses)
        self.assertIn({"website": {"$in": ["example.com"]}}, clauses)
        self.assertIn({"section": {"$in": ["News"]}}, clauses)
        self.assertEqual(len(clauses), 5)

    def test_app_adapter_forwards_non_default_retrieval_scope(self):
        rag = FakeRAG()
        app.answer_query(
            "What is in my upload?",
            rag,
            author_filter=[],
            website_filter=[],
            section_filter=[],
            start_date=None,
            end_date=None,
            max_docs=3,
            scope=RetrievalScope.UPLOADED_ONLY,
        )
        self.assertEqual(rag.calls[0]["scope"], RetrievalScope.UPLOADED_ONLY)

    def test_stream_app_adapter_forwards_filters_and_update_callback(self):
        rag = FakeRAG()
        updates = []
        result = app.stream_answer_query(
            "Τι συνέβη;",
            rag,
            author_filter=["Reporter"],
            website_filter=[],
            section_filter=[],
            start_date=None,
            end_date=None,
            max_docs=4,
            on_update=updates.append,
        )
        self.assertEqual(result.status, AnswerStatus.ANSWERED)
        self.assertEqual(updates, ["Προσωρινή απάντηση"])
        self.assertEqual(rag.calls[0]["limit"], 4)
        self.assertIsNotNone(rag.calls[0]["on_update"])
        self.assertEqual(rag.calls[0]["where"], {"author": {"$in": ["Reporter"]}})

    def test_research_adapter_reuses_filters_scope_and_history(self):
        agent = FakeResearchAgent()
        history = [{"role": "user", "content": "Earlier question"}]

        result = app.research_brief_query(
            "Τι νέο υπάρχει;",
            agent,
            author_filter=["Reporter"],
            website_filter=[],
            section_filter=["News"],
            start_date=None,
            end_date=None,
            max_docs=4,
            scope=RetrievalScope.BOTH,
            history=history,
        )

        self.assertEqual(result.brief, "A cited brief [C1] [W1].")
        self.assertEqual(agent.calls[0]["limit"], 4)
        self.assertEqual(agent.calls[0]["scope"], RetrievalScope.BOTH)
        self.assertEqual(agent.calls[0]["history"], history)
        self.assertEqual(
            agent.calls[0]["where"],
            {
                "$and": [
                    {"section": {"$in": ["News"]}},
                    {"author": {"$in": ["Reporter"]}},
                ]
            },
        )

    def test_ingest_uploaded_file_adds_multiple_files_to_one_catalog(self):
        catalog = IndexedSourceCatalog(InMemorySourceStore(), UploadedSourceIndex(settings=Settings()))

        created_first = app.ingest_uploaded_file(
            catalog,
            FakeUploadedFile("first.md", b"The first uploaded report."),
        )
        created_second = app.ingest_uploaded_file(
            catalog,
            FakeUploadedFile("second.txt", b"The second uploaded report."),
        )

        self.assertTrue(created_first)
        self.assertTrue(created_second)
        self.assertEqual(
            [source.display_name for source in catalog.list_sources()],
            ["first.md", "second.txt"],
        )

    def test_source_payload_preserves_labels_and_usage(self):
        sources = app.source_payload(grounded_result())
        self.assertEqual(sources[0]["source_id"], "S1")
        self.assertTrue(sources[0]["used"])
        self.assertEqual(sources[0]["score"], 0.9123)

    def test_research_source_payload_keeps_clickable_web_metadata(self):
        sources = app.research_source_payload(research_result())

        web_source = sources[1]
        self.assertEqual(web_source["source_id"], "W1")
        self.assertEqual(web_source["source_type"], "web")
        self.assertEqual(web_source["url"], "https://news.example/tempi")
        self.assertEqual(web_source["published_at"], "2026-09-09T08:15+00:00")
        self.assertEqual(web_source["retrieved_at"], "2026-09-10T09:30+00:00")
        self.assertTrue(web_source["used"])

    def test_citation_diagnostics_exposes_validation_details_without_answer(self):
        diagnostics = app.citation_diagnostics(rejected_result())
        self.assertEqual(diagnostics["invalid_source_ids"], ["S9"])
        self.assertEqual(diagnostics["uncited_claims"], ["Η σύγκρουση έγινε τη νύχτα."])
        self.assertTrue(diagnostics["coverage_warning"])
        self.assertEqual(diagnostics["retrieval_query"], "σύγκρουση τρένων Τέμπη")
        self.assertNotIn("answer", diagnostics)
        self.assertEqual(diagnostics["generation"]["finish_reason"], "stop")
        self.assertEqual(diagnostics["generation"]["total_tokens"], 18)
        self.assertEqual(diagnostics["generation"]["output_characters"], len(rejected_result().generation.text))

    def test_citation_diagnostics_reports_scope_and_session_sources(self):
        diagnostics = app.citation_diagnostics(
            rejected_result(),
            retrieval_scope=RetrievalScope.UPLOADED_ONLY,
            uploaded_source_ids=("upload-test",),
        )
        self.assertEqual(diagnostics["status"], "abstained")
        self.assertEqual(diagnostics["abstention_reason"], "citation_validation_failed")
        self.assertEqual(diagnostics["retrieved_source_count"], 1)
        self.assertEqual(diagnostics["retrieved_source_ids"], ["S1"])
        self.assertEqual(diagnostics["retrieval_scope"], "uploaded_only")
        self.assertEqual(diagnostics["uploaded_source_count"], 1)
        self.assertEqual(diagnostics["uploaded_source_ids"], ["upload-test"])

    def test_failure_diagnostics_are_safe_and_include_active_model_settings(self):
        settings = Settings(
            llm_provider="deepseek",
            llm_model="deepseek-flash",
            llm_max_tokens=2048,
            deepseek_thinking="disabled",
            deepseek_api_key="secret-do-not-display",
        )
        diagnostics = app.failure_diagnostics(
            RuntimeError("finish_reason=length"),
            query="Ποια γεγονότα;",
            settings=settings,
        )
        self.assertEqual(diagnostics["error_type"], "RuntimeError")
        self.assertEqual(diagnostics["query"], "Ποια γεγονότα;")
        self.assertEqual(diagnostics["model"], "deepseek-flash")
        self.assertEqual(diagnostics["max_tokens"], 2048)
        self.assertEqual(diagnostics["deepseek_thinking"], "disabled")
        self.assertNotIn("secret-do-not-display", str(diagnostics))

    def test_pipeline_cache_key_changes_when_model_settings_change(self):
        base = Settings()
        changed_model = Settings(llm_model="deepseek-v4-pro")
        changed_tokens = Settings(llm_max_tokens=2048)
        changed_reasoning = Settings(ollama_reasoning_effort="low")
        changed_embedding_batch = Settings(embedding_batch_size=16)
        self.assertNotEqual(app.pipeline_cache_key(base), app.pipeline_cache_key(changed_model))
        self.assertNotEqual(app.pipeline_cache_key(base), app.pipeline_cache_key(changed_tokens))
        self.assertNotEqual(app.pipeline_cache_key(base), app.pipeline_cache_key(changed_reasoning))
        self.assertNotEqual(app.pipeline_cache_key(base), app.pipeline_cache_key(changed_embedding_batch))

    def test_pipeline_cache_key_rotates_for_credentials_without_exposing_them(self):
        first = Settings(deepseek_api_key="first-secret")
        second = Settings(deepseek_api_key="second-secret")

        first_key = app.pipeline_cache_key(first)
        second_key = app.pipeline_cache_key(second)

        self.assertNotEqual(first_key, second_key)
        self.assertNotIn("first-secret", repr(first_key))
        self.assertNotIn("second-secret", repr(second_key))

    def test_retrieval_stack_cache_reuses_heavy_objects(self):
        settings = Settings(reranker_enabled=True)
        fake_retriever = object()
        fake_reranker = object()
        app.load_retrieval_stack.clear()
        with (
            patch.object(app, "HybridRetriever", return_value=fake_retriever) as retriever,
            patch.object(app, "create_reranker", return_value=fake_reranker) as reranker,
        ):
            first = app.load_retrieval_stack(app.retrieval_cache_key(settings), settings)
            second = app.load_retrieval_stack(app.retrieval_cache_key(settings), settings)

        self.assertIs(first, second)
        self.assertIs(first[0], fake_retriever)
        self.assertIs(first[1], fake_reranker)
        retriever.assert_called_once_with(settings)
        reranker.assert_called_once_with(
            settings.reranker_model,
            device=settings.reranker_device,
            max_length=settings.reranker_max_length,
            batch_size=settings.reranker_batch_size,
            local_files_only=True,
        )
        app.load_retrieval_stack.clear()

    def test_runtime_provider_switch_uses_provider_specific_models_and_cache_keys(self):
        hosted = Settings(deepseek_api_key="secret")
        local = app.settings_for_provider(hosted, "ollama")
        switched_back = app.settings_for_provider(local, "deepseek")

        self.assertEqual(local.llm_provider, "ollama")
        self.assertEqual(local.llm_model, "journalism-rag-qwen3.5:9b")
        self.assertEqual(local.llm_base_url, "http://localhost:11434/v1")
        self.assertEqual(local.llm_max_tokens, 8192)
        self.assertEqual(switched_back.llm_provider, "deepseek")
        self.assertEqual(switched_back.llm_model, "deepseek-flash")
        self.assertEqual(switched_back.llm_base_url, "https://api.deepseek.com")
        self.assertEqual(switched_back.llm_max_tokens, 8192)
        self.assertNotEqual(app.pipeline_cache_key(hosted), app.pipeline_cache_key(local))

    def test_runtime_provider_switch_rejects_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "provider"):
            app.settings_for_provider(Settings(), "unsupported")

    def test_hosted_provider_notice_discloses_followup_conversation_data(self):
        hosted_notice = app.provider_privacy_notice(
            Settings(llm_provider="deepseek", deepseek_api_key="secret")
        )
        local_notice = app.provider_privacy_notice(
            Settings(llm_provider="ollama", llm_base_url="http://localhost:11434/v1")
        )

        self.assertIn("hosted DeepSeek", hosted_notice)
        self.assertIn("πρόσφατα μηνύματα της συνομιλίας", hosted_notice)
        self.assertIn("ερωτήσεις συνέχειας", hosted_notice)
        self.assertNotIn("πρόσφατα μηνύματα", local_notice)

    def test_debug_print_turn_writes_query_and_summary_when_enabled(self):
        with patch.object(app, "SHOW_DIAGNOSTICS", True), patch("builtins.print") as mocked_print:
            app.debug_print_turn("Και μετά;", rejected_result())

        output = "\n".join(call.args[0] for call in mocked_print.call_args_list)
        self.assertIn("original query: Και μετά;", output)
        self.assertIn("rewritten retrieval query: σύγκρουση τρένων Τέμπη", output)
        self.assertIn("raw model summary (unvalidated): Αναφέρονται 57 νεκροί [S1].", output)
        self.assertIn("status: abstained", output)

    def test_debug_print_turn_uses_validated_settings_flag(self):
        with patch.object(app, "SHOW_DIAGNOSTICS", False), patch("builtins.print") as mocked_print:
            app.debug_print_turn(
                "Question",
                grounded_result(),
                show_diagnostics=Settings(show_diagnostics=True).show_diagnostics,
            )

        output = "\n".join(call.args[0] for call in mocked_print.call_args_list)
        self.assertIn("raw model summary (unvalidated):", output)

    def test_debug_print_turn_is_silent_by_default(self):
        with patch.object(app, "SHOW_DIAGNOSTICS", False), patch("builtins.print") as mocked_print:
            app.debug_print_turn("Question", grounded_result())
        mocked_print.assert_not_called()

    def test_debug_print_turn_reports_failures_by_default_without_raw_output(self):
        with patch.object(app, "SHOW_DIAGNOSTICS", False), patch("builtins.print") as mocked_print:
            app.debug_print_turn("Question", rejected_result())

        output = "\n".join(call.args[0] for call in mocked_print.call_args_list)
        self.assertIn("status: abstained", output)
        self.assertIn("abstention reason: citation_validation_failed", output)
        self.assertIn("raw model summary: <hidden", output)
        self.assertNotIn("Αναφέρονται 57 νεκροί [S1].", output)

    def test_streamlit_landing_page_starts_without_loading_a_model(self):
        application = AppTest.from_file("app.py").run(timeout=20)
        self.assertEqual(len(application.exception), 0)
        self.assertEqual(
            application.title[0].value,
            "Καλώς ήρθες στο Greek Journalism RAG Chatbot! 💬",
        )
        self.assertEqual(len(application.text_input), 1)
        self.assertEqual(len(application.button), 1)


if __name__ == "__main__":
    unittest.main()
