import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_chroma import Chroma

from journalism_rag.config import Settings
from journalism_rag.indexer import (
    IndexContractError,
    _collection_contract,
    _reset_collection,
    _validate_collection,
)
from journalism_rag.retrieval import HybridRetriever


class FakeEmbeddings:
    """Deterministic embeddings for the collection-scope regression test."""

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class FakeCollection:
    def __init__(self, metadata, ids):
        self._collection = SimpleNamespace(metadata=metadata)
        self._ids = ids

    def get(self):
        return {"ids": list(self._ids)}


class Phase9RepairTests(unittest.TestCase):
    def test_local_demo_environment_resolves_the_intended_profile(self):
        with patch.dict(
            "os.environ",
            {
                "RAG_LLM_PROVIDER": "ollama",
                "RAG_LLM_MODEL": "journalism-rag-qwen3.5:9b",
                "RAG_LLM_BASE_URL": "http://localhost:11434/v1",
                "RAG_LLM_MAX_TOKENS": "8192",
                "RAG_EMBEDDING_DEVICE": "mps",
                "RAG_RERANKER_ENABLED": "1",
                "RAG_RERANKER_MODEL": "Qwen/Qwen3-Reranker-0.6B",
                "RAG_RERANKER_DEVICE": "mps",
                "RAG_RERANKER_CANDIDATE_LIMIT": "20",
                "RAG_RERANKER_EVIDENCE_LIMIT": "5",
                "RAG_RERANKER_BATCH_SIZE": "4",
                "RAG_RERANKER_MAX_LENGTH": "256",
                "RAG_OLLAMA_REASONING_EFFORT": "none",
                "RAG_ENABLE_LOGGING": "0",
                "RAG_SHOW_DIAGNOSTICS": "0",
                "DEEPSEEK_API_KEY": "preserved-deepseek-secret",
                "TAVILY_API_KEY": "preserved-tavily-secret",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.llm_provider, "ollama")
        self.assertEqual(settings.llm_model, "journalism-rag-qwen3.5:9b")
        self.assertEqual(settings.llm_base_url, "http://localhost:11434/v1")
        self.assertEqual(settings.llm_max_tokens, 8192)
        self.assertEqual(settings.embedding_device, "mps")
        self.assertTrue(settings.reranker_enabled)
        self.assertEqual(settings.reranker_device, "mps")
        self.assertEqual(settings.reranker_candidate_limit, 20)
        self.assertEqual(settings.reranker_evidence_limit, 5)
        self.assertEqual(settings.ollama_reasoning_effort, "none")
        self.assertFalse(settings.enable_logging)
        self.assertFalse(settings.show_diagnostics)
        self.assertEqual(settings.deepseek_api_key, "preserved-deepseek-secret")
        self.assertEqual(settings.tavily_api_key, "preserved-tavily-secret")

    def test_env_example_has_empty_secret_placeholders(self):
        example = Path(__file__).parents[1] / ".env.example"
        values = {}
        for line in example.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value

        self.assertEqual(values["RAG_LLM_PROVIDER"], "ollama")
        self.assertEqual(values["RAG_LLM_MODEL"], "journalism-rag-qwen3.5:9b")
        self.assertEqual(values["RAG_EMBEDDING_DEVICE"], "mps")
        self.assertEqual(values["RAG_RERANKER_ENABLED"], "1")
        self.assertEqual(values["RAG_RERANKER_DEVICE"], "mps")
        self.assertEqual(values["DEEPSEEK_API_KEY"], "")
        self.assertEqual(values["TAVILY_API_KEY"], "")

    def test_reset_deletes_only_selected_chroma_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            embedding = FakeEmbeddings()
            Chroma(
                collection_name="selected",
                persist_directory=directory,
                embedding_function=embedding,
            ).add_texts(["selected document"], ids=["selected-1"])
            Chroma(
                collection_name="comparison",
                persist_directory=directory,
                embedding_function=embedding,
            ).add_texts(["comparison document"], ids=["comparison-1"])

            _reset_collection(
                Settings(
                    chroma_path=Path(directory),
                    collection_name="selected",
                )
            )

            remaining = Chroma(
                collection_name="comparison",
                persist_directory=directory,
                embedding_function=embedding,
            )
            self.assertEqual(remaining.get()["ids"], ["comparison-1"])
            selected = Chroma(
                collection_name="selected",
                persist_directory=directory,
                embedding_function=embedding,
            )
            self.assertEqual(selected.get()["ids"], [])

    def test_resume_accepts_missing_chunks_but_returns_existing_ids(self):
        settings = Settings()
        collection = FakeCollection(_collection_contract(settings), ["article:old:0"])

        existing = _validate_collection(
            collection,
            settings,
            expected_chunk_ids={"article:old:0", "article:new:0"},
        )

        self.assertEqual(existing, {"article:old:0"})

    def test_collection_contract_persists_in_chroma_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                chroma_path=Path(directory),
                collection_name="contract",
            )
            Chroma(
                collection_name=settings.collection_name,
                persist_directory=directory,
                embedding_function=FakeEmbeddings(),
                collection_metadata=_collection_contract(settings),
            )

            reopened = Chroma(
                collection_name=settings.collection_name,
                persist_directory=directory,
                embedding_function=FakeEmbeddings(),
            )

            self.assertEqual(_validate_collection(reopened, settings), set())

    def test_resume_rejects_stale_chunks_from_changed_or_removed_corpus(self):
        settings = Settings()
        collection = FakeCollection(_collection_contract(settings), ["article:old:0"])

        with self.assertRaisesRegex(IndexContractError, "stale"):
            _validate_collection(
                collection,
                settings,
                expected_chunk_ids={"article:new:0"},
            )

    def test_resume_rejects_incompatible_embedding_or_chunk_contract(self):
        original = Settings()
        changed = Settings(chunk_size_tokens=256, chunk_overlap_tokens=32)
        collection = FakeCollection(_collection_contract(original), [])

        with self.assertRaisesRegex(IndexContractError, "no compatible"):
            _validate_collection(collection, changed)

    def test_missing_or_empty_collection_blocks_hybrid_lexical_fallback(self):
        settings = Settings()
        for metadata in (_collection_contract(settings), None):
            with self.subTest(metadata=metadata):
                empty_collection = FakeCollection(metadata, [])
                with (
                    patch("journalism_rag.retrieval.Chroma", return_value=empty_collection),
                    patch(
                        "journalism_rag.retrieval._embedding_function",
                        return_value=FakeEmbeddings(),
                    ),
                    patch("journalism_rag.retrieval.BM25Retriever") as lexical,
                ):
                    with self.assertRaisesRegex(IndexContractError, "missing or empty"):
                        HybridRetriever(settings)
                    lexical.assert_not_called()


if __name__ == "__main__":
    unittest.main()
