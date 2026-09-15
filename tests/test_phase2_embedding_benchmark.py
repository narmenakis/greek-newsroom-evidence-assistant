import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langchain_core.documents import Document

from journalism_rag.config import Settings
from journalism_rag.evaluation import embedding_benchmark


class FakeEmbedder:
    def __init__(self, *, model_name, model_kwargs, encode_kwargs):
        self.device = model_kwargs["device"]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class Phase2EmbeddingBenchmarkTests(unittest.TestCase):
    def test_benchmark_passes_explicit_device_to_embedder(self):
        document = Document(
            page_content="Το άρθρο.",
            metadata={"url": "https://example.test/article"},
        )
        question = {
            "id": "case-1",
            "question": "Τι αναφέρει;",
            "answerable": True,
            "relevant_urls": ["https://example.test/article"],
        }
        with (
            patch.object(embedding_benchmark, "load_corpus", return_value=[document]),
            patch.object(embedding_benchmark, "chunk_documents", return_value=[document]),
            patch.object(
                embedding_benchmark,
                "HuggingFaceEmbeddings",
                return_value=FakeEmbedder(
                    model_name="Qwen/Qwen3-Embedding-0.6B",
                    model_kwargs={"device": "mps"},
                    encode_kwargs={},
                ),
            ),
            patch.object(embedding_benchmark.AutoTokenizer, "from_pretrained", return_value=object()),
        ):
            result = embedding_benchmark.benchmark_model(
                "Qwen/Qwen3-Embedding-0.6B",
                Settings(),
                [question],
                2,
                device="mps",
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["device"], "mps")


if __name__ == "__main__":
    unittest.main()
