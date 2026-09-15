import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journalism_rag.chunker import chunk_documents
from journalism_rag.config import Settings
from journalism_rag.embeddings import format_document, format_query
from journalism_rag.loader import CorpusError, canonicalize_url, load_corpus


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return "token " * len(token_ids)


class Phase1FoundationTests(unittest.TestCase):
    def test_e5_embedding_prompts_are_consistent(self):
        self.assertEqual(format_document("intfloat/multilingual-e5-large-instruct", "κείμενο"), "passage: κείμενο")
        self.assertTrue(format_query("intfloat/multilingual-e5-base", "ερώτηση").startswith("Instruct: Given a journalistic question"))
        self.assertEqual(format_document("Qwen/Qwen3-Embedding-0.6B", "κείμενο"), "κείμενο")

    def test_canonicalize_url_removes_fragment_and_trailing_slash(self):
        self.assertEqual(
            canonicalize_url("HTTPS://WWW.Example.com/story/#source"),
            "https://www.example.com/story",
        )

    def test_loader_rejects_duplicate_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "articles.csv"
            row = {
                "url": "https://example.com/a",
                "title": "Title",
                "full-text": "Body",
                "author": "",
                "website": "example.com",
                "datetime": "2026-01-01",
                "section": "News",
            }
            pd.DataFrame([row, row]).to_csv(path, index=False)
            with self.assertRaisesRegex(CorpusError, "Duplicate canonical URL"):
                load_corpus(path)

    def test_chunk_ids_are_stable_and_token_bounded(self):
        document = Document(
            page_content="one two three four five six seven eight nine ten",
            metadata={"article_id": "abc", "title": "Headline"},
        )
        first = chunk_documents([document], FakeTokenizer(), chunk_size_tokens=4, chunk_overlap_tokens=1)
        second = chunk_documents([document], FakeTokenizer(), chunk_size_tokens=4, chunk_overlap_tokens=1)
        self.assertEqual([x.metadata["chunk_id"] for x in first], [x.metadata["chunk_id"] for x in second])
        self.assertTrue(all(x.metadata["token_count"] <= 4 for x in first))

    def test_settings_reject_invalid_overlap(self):
        with self.assertRaises(ValueError):
            Settings(chunk_size_tokens=4, chunk_overlap_tokens=4).validate()

    def test_default_collection_uses_prompt_aware_e5_name(self):
        self.assertEqual(
            Settings().collection_name,
            "journalism_rag__e5-large-instruct",
        )

    def test_loader_rejects_invalid_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "articles.csv"
            pd.DataFrame(
                [{
                    "url": "https://example.com/a",
                    "title": "Title",
                    "full-text": "Body",
                    "author": "",
                    "website": "example.com",
                    "datetime": "not-a-date",
                    "section": "News",
                }]
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(CorpusError, "invalid publication date"):
                load_corpus(path)


if __name__ == "__main__":
    unittest.main()
