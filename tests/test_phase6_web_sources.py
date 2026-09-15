import unittest
from datetime import datetime, timezone

from journalism_rag.web_sources import (
    MAX_WEB_EXCERPT_CHARACTERS,
    WebEvidence,
    deduplicate_web_evidence,
    normalize_web_url,
)


RETRIEVED_AT = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


class Phase6WebSourceTests(unittest.TestCase):
    def test_normalizes_equivalent_http_url_forms(self):
        self.assertEqual(
            normalize_web_url(
                "HTTPS://Example.COM:443/story/?utm_source=newsletter&b=2&a=1#comments"
            ),
            "https://example.com/story?a=1&b=2",
        )
        self.assertEqual(
            normalize_web_url("https://example.com/story/"),
            "https://example.com/story",
        )

    def test_rejects_non_web_or_credential_urls(self):
        for url in ("", "file:///tmp/article", "https:///missing-host", "https://user:pass@example.com/a"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    normalize_web_url(url)

    def test_web_evidence_is_immutable_normalized_and_timezone_aware(self):
        evidence = WebEvidence(
            title="  Latest report  ",
            url="HTTP://Example.COM:80/article/#top",
            excerpt="  Relevant article text.  ",
            retrieved_at=datetime(2026, 9, 10, 15, 0, tzinfo=timezone.utc),
            published_at=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(evidence.title, "Latest report")
        self.assertEqual(evidence.url, "http://example.com/article")
        self.assertEqual(evidence.excerpt, "Relevant article text.")
        with self.assertRaises((AttributeError, TypeError)):
            evidence.title = "changed"

    def test_rejects_empty_or_oversized_evidence(self):
        with self.assertRaises(ValueError):
            WebEvidence("", "https://example.com/a", "text", RETRIEVED_AT)
        with self.assertRaises(ValueError):
            WebEvidence("Title", "https://example.com/a", "", RETRIEVED_AT)
        with self.assertRaises(ValueError):
            WebEvidence(
                "Title",
                "https://example.com/a",
                "x" * (MAX_WEB_EXCERPT_CHARACTERS + 1),
                RETRIEVED_AT,
            )

    def test_deduplicates_by_normalized_url_and_keeps_first_result(self):
        first = WebEvidence("First", "https://example.com/a?utm_medium=x", "one", RETRIEVED_AT)
        duplicate = WebEvidence("Duplicate", "https://EXAMPLE.com/a", "two", RETRIEVED_AT)
        second = WebEvidence("Second", "https://example.com/b", "three", RETRIEVED_AT)

        self.assertEqual(deduplicate_web_evidence([first, duplicate, second]), (first, second))


if __name__ == "__main__":
    unittest.main()
