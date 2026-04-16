import unittest
from unittest.mock import patch

from feedback_store import PineconeTextStore, SQLiteFeedbackStore


class FeedbackStoreTests(unittest.TestCase):
    def test_source_prediction_content_and_review_are_linked(self):
        store = SQLiteFeedbackStore(":memory:")

        source_id = store.create_source("text", url="https://example.com/story")
        store.save_extracted_content(source_id, [{"type": "article_text", "text": "Example article body."}])
        prediction_id = store.save_prediction(
            source_id,
            {
                "model_version": "ensemble_v1",
                "label": "real",
                "confidence": 0.84,
                "status": "high_confidence",
                "needs_verification": False,
            },
            "Example article body.",
        )
        review = store.save_review_label(source_id, prediction_id, "real", notes="Verified.")

        conn = store.connect()
        source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        content_count = conn.execute("SELECT COUNT(*) FROM extracted_content").fetchone()[0]
        prediction_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        review_count = conn.execute("SELECT COUNT(*) FROM review_labels").fetchone()[0]

        self.assertEqual(source_count, 1)
        self.assertEqual(content_count, 1)
        self.assertEqual(prediction_count, 1)
        self.assertEqual(review_count, 1)
        self.assertEqual(review["source_id"], source_id)
        self.assertEqual(review["prediction_id"], prediction_id)


class PineconeTextStoreTests(unittest.TestCase):
    def test_pinecone_provider_does_not_require_openai_key(self):
        env = {
            "PINECONE_API_KEY": "pc-test",
            "PINECONE_INDEX_NAME": "truth-tell-news",
            "PINECONE_EMBEDDING_PROVIDER": "pinecone",
        }
        with patch.dict("os.environ", env, clear=True):
            store = PineconeTextStore()

        self.assertTrue(store.enabled)
        self.assertEqual(store.embedding_provider, "pinecone")

    def test_openai_provider_requires_openai_key(self):
        env = {
            "PINECONE_API_KEY": "pc-test",
            "PINECONE_INDEX_NAME": "truth-tell-news",
            "PINECONE_EMBEDDING_PROVIDER": "openai",
        }
        with patch.dict("os.environ", env, clear=True):
            store = PineconeTextStore()

        self.assertFalse(store.enabled)


if __name__ == "__main__":
    unittest.main()
