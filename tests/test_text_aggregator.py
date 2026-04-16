import unittest

from text_aggregator import aggregate_text, prepare_content_pieces


class TextAggregatorTests(unittest.TestCase):
    def test_article_text_is_preserved_as_primary_input(self):
        text = aggregate_text([
            {"type": "frame_ocr", "text": "BREAKING"},
            {"type": "article_text", "text": "The article body contains the actual claim."},
        ])

        self.assertTrue(text.startswith("The article body contains the actual claim."))
        self.assertIn("BREAKING", text)

    def test_transcript_outranks_ocr_for_video(self):
        text = aggregate_text([
            {"type": "frame_ocr", "text": "Breaking News"},
            {"type": "transcript", "text": "The reporter described the claim in full."},
        ])

        self.assertTrue(text.startswith("The reporter described the claim in full."))

    def test_duplicate_ocr_fragments_are_removed(self):
        pieces = prepare_content_pieces([
            {"type": "frame_ocr", "text": "Breaking News"},
            {"type": "frame_ocr", "text": "Breaking News"},
        ])

        self.assertEqual(len(pieces), 1)

    def test_generic_captions_are_ignored(self):
        text = aggregate_text([
            {"type": "caption", "text": "person standing"},
            {"type": "transcript", "text": "The transcript has useful context."},
        ])

        self.assertNotIn("person standing", text)
        self.assertIn("The transcript has useful context.", text)


if __name__ == "__main__":
    unittest.main()
