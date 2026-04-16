import unittest

import numpy as np

from fake_news_predictor import FakeNewsPredictor


class DummyPipeline:
    def transform(self, texts):
        return np.asarray([[len(texts[0])]])


class DummyModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, features):
        return np.asarray([self.probabilities])


class FakeNewsPredictorTests(unittest.TestCase):
    def make_predictor(self, probabilities):
        return FakeNewsPredictor(model=DummyModel(probabilities), pipeline=DummyPipeline())

    def test_empty_text_returns_uncertain(self):
        result = self.make_predictor([0.1, 0.9]).predict("")

        self.assertEqual(result["label"], "uncertain")
        self.assertEqual(result["confidence"], 0.0)

    def test_class_one_returns_real(self):
        result = self.make_predictor([0.1, 0.9]).predict("verified report")

        self.assertEqual(result["raw_label"], 1)
        self.assertEqual(result["label"], "real")
        self.assertEqual(result["status"], "high_confidence")

    def test_class_zero_returns_fake(self):
        result = self.make_predictor([0.91, 0.09]).predict("fabricated report")

        self.assertEqual(result["raw_label"], 0)
        self.assertEqual(result["label"], "fake")
        self.assertEqual(result["status"], "high_confidence")

    def test_low_probability_returns_uncertain(self):
        result = self.make_predictor([0.52, 0.48]).predict("ambiguous report")

        self.assertEqual(result["label"], "uncertain")
        self.assertEqual(result["confidence"], 0.52)

    def test_confidence_uses_max_probability(self):
        result = self.make_predictor([0.61, 0.39]).predict("likely fake report")

        self.assertEqual(result["confidence"], 0.61)
        self.assertEqual(result["status"], "likely_fake")


if __name__ == "__main__":
    unittest.main()
