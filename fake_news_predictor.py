import logging
import os
import sys

import joblib
import numpy as np

# Keep the custom estimator import available for joblib unpickling.
from models import EnsembleModel  # noqa: F401


logger = logging.getLogger(__name__)


class FakeNewsPredictor:
    def __init__(
        self,
        model_path=None,
        pipeline_path=None,
        model_version="ensemble_v1",
        uncertain_threshold=0.55,
        high_confidence_threshold=0.80,
        model=None,
        pipeline=None,
    ):
        self.model_path = model_path or os.path.join("saved_models", "ensemble_model.pkl")
        self.pipeline_path = pipeline_path or os.path.join("saved_models", "pipeline.pkl")
        self.model_version = model_version
        self.uncertain_threshold = uncertain_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.model = model
        self.pipeline = pipeline

        if self.model is None or self.pipeline is None:
            self._load()

    def _load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.pipeline_path):
            raise FileNotFoundError("Model or pipeline not found. Please train and save them first.")
        # Older pickles were saved when EnsembleModel was defined in __main__.
        # Expose it there before loading so joblib can resolve the class.
        main_module = sys.modules.get("__main__")
        if main_module is not None and not hasattr(main_module, "EnsembleModel"):
            setattr(main_module, "EnsembleModel", EnsembleModel)
        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)

    def predict(self, text, source_type=None):
        text = (text or "").strip()
        if not text:
            return self._result(label="uncertain", raw_label=None, confidence=0.0, status="uncertain")

        try:
            features = self.pipeline.transform([text])
            raw_label, confidence = self._predict_label_and_confidence(features)
            base_label = self._label_from_raw(raw_label)
            label, status = self._label_with_status(base_label, confidence)
            return self._result(label=label, raw_label=raw_label, confidence=confidence, status=status)
        except Exception as exc:
            logger.error("Error predicting fake-news label: %s", exc)
            return self._result(label="uncertain", raw_label=None, confidence=0.0, status="uncertain")

    def _predict_label_and_confidence(self, features):
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(features))[0]
            confidence = float(np.max(probabilities))
            raw_label = self._raw_label_from_probabilities(probabilities)
            return raw_label, confidence

        prediction = np.asarray(self.model.predict(features))[0]
        return self._coerce_raw_label(prediction), 1.0

    def _raw_label_from_probabilities(self, probabilities):
        classes = getattr(self.model, "classes_", None)
        best_index = int(np.argmax(probabilities))
        if classes is not None and len(classes) > best_index:
            return self._coerce_raw_label(classes[best_index])
        return best_index

    def _coerce_raw_label(self, raw_label):
        try:
            return int(raw_label)
        except (TypeError, ValueError):
            normalized = str(raw_label).strip().lower()
            if normalized in {"1", "true", "real"}:
                return 1
            return 0

    def _label_from_raw(self, raw_label):
        return "real" if raw_label == 1 else "fake"

    def _label_with_status(self, base_label, confidence):
        if confidence < self.uncertain_threshold:
            return "uncertain", "uncertain"
        if confidence < self.high_confidence_threshold:
            return base_label, f"likely_{base_label}"
        return base_label, "high_confidence"

    def _result(self, label, raw_label, confidence, status):
        return {
            "label": label,
            "raw_label": raw_label,
            "confidence": float(confidence),
            "status": status,
            "model_version": self.model_version,
            "prediction": label,
        }
