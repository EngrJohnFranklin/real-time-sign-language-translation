"""
XGBoost-based sign language classifier using MediaPipe hand landmarks.

Drop-in replacement for geometric rule-based sign detection in sign_detector.py.
Expects a trained model bundle saved by scripts/train_xgboost_model.py.

Uses landmark_normalizer for consistent preprocessing across training and inference.
"""

import os
import pickle
import logging
import pathlib
import numpy as np
from typing import Optional, Tuple, List

from utils.landmark_normalizer import normalize_landmarks, normalize_dual_hand_features

logger = logging.getLogger(__name__)

# Default model location relative to this file (src/models/ → project root)
_DEFAULT_MODEL_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "models" / "sign_model.pkl"

# Feature dimensions
FEATURES_PER_HAND = 63
TOTAL_FEATURES = 126  # 2 hands × 63 features each


class XGBoostSignClassifier:
    """
    Lightweight XGBoost classifier for sign language recognition.

    Accepts 21 MediaPipe hand landmarks per frame and returns the predicted
    sign label with a probability-based confidence score.
    Inference runs in 1-5 ms on CPU.
    """

    def __init__(self, model_path: str = str(_DEFAULT_MODEL_PATH)):
        """
        Load a trained XGBoost model bundle from disk.

        Args:
            model_path: Path to the .pkl bundle produced by train_xgboost_model.py.

        Raises:
            FileNotFoundError: Model file does not exist.
            RuntimeError: Bundle is corrupted or incompatible.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found: {model_path}")

        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.label_encoder = bundle["label_encoder"]
            self.feature_count = bundle.get("feature_count", 126)
            self.classes: List[str] = bundle.get("classes", list(self.label_encoder.classes_))
            logger.info(
                f"XGBoostSignClassifier loaded: {len(self.classes)} classes from {model_path}"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load XGBoost model bundle: {exc}") from exc

    @staticmethod
    def _normalize_single_hand(
        landmarks: Optional[List[Tuple[float, float, float]]]
    ) -> Optional[np.ndarray]:
        """
        Normalize a single hand's 21 landmarks to a 63-element feature vector.

        Delegates to landmark_normalizer.normalize_landmarks for consistent preprocessing.

        Args:
            landmarks: List of 21 (x, y, z) tuples from MediaPipe, or None.

        Returns:
            Normalized 63-element numpy array, or None when input is invalid or missing.
        """
        return normalize_landmarks(landmarks)

    @staticmethod
    def extract_features(
        left_landmarks: Optional[List[Tuple[float, float, float]]] = None,
        right_landmarks: Optional[List[Tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """
        Convert up to 2 hands (21 landmarks each) to a fixed 126-element feature vector.

        Feature vector structure:
        - Elements 0-62: Left hand features (or zeros if not detected)
        - Elements 63-125: Right hand features (or zeros if not detected)

        Normalization is applied per-hand independently using landmark_normalizer.

        Args:
            left_landmarks: List of 21 (x, y, z) tuples for left hand, or None.
            right_landmarks: List of 21 (x, y, z) tuples for right hand, or None.

        Returns:
            Normalized 126-element numpy array (always valid, never None).
        """
        return normalize_dual_hand_features(left_landmarks, right_landmarks)

    def predict_both_hands(
        self,
        left_landmarks: Optional[List[Tuple[float, float, float]]],
        right_landmarks: Optional[List[Tuple[float, float, float]]],
    ) -> Tuple[Optional[str], float]:
        """
        Predict sign label and confidence probability from both hands (or one + padding).

        This is the primary inference method that uses the 126-element feature vector.

        Args:
            left_landmarks: 21 MediaPipe hand landmarks for left hand, or None.
            right_landmarks: 21 MediaPipe hand landmarks for right hand, or None.

        Returns:
            (sign_label_string, confidence_float), or (None, 0.0) on failure.
        """
        try:
            features = self.extract_features(left_landmarks, right_landmarks)
            proba = self.model.predict_proba([features])[0]
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            label: str = self.label_encoder.inverse_transform([best_idx])[0]
            return label, confidence
        except Exception as exc:
            logger.error(f"XGBoost prediction error: {exc}")
            return None, 0.0

    def predict(
        self, landmarks: List[Tuple[float, float, float]]
    ) -> Tuple[Optional[str], float]:
        """
        (DEPRECATED) Predict sign from a single hand pose.
        
        Kept for backward compatibility. Internally converts to the two-hand interface
        by padding the missing hand with zeros.

        Args:
            landmarks: 21 MediaPipe hand landmarks as (x, y, z) tuples.

        Returns:
            (sign_label_string, confidence_float), or (None, 0.0) on failure.
        """
        # Interpret single hand as left hand; right hand is zeros
        return self.predict_both_hands(landmarks, None)
