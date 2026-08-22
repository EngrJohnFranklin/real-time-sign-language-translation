"""Single-frame XGBoost classifier for static PSL words.

Separate from the letter-recognition XGBoost model. Handles classification
of static or nearly-static words using a single
126-dimensional normalized dual-hand feature vector.

Model is trained on single frames from word template recordings.
Expected model path: data/models/word_model.pkl
"""

import os
import pickle
import logging
import pathlib
import numpy as np
from typing import Optional, Tuple, List

from utils.landmark_normalizer import normalize_dual_hand_features
from utils.paths import get_model_path

logger = logging.getLogger(__name__)

DEBUG_WORD_CLASSIFIER = False

# Default model location
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
_DEFAULT_WORD_MODEL_PATH = _PROJECT_ROOT / "data" / "models" / "word_model.pkl"


class XGBoostWordClassifier:
    """Single-frame XGBoost classifier for static word recognition.
    
    Accepts a single normalized 126-element feature vector and returns
    a predicted word label with confidence.
    
    This is a separate model from sign_detector.py's letter classifier
    to avoid mixing word and letter labels.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Load a trained XGBoost word model bundle from disk.
        
        Args:
            model_path: Path to the .pkl bundle. If None, uses default location.
        
        Raises:
            FileNotFoundError: Model file does not exist.
            RuntimeError: Bundle is corrupted or incompatible.
        """
        if model_path is None:
            model_path = str(_DEFAULT_WORD_MODEL_PATH)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Word model not found: {model_path}")
        
        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.label_encoder = bundle["label_encoder"]
            self.feature_count = bundle.get("feature_count", 126)
            self.classes: List[str] = bundle.get("classes", list(self.label_encoder.classes_))
            logger.info(
                f"XGBoostWordClassifier loaded: {len(self.classes)} classes from {model_path}"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load word model bundle: {exc}") from exc

    def predict(self, features: np.ndarray) -> Tuple[Optional[str], float]:
        """Predict word label and confidence from a feature vector.
        
        Args:
            features: Either (126,) or (1, 126) normalized dual-hand feature array.
        
        Returns:
            (word_label, confidence) or (None, 0.0) on error.
        """
        try:
            # Handle both 1D and 2D input
            if features.ndim == 1:
                if features.shape[0] != 126:
                    logger.error(f"Expected 126 features, got {features.shape[0]}")
                    return None, 0.0
                features = features.reshape(1, -1)
            elif features.ndim == 2:
                if features.shape[1] != 126:
                    logger.error(f"Expected 126 features, got {features.shape[1]}")
                    return None, 0.0
            else:
                logger.error(f"Invalid feature shape: {features.shape}")
                return None, 0.0
            
            # Predict
            proba = self.model.predict_proba(features)[0]
            best_idx = int(np.argmax(proba))
            confidence = float(proba[best_idx])
            label: str = self.label_encoder.inverse_transform([best_idx])[0]
            if DEBUG_WORD_CLASSIFIER:
                probabilities = dict(zip(self.label_encoder.classes_, proba))
                logger.info(
                    "Static word frame: predicted=%s probability=%.4f probabilities=%s",
                    label,
                    confidence,
                    probabilities,
                )
            return label, confidence
        except Exception as exc:
            logger.error(f"Word classification error: {exc}")
            return None, 0.0
