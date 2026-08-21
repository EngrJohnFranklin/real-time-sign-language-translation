"""Word-level sign recognition using DTW and single-frame classification.

Routes different word types through appropriate recognition pipelines:
- Motion-based words (hello, thank_you, sorry, yes, please, good, help)
  use DTW sequence matching against recorded templates.
- Static words (love, i_love_you, stop) use single-frame XGBoost-style
  classification on a dedicated word classifier model.

This module is intentionally independent from sign_detector.py to avoid
disrupting the existing letter recognition pipeline.
"""

import logging
from typing import Optional, Tuple, Dict, List
import numpy as np

from translation.dtw_matcher import dtw_distance
from translation.word_template_loader import (
    WordTemplateManager,
    MOTION_BASED_WORDS,
    STATIC_WORDS,
    WORD_CLASSES,
)

logger = logging.getLogger(__name__)


class WordRecognizer:
    """Recognizes PSL words using DTW and single-frame classification."""

    def __init__(self, word_classifier_path: Optional[str] = None):
        """Initialize word recognizer.
        
        Args:
            word_classifier_path: Path to trained word classifier model for static words.
                                 If None, static word recognition is disabled.
        """
        self.template_manager = WordTemplateManager()
        self.template_manager.load_all_templates()
        
        # Track template stats
        sample_counts = self.template_manager.count_samples_per_word()
        logger.info(f"Loaded word templates: {sample_counts}")
        
        # Load word classifier for static words if available
        self.word_classifier = None
        if word_classifier_path:
            try:
                # Lazy import to avoid circular dependencies
                from models.word_classifier import XGBoostWordClassifier
                self.word_classifier = XGBoostWordClassifier(word_classifier_path)
                logger.info("Loaded static word classifier")
            except Exception as e:
                logger.warning(f"Could not load static word classifier: {e}")
        
        # DTW distance threshold for accepting matches
        self.dtw_threshold = 10.0  # Tunable parameter

    def recognize_from_sequence(
        self, landmark_sequence: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Recognize a word from a temporal landmark sequence.
        
        Args:
            landmark_sequence: Array of shape (T, 126) representing normalized
                             dual-hand landmarks over time.
        
        Returns:
            (word_label, confidence) or (None, 0.0) if no match found.
        """
        if landmark_sequence is None or landmark_sequence.shape[0] == 0:
            return None, 0.0
        
        if landmark_sequence.ndim != 2 or landmark_sequence.shape[1] != 126:
            logger.warning(
                f"Invalid sequence shape {landmark_sequence.shape}; "
                f"expected (T, 126)"
            )
            return None, 0.0
        
        motion_word, motion_confidence = self.recognize_motion_from_sequence(
            landmark_sequence
        )
        if motion_word is not None:
            return motion_word, motion_confidence
        
        return self.recognize_static_from_frame(landmark_sequence[-1])

    def recognize_motion_from_sequence(
        self, landmark_sequence: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Recognize only motion-based words from a landmark sequence."""
        best_word, best_confidence, _ = self.recognize_motion_candidates(
            landmark_sequence
        )
        return best_word, best_confidence

    def recognize_motion_candidates(
        self, landmark_sequence: np.ndarray
    ) -> Tuple[Optional[str], float, float]:
        """Return the best motion word, its confidence, and runner-up confidence."""
        if landmark_sequence is None or landmark_sequence.ndim != 2:
            return None, 0.0, 0.0

        word_distances = []
        for word in MOTION_BASED_WORDS:
            templates = self.template_manager.get_templates_for_word(word)
            distances = []
            for template in templates:
                try:
                    distances.append(dtw_distance(template, landmark_sequence))
                except Exception as error:
                    logger.debug("DTW error for '%s': %s", word, error)

            if distances:
                word_distances.append((word, float(np.mean(distances))))

        if not word_distances:
            return None, 0.0, 0.0

        word_distances.sort(key=lambda item: item[1])
        best_word, best_distance = word_distances[0]
        best_confidence = self._distance_to_confidence(best_distance)
        runner_up_confidence = (
            self._distance_to_confidence(word_distances[1][1])
            if len(word_distances) > 1
            else 0.0
        )

        return best_word, best_confidence, runner_up_confidence

    def recognize_static_from_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Recognize only static words from one normalized feature frame."""
        if self.word_classifier is not None:
            static_word, static_conf = self._classify_static_word(frame)
            if static_word is not None and static_conf > 0.5:
                return static_word, static_conf
        
        return None, 0.0

    def _match_motion_words(self, query_sequence: np.ndarray) -> Tuple[Optional[str], float]:
        """Match query against motion-based word templates using DTW.
        
        Returns:
            (best_word, min_distance) or (None, inf) if no templates available.
        """
        best_word = None
        min_distance = float('inf')
        
        for word in MOTION_BASED_WORDS:
            templates = self.template_manager.get_templates_for_word(word)
            if not templates:
                continue
            
            # Compute DTW distance to each template
            word_distances = []
            for template in templates:
                try:
                    dist = dtw_distance(template, query_sequence)
                    word_distances.append(dist)
                except Exception as e:
                    logger.debug(f"DTW error for '{word}': {e}")
                    continue
            
            if word_distances:
                avg_distance = np.mean(word_distances)
                if avg_distance < min_distance:
                    min_distance = avg_distance
                    best_word = word
        
        return best_word, min_distance

    def _classify_static_word(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """Classify a single frame as a static word using the word classifier.
        
        Args:
            frame: Array of shape (1, 126) or (126,).
        
        Returns:
            (word_label, confidence) or (None, 0.0) if classifier unavailable.
        """
        if self.word_classifier is None:
            return None, 0.0
        
        try:
            label, confidence = self.word_classifier.predict(frame)
            return label, confidence
        except Exception as e:
            logger.debug(f"Static word classification error: {e}")
            return None, 0.0

    def _distance_to_confidence(self, distance: float) -> float:
        """Convert DTW distance to confidence score [0, 1].
        
        Lower distances should map to higher confidence.
        Using inverse sigmoid-like curve.
        """
        if distance < 0:
            return 0.0
        if distance > self.dtw_threshold * 2:
            return 0.0
        
        # Simple inverse mapping: confidence = 1 - (distance / threshold)
        confidence = max(0.0, 1.0 - (distance / self.dtw_threshold))
        return confidence

    def get_loaded_words(self) -> Dict[str, int]:
        """Return count of templates per word."""
        return self.template_manager.count_samples_per_word()

    def get_word_type(self, word: str) -> str:
        """Return 'motion', 'static', or 'unknown' for a word."""
        if word in MOTION_BASED_WORDS:
            return "motion"
        if word in STATIC_WORDS:
            return "static"
        return "unknown"
