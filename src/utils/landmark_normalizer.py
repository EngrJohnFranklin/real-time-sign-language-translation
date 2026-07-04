"""
Landmark normalization preprocessing for hand pose data.

Normalizes MediaPipe hand landmarks by:
1. Translating to wrist origin (landmark 0)
2. Scaling by hand size (wrist-to-middle-finger-MCP distance)
3. Outputting relative, scale-invariant coordinates

Usage:
    landmarks = [...] # 21 MediaPipe hand landmarks
    normalized = normalize_landmarks(landmarks)
"""

import logging
import numpy as np
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# MediaPipe hand landmark indices
WRIST_IDX = 0
MIDDLE_FINGER_MCP_IDX = 9  # Middle finger metacarpophalangeal joint (start of middle finger)
NUM_HAND_LANDMARKS = 21


def normalize_landmarks(
    landmarks: Optional[List[Tuple[float, float, float]]], min_scale: float = 1e-6
) -> Optional[np.ndarray]:
    """
    Normalize a single hand's 21 landmarks to be scale and translation invariant.

    Normalization process:
    1. Translate all landmarks so wrist (landmark 0) is at origin (0, 0, 0).
    2. Compute hand size as Euclidean distance from wrist to middle-finger MCP.
    3. Scale all coordinates by hand size to achieve scale invariance.
    4. Return normalized landmarks as 1-D array of 63 values (21 landmarks × 3 axes).

    Args:
        landmarks: List of 21 (x, y, z) tuples from MediaPipe hands, or None.
        min_scale: Minimum hand size threshold to prevent division by zero.
                   If hand size < min_scale, returns None.

    Returns:
        Normalized 63-element numpy array (float32), or None if:
        - Input is None or has fewer than 21 landmarks
        - Hand size is below min_scale threshold
        - Any error occurs during normalization

    Notes:
        - Result is scale and translation invariant (robust to distance/hand size variation)
        - Z-axis is included (important for depth perception)
        - This must be applied consistently during training AND inference
    """
    if landmarks is None or len(landmarks) < NUM_HAND_LANDMARKS:
        return None

    try:
        # Convert to numpy array
        pts = np.array(landmarks, dtype=np.float32)  # shape (21, 3)

        # Step 1: Translate to wrist origin
        wrist = pts[WRIST_IDX]
        pts = pts - wrist  # Now wrist is at (0, 0, 0)

        # Step 2: Compute hand size (distance from wrist to middle-finger MCP)
        middle_mcp = pts[MIDDLE_FINGER_MCP_IDX]
        hand_size = float(np.linalg.norm(middle_mcp))

        # Step 3: Check for divide-by-zero
        if hand_size < min_scale:
            logger.debug(
                f"Hand size {hand_size} below minimum {min_scale}; "
                "landmarks likely too close together or invalid"
            )
            return None

        # Step 4: Normalize by hand size
        pts = pts / hand_size

        # Step 5: Flatten to 63-element vector
        return pts.flatten()  # shape (63,)

    except Exception as exc:
        logger.error(f"Landmark normalization error: {exc}")
        return None


def normalize_dual_hand_features(
    left_landmarks: Optional[List[Tuple[float, float, float]]] = None,
    right_landmarks: Optional[List[Tuple[float, float, float]]] = None,
    min_scale: float = 1e-6,
) -> np.ndarray:
    """
    Normalize up to 2 hands and combine into a single 126-element feature vector.

    Feature vector structure:
    - Elements 0-62: Left hand features (or zeros if not detected)
    - Elements 63-125: Right hand features (or zeros if not detected)

    This is the primary interface for converting raw landmarks to ML-ready features.

    Args:
        left_landmarks: List of 21 (x, y, z) tuples for left hand, or None.
        right_landmarks: List of 21 (x, y, z) tuples for right hand, or None.
        min_scale: Minimum hand size threshold (passed to normalize_landmarks).

    Returns:
        126-element numpy array (float32) with normalized features.
        Missing hands are represented as 63 zeros.

    Notes:
        - Always returns a valid 126-element array (never None)
        - Each hand is normalized independently
        - Consistent with XGBoost model training expectations
    """
    try:
        # Normalize each hand independently
        left_feats = normalize_landmarks(left_landmarks, min_scale)
        if left_feats is None:
            left_feats = np.zeros(63, dtype=np.float32)

        right_feats = normalize_landmarks(right_landmarks, min_scale)
        if right_feats is None:
            right_feats = np.zeros(63, dtype=np.float32)

        # Concatenate: left (63) + right (63) = 126 total
        return np.concatenate([left_feats, right_feats], dtype=np.float32)
    except Exception as exc:
        logger.error(f"Dual-hand feature normalization error: {exc}")
        # Fallback: return all zeros
        return np.zeros(126, dtype=np.float32)


def normalize_batch_landmarks(
    landmarks_batch: List[List[Tuple[float, float, float]]],
    min_scale: float = 1e-6,
) -> np.ndarray:
    """
    Normalize a batch of landmarks (e.g., for training dataset).

    Args:
        landmarks_batch: List of landmark sequences, each with 21 (x, y, z) tuples.
        min_scale: Minimum hand size threshold.

    Returns:
        Array of shape (batch_size, 63) with normalized features.
        Invalid landmarks are replaced with zeros.
    """
    normalized = []
    for landmarks in landmarks_batch:
        norm_lms = normalize_landmarks(landmarks, min_scale)
        if norm_lms is None:
            norm_lms = np.zeros(63, dtype=np.float32)
        normalized.append(norm_lms)
    return np.array(normalized, dtype=np.float32)
