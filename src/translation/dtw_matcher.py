"""Standalone Dynamic Time Warping utilities for word-template comparison.

This module intentionally stays independent from MediaPipe, OpenCV, XGBoost,
UI, and camera logic. It compares two temporal feature sequences of shape
(T, 126) using standard DTW with Euclidean frame-to-frame distance.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Iterable[Sequence[float]]]


def _validate_sequence(sequence: ArrayLike, name: str) -> np.ndarray:
    """Validate and convert a sequence to a 2D NumPy array.

    Args:
        sequence: Input sequence shaped like (T, 126).
        name: Human-readable label for validation errors.

    Returns:
        NumPy array with shape (T, 126).

    Raises:
        ValueError: If the sequence is empty, invalid, or not 2D.
    """
    arr = np.asarray(sequence, dtype=np.float64)

    if arr.size == 0:
        raise ValueError(f"{name} is empty and cannot be used for DTW.")

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (T, 126).")

    if arr.shape[1] != 126:
        raise ValueError(
            f"{name} must have shape (T, 126); got {arr.shape} instead."
        )

    return arr


def _euclidean_distance(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Return Euclidean distance between two 126-dimensional feature vectors."""
    return float(np.linalg.norm(frame_a - frame_b))


def dtw_distance(reference_sequence: ArrayLike, query_sequence: ArrayLike) -> float:
    """Compute a normalized DTW distance between two temporal feature sequences.

    Args:
        reference_sequence: Reference sequence shaped (T1, 126).
        query_sequence: Query sequence shaped (T2, 126).

    Returns:
        A normalized DTW distance in the range [0, inf). Lower values indicate
        greater similarity.

    Raises:
        ValueError: If either sequence is empty or has an invalid feature width.
    """
    ref = _validate_sequence(reference_sequence, "reference_sequence")
    query = _validate_sequence(query_sequence, "query_sequence")

    t1 = ref.shape[0]
    t2 = query.shape[0]

    if t1 == 0 or t2 == 0:
        raise ValueError("DTW requires non-empty sequences.")

    cost_matrix = np.full((t1 + 1, t2 + 1), np.inf, dtype=np.float64)
    cost_matrix[0, 0] = 0.0

    for i in range(1, t1 + 1):
        for j in range(1, t2 + 1):
            local_cost = _euclidean_distance(ref[i - 1], query[j - 1])
            cost_matrix[i, j] = local_cost + min(
                cost_matrix[i - 1, j],
                cost_matrix[i, j - 1],
                cost_matrix[i - 1, j - 1],
            )

    raw_cost = cost_matrix[t1, t2]
    normalization_factor = max(t1, t2)
    if normalization_factor == 0:
        return 0.0

    return float(raw_cost / normalization_factor)
