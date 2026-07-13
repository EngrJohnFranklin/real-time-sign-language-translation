"""
Recognition Service Module.

Provides temporal smoothing for sign detection predictions.
Extracted from the original CameraPanel to satisfy
Single-Responsibility Principle.

Responsibilities:
  - Maintain per-hand prediction histories.
  - Apply majority-vote stability filter (3-of-4 frames).
  - Expose a hold-down counter so stable labels persist briefly
    after the hand moves away.
  - Determine whether a raw SignResult should be shown to the user
    (confidence gate + UNKNOWN filter).

The service owns no UI widgets and no camera hardware.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

from models.sign_detector import SignResult, SignType

logger = logging.getLogger(__name__)


@dataclass
class _HandState:
    """Mutable per-hand temporal state.  Not part of the public API."""
    history: deque = field(default_factory=lambda: deque(maxlen=4))
    stable_result: Optional[SignResult] = None
    hold_counter: int = 0


class RecognitionService:
    """
    Applies temporal smoothing to raw per-frame sign predictions.

    A prediction is considered *stable* when the same sign label appears
    in at least ``STABILITY_THRESHOLD`` out of the last ``STABILITY_FRAMES``
    frames.  Once stable the label is held for ``LABEL_HOLD_FRAMES`` frames
    to prevent flickering when detection briefly drops.

    Usage::

        service = RecognitionService()
        left_stable, left_new, right_stable, right_new = service.update(
            left_result, right_result
        )
    """

    CONFIDENCE_THRESHOLD: float = 0.75
    STABILITY_FRAMES: int = 4
    STABILITY_THRESHOLD: int = 3   # votes required out of STABILITY_FRAMES
    LABEL_HOLD_FRAMES: int = 6

    def __init__(self) -> None:
        self._left = _HandState(history=deque(maxlen=self.STABILITY_FRAMES))
        self._right = _HandState(history=deque(maxlen=self.STABILITY_FRAMES))

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def update(
        self,
        left_result: Optional[SignResult],
        right_result: Optional[SignResult],
    ) -> Tuple[
        Optional[SignResult], bool,  # left stable + is_new
        Optional[SignResult], bool,  # right stable + is_new
    ]:
        """
        Process one frame's raw results and return stable outputs.

        Args:
            left_result:  Raw SignResult for the left hand (may be None).
            right_result: Raw SignResult for the right hand (may be None).

        Returns:
            ``(left_stable, left_is_new, right_stable, right_is_new)``

            *_stable*  — the smoothed SignResult to display (or None).
            *_is_new*  — True when the stable label changed this frame
                         (use to trigger callbacks exactly once per sign).
        """
        left_stable, left_new = self._update_hand(left_result, self._left)
        right_stable, right_new = self._update_hand(right_result, self._right)
        return left_stable, left_new, right_stable, right_new

    def reset(self) -> None:
        """Clear all per-hand history and hold counters."""
        self._left = _HandState(history=deque(maxlen=self.STABILITY_FRAMES))
        self._right = _HandState(history=deque(maxlen=self.STABILITY_FRAMES))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _is_displayable(self, result: Optional[SignResult]) -> bool:
        """Return True when the result clears the confidence + UNKNOWN gate."""
        if not result:
            return False
        if result.sign_type == SignType.UNKNOWN:
            return False
        return result.confidence >= self.CONFIDENCE_THRESHOLD

    def _update_hand(
        self, result: Optional[SignResult], state: _HandState
    ) -> Tuple[Optional[SignResult], bool]:
        """Apply temporal smoothing for a single hand and return (stable, is_new)."""
        is_valid = self._is_displayable(result)
        current_label = result.sign_type.value if is_valid else None
        state.history.append(current_label)

        is_newly_accepted = False

        if len(state.history) == self.STABILITY_FRAMES:
            # Count non-None label occurrences
            counts: dict = {}
            for lbl in state.history:
                if lbl is not None:
                    counts[lbl] = counts.get(lbl, 0) + 1

            best_label = max(counts, key=counts.get) if counts else None
            best_count = counts.get(best_label, 0) if best_label else 0

            if best_count >= self.STABILITY_THRESHOLD:
                # Stable prediction
                previous_label = (
                    state.stable_result.sign_type.value if state.stable_result else None
                )
                if current_label == best_label:
                    state.stable_result = result
                state.hold_counter = self.LABEL_HOLD_FRAMES
                is_newly_accepted = previous_label != best_label
            else:
                # Not stable enough — enter hold-down phase
                if state.hold_counter > 0:
                    state.hold_counter -= 1
                else:
                    state.stable_result = None

        elif state.stable_result and state.hold_counter > 0:
            state.hold_counter -= 1
        else:
            state.stable_result = None
            state.hold_counter = 0

        return state.stable_result, is_newly_accepted
