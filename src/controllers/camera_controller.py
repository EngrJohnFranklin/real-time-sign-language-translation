"""
Camera Controller Module.

Handles all camera-related user events and routes data between the
CameraService, RecognitionService, and the camera view panel.

Responsibilities:
  - Toggle the camera on / off in response to the Start/Stop button.
  - Receive raw frames from CameraService, apply temporal smoothing via
    RecognitionService, then dispatch the overlaid frame to the view.
  - Fire the *sign_recognized_callback* exactly once per newly stable sign.
  - Update AppState to reflect camera running state.

Knows about services and state — does NOT contain UI widget logic.
"""

import logging
import time
from collections import deque
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from app.app_state import AppState
from services.camera_service import CameraService
from services.recognition_service import RecognitionService
from models.sign_detector import SignResult
from models.word_recognizer import WordRecognizer

logger = logging.getLogger(__name__)


class WordRecognitionState(Enum):
    """Lifecycle of one word-signing session."""

    IDLE = auto()
    COLLECTING = auto()
    COMMITTED = auto()
    WAIT_FOR_RELEASE = auto()


class CameraController:
    """
    Mediates between the camera Start/Stop button, CameraService,
    RecognitionService, and the camera view panel.

    The *view* is any object that exposes:
      - ``update_frame(frame, left_raw, right_raw, left_stable, right_stable)``
      - ``set_status(text: str)``
      - ``set_camera_button_text(text: str)``

    The optional *sign_recognized_callback* has signature::

        callback(sign_name: str, hand_side: str, confidence: float) -> None

    and is called from the capture background thread; the controller
    schedules UI work via ``view.after(0, …)``.
    """

    WORD_WINDOW_FRAMES = 300
    DETECTION_WINDOW_SECONDS = 3.0
    STATIC_CONFIDENCE_THRESHOLD = 0.92
    MOTION_CONFIDENCE_THRESHOLD = 0.80
    COMMITTED_HOLD_SECONDS = 1.5
    RELEASE_FRAMES = 20
    ENABLE_LETTER_RECOGNITION = False

    def __init__(
        self,
        camera_service: CameraService,
        recognition_service: RecognitionService,
        state: AppState,
        view,
        sign_recognized_callback: Optional[Callable[[str, str, float], None]] = None,
        word_recognizer: Optional[WordRecognizer] = None,
    ) -> None:
        self._camera = camera_service
        self._recognition = recognition_service
        self._state = state
        self._view = view
        self._sign_cb = sign_recognized_callback
        self._word_recognizer = word_recognizer
        self._word_frames = deque(maxlen=self.WORD_WINDOW_FRAMES)
        self._word_state = WordRecognitionState.IDLE
        self._collection_started_at: Optional[float] = None
        self._absent_frames = 0
        self._committed_at: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public event handlers (called by the view)                          #
    # ------------------------------------------------------------------ #

    def on_camera_button_clicked(self) -> None:
        """Toggle the camera on or off."""
        if self._state.camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _start_camera(self) -> None:
        if self._camera.start(self._on_frame_received):
            self._state.set_camera_running(True)
            self._view.set_status("Camera running")
            self._view.set_camera_button_text("⏹ Stop Camera")
            logger.info("Camera started")
        else:
            self._view.set_status("Failed to start camera")
            logger.error("CameraController: failed to start camera")

    def _stop_camera(self) -> None:
        self._camera.stop()
        self._state.set_camera_running(False)
        self._recognition.reset()
        self._reset_word_session()
        self._view.set_status("Camera stopped")
        self._view.set_camera_button_text("▶ Start Camera")
        logger.info("Camera stopped")

    def _on_frame_received(
        self,
        frame: np.ndarray,
        left_raw: Optional[SignResult],
        right_raw: Optional[SignResult],
        features: Optional[np.ndarray],
    ) -> None:
        """
        Invoked on the camera background thread for every captured frame.

        Applies temporal smoothing, then schedules a UI update on the main
        thread.  Sign-recognized callbacks are also scheduled on the main
        thread to ensure thread-safe widget access.
        """
        try:
            (
                left_stable, left_new,
                right_stable, right_new,
            ) = self._recognition.update(left_raw, right_raw)

            # Schedule view update on main thread.
            self._view.after(
                0,
                lambda f=frame, lr=left_raw, rr=right_raw, ls=left_stable, rs=right_stable: (
                    self._view.update_frame(f, lr, rr, ls, rs)
                ),
            )

            word_match = self._advance_word_session(features)
            if word_match and self._sign_cb:
                word, confidence = word_match
                self._view.after(
                    0,
                    lambda: self._sign_cb(word, "Both", confidence),
                )
                return

            # Optional legacy fallback, disabled while word recognition is tuned.
            if (
                self.ENABLE_LETTER_RECOGNITION
                and left_stable
                and left_new
                and self._sign_cb
            ):
                self._view.after(
                    0,
                    lambda s=left_stable: self._sign_cb(
                        s.sign_type.value, "Left", s.confidence
                    ),
                )
            if (
                self.ENABLE_LETTER_RECOGNITION
                and right_stable
                and right_new
                and self._sign_cb
            ):
                self._view.after(
                    0,
                    lambda s=right_stable: self._sign_cb(
                        s.sign_type.value, "Right", s.confidence
                    ),
                )

        except Exception:
            logger.exception("CameraController._on_frame_received error")

    def _advance_word_session(
        self, features: Optional[np.ndarray]
    ) -> Optional[tuple[str, float]]:
        """Advance one word-signing session and return only committed matches."""
        if self._word_recognizer is None:
            return None

        if features is None:
            return self._handle_missing_features()

        self._absent_frames = 0
        if self._word_state in (
            WordRecognitionState.COMMITTED,
            WordRecognitionState.WAIT_FOR_RELEASE,
        ):
            return None

        if self._word_state == WordRecognitionState.IDLE:
            self._begin_word_session()

        self._word_frames.append(features)
        if (
            self._collection_started_at is None
            or time.monotonic() - self._collection_started_at
            < self.DETECTION_WINDOW_SECONDS
        ):
            return None

        sequence = np.asarray(self._word_frames, dtype=np.float32)
        motion_word, motion_confidence = (
            self._word_recognizer.recognize_motion_from_sequence(sequence)
        )
        if motion_word and motion_confidence >= self.MOTION_CONFIDENCE_THRESHOLD:
            return self._commit_word((motion_word, motion_confidence))

        static_word, static_confidence = (
            self._word_recognizer.recognize_static_from_frame(features)
        )
        if static_word and static_confidence >= self.STATIC_CONFIDENCE_THRESHOLD:
            return self._commit_word((static_word, static_confidence))

        self._word_state = WordRecognitionState.WAIT_FOR_RELEASE
        return None

    def _begin_word_session(self) -> None:
        """Initialize collection state when a hand first enters the frame."""
        self._word_state = WordRecognitionState.COLLECTING
        self._collection_started_at = time.monotonic()
        self._word_frames.clear()

    def _commit_word(self, match: tuple[str, float]) -> tuple[str, float]:
        """Lock the result until the signer releases their hands."""
        self._word_state = WordRecognitionState.COMMITTED
        self._committed_at = time.monotonic()
        return match

    def _handle_missing_features(self) -> Optional[tuple[str, float]]:
        """Reset incomplete sessions or release a committed word after hand absence."""
        if self._word_state == WordRecognitionState.COLLECTING:
            self._reset_word_session()
            return None

        self._absent_frames += 1
        released_after_commit = (
            self._word_state != WordRecognitionState.COMMITTED
            or (
                self._committed_at is not None
                and time.monotonic() - self._committed_at >= self.COMMITTED_HOLD_SECONDS
            )
        )
        if released_after_commit and self._absent_frames >= self.RELEASE_FRAMES:
            self._reset_word_session()
        return None

    def _reset_word_session(self) -> None:
        """Clear all state so the next visible hand starts a new session."""
        self._word_state = WordRecognitionState.IDLE
        self._word_frames.clear()
        self._collection_started_at = None
        self._absent_frames = 0
        self._committed_at = None
