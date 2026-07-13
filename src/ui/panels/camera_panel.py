"""
Camera Panel — View Layer.

A pure CustomTkinter view component that displays the live webcam feed
with hand-landmark overlays.

Responsibilities (view only):
  - Render the camera frame label and status text.
  - Draw hand landmarks and confidence labels on incoming frames.
  - Convert OpenCV frames to CTkImage and schedule display on the UI thread.

This panel owns NO camera hardware and NO recognition logic.
All business decisions are made in CameraController and delegated here
via ``update_frame()`` and ``set_status()``.

Backward-compatible note:
  ``set_recognized_sign_callback()`` is retained for callers that use
  the old CameraPanel API directly.  In the refactored architecture the
  callback is registered at the CameraController level instead.
"""

import logging
from typing import Optional, Tuple, Callable

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image

from models.sign_detector import SignResult, SignType

logger = logging.getLogger(__name__)

# Left/right hand overlay colours (BGR)
_LEFT_COLOR: Tuple[int, int, int] = (0, 255, 0)
_RIGHT_COLOR: Tuple[int, int, int] = (0, 165, 255)

# Hand skeleton connections (MediaPipe 21-landmark model)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
]

_DISPLAY_HEIGHT = 400  # Target pixel height for the camera feed display


class CameraPanel(ctk.CTkFrame):
    """
    Live camera feed display panel (view only).

    Receives pre-processed frame data from the controller and handles
    all rendering: landmark dots, skeleton lines, confidence labels, and
    the final image conversion.

    Parameters
    ----------
    parent:
        Parent CustomTkinter widget.
    **kwargs:
        Forwarded to ``ctk.CTkFrame``.
    """

    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent, **kwargs)

        self._frame_label: Optional[ctk.CTkLabel] = None
        self._status_label: Optional[ctk.CTkLabel] = None
        # Kept for backward-compat with callers that set a sign callback directly.
        self._recognized_sign_callback: Optional[Callable] = None

        try:
            self._build_ui()
            logger.debug("CameraPanel initialized")
        except Exception:
            logger.exception("CameraPanel: error during __init__")
            raise

    # ------------------------------------------------------------------ #
    # Public view API (called by CameraController)                        #
    # ------------------------------------------------------------------ #

    def update_frame(
        self,
        frame: np.ndarray,
        left_raw: Optional[SignResult] = None,
        right_raw: Optional[SignResult] = None,
        left_stable: Optional[SignResult] = None,
        right_stable: Optional[SignResult] = None,
    ) -> None:
        """
        Draw overlays on *frame* and display it.

        Must be called on the main UI thread.

        Args:
            frame:        Raw BGR frame from the camera service.
            left_raw:     Unsmoothed left-hand result (used for landmarks).
            right_raw:    Unsmoothed right-hand result (used for landmarks).
            left_stable:  Smoothed left-hand result (used for text labels).
            right_stable: Smoothed right-hand result (used for text labels).
        """
        if frame is None or frame.size == 0:
            return
        try:
            # Draw landmarks from raw (unfiltered) results so skeleton is
            # always visible even for low-confidence detections.
            if left_raw and left_raw.landmarks:
                self._draw_landmarks(frame, left_raw.landmarks, _LEFT_COLOR)
            if right_raw and right_raw.landmarks:
                self._draw_landmarks(frame, right_raw.landmarks, _RIGHT_COLOR)

            # Draw text labels only for stable (threshold-cleared) results.
            if left_stable:
                self._draw_label(
                    frame,
                    f"L: {left_stable.sign_type.value} ({left_stable.confidence:.2f})",
                    (10, 30),
                    _LEFT_COLOR,
                )
            if right_stable:
                self._draw_label(
                    frame,
                    f"R: {right_stable.sign_type.value} ({right_stable.confidence:.2f})",
                    (10, 70),
                    _RIGHT_COLOR,
                )

            self._display_frame(frame)

        except Exception:
            logger.exception("CameraPanel.update_frame error")

    def set_status(self, text: str) -> None:
        """Update the status label text."""
        try:
            if self._status_label and self._status_label.winfo_exists():
                self._status_label.configure(text=f"Status: {text}")
        except Exception:
            logger.exception("CameraPanel.set_status error")

    # ------------------------------------------------------------------ #
    # Backward-compatibility shim                                         #
    # ------------------------------------------------------------------ #

    def set_recognized_sign_callback(self, callback: Callable) -> None:
        """
        Register a callback for recognized signs.

        Retained for API compatibility with legacy callers.
        In the refactored architecture this is wired at the controller level.
        """
        self._recognized_sign_callback = callback

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        title = ctk.CTkLabel(
            self, text="📷 Live Camera Feed", font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        self._frame_label = ctk.CTkLabel(
            self, text="Camera Inactive", fg_color="gray20", height=_DISPLAY_HEIGHT
        )
        self._frame_label.pack(fill="both", expand=True, padx=10, pady=10)

        self._status_label = ctk.CTkLabel(
            self,
            text="Status: Ready",
            fg_color="gray30",
            text_color="lightblue",
        )
        self._status_label.pack(fill="x", padx=10, pady=5)

    # ------------------------------------------------------------------ #
    # Rendering helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _draw_landmarks(
        frame: np.ndarray,
        landmarks: list,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw landmark dots and skeleton connections onto *frame* (in-place)."""
        h, w = frame.shape[:2]
        for lm in landmarks:
            cv2.circle(frame, (int(lm[0] * w), int(lm[1] * h)), 5, color, -1)

        if len(landmarks) >= 21:
            for start, end in _HAND_CONNECTIONS:
                if start < len(landmarks) and end < len(landmarks):
                    p1 = (int(landmarks[start][0] * w), int(landmarks[start][1] * h))
                    p2 = (int(landmarks[end][0] * w), int(landmarks[end][1] * h))
                    cv2.line(frame, p1, p2, color, 2)

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Draw a confidence label onto *frame* (in-place)."""
        cv2.putText(
            frame, text, origin,
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )

    def _display_frame(self, frame: np.ndarray) -> None:
        """Resize *frame*, convert to CTkImage, and schedule label update."""
        if self._frame_label is None:
            return
        try:
            h, w = frame.shape[:2]
            if h == 0 or w == 0:
                return
            display_w = int(_DISPLAY_HEIGHT * w / h)
            frame_resized = cv2.resize(frame, (display_w, _DISPLAY_HEIGHT))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            ctk_image = ctk.CTkImage(
                light_image=Image.fromarray(frame_rgb),
                size=(display_w, _DISPLAY_HEIGHT),
            )
            # Already on main thread (controller schedules via after(0, …))
            if self._frame_label.winfo_exists():
                self._frame_label.configure(image=ctk_image, text="")
                self._frame_label.image = ctk_image  # prevent GC
        except Exception:
            logger.exception("CameraPanel._display_frame error")
