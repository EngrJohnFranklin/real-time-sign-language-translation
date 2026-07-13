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
from typing import Callable, Optional

import numpy as np

from app.app_state import AppState
from services.camera_service import CameraService
from services.recognition_service import RecognitionService
from models.sign_detector import SignResult

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        camera_service: CameraService,
        recognition_service: RecognitionService,
        state: AppState,
        view,
        sign_recognized_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        self._camera = camera_service
        self._recognition = recognition_service
        self._state = state
        self._view = view
        self._sign_cb = sign_recognized_callback

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
        self._view.set_status("Camera stopped")
        self._view.set_camera_button_text("▶ Start Camera")
        logger.info("Camera stopped")

    def _on_frame_received(
        self,
        frame: np.ndarray,
        left_raw: Optional[SignResult],
        right_raw: Optional[SignResult],
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

            # Fire sign callbacks for newly-confirmed signs.
            if left_stable and left_new and self._sign_cb:
                self._view.after(
                    0,
                    lambda s=left_stable: self._sign_cb(
                        s.sign_type.value, "Left", s.confidence
                    ),
                )
            if right_stable and right_new and self._sign_cb:
                self._view.after(
                    0,
                    lambda s=right_stable: self._sign_cb(
                        s.sign_type.value, "Right", s.confidence
                    ),
                )

        except Exception:
            logger.exception("CameraController._on_frame_received error")
