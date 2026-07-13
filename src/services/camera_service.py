"""
Camera Service Module.

Owns the camera hardware lifecycle and per-frame sign-detection pipeline.
Extracted from the original CameraPanel god class to satisfy
Single-Responsibility Principle.

Responsibilities:
  - Open / release the OpenCV VideoCapture device.
  - Run the capture loop in a background thread.
  - Invoke SignRecognizer.process_frame() on every captured frame.
  - Deliver (frame, left_result, right_result) to a registered callback.

The service intentionally knows nothing about UI widgets or temporal
smoothing — those are handled by the view and RecognitionService respectively.
"""

import threading
import logging
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from models.sign_detector import SignRecognizer, SignResult

logger = logging.getLogger(__name__)

# Type alias for the per-frame callback signature.
FrameCallback = Callable[
    [np.ndarray, Optional[SignResult], Optional[SignResult]], None
]


class CameraService:
    """
    Manages webcam capture and per-frame sign detection.

    Usage::

        service = CameraService(sign_recognizer)
        service.start(on_frame_callback)
        # … later …
        service.stop()

    The *on_frame_callback* is called from the background capture thread
    with signature ``(frame, left_result, right_result)``.
    Callers that update UI widgets MUST schedule those updates on the main
    thread (e.g. ``widget.after(0, …)``).
    """

    # Camera device properties
    _FRAME_WIDTH = 640
    _FRAME_HEIGHT = 480
    _TARGET_FPS = 30

    def __init__(self, sign_recognizer: SignRecognizer) -> None:
        self._sign_recognizer = sign_recognizer
        self._camera: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self._running = False
        self._frame_callback: Optional[FrameCallback] = None

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, frame_callback: FrameCallback) -> bool:
        """
        Open the default webcam and begin frame capture.

        Args:
            frame_callback: Called for every captured frame with
                ``(frame, left_result, right_result)``.

        Returns:
            ``True`` on success, ``False`` if the camera could not be opened.
        """
        if self._running:
            logger.warning("CameraService.start() called while already running")
            return False

        try:
            self._camera = cv2.VideoCapture(0)
            if not self._camera.isOpened():
                logger.error("Cannot open camera device 0")
                return False

            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._FRAME_WIDTH)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._FRAME_HEIGHT)
            self._camera.set(cv2.CAP_PROP_FPS, self._TARGET_FPS)

            self._frame_callback = frame_callback
            self._running = True

            with self._thread_lock:
                self._thread = threading.Thread(
                    target=self._capture_loop, daemon=True, name="CameraCapture"
                )
                self._thread.start()

            logger.info("CameraService started")
            return True

        except Exception:
            logger.exception("Failed to start CameraService")
            self._running = False
            return False

    def stop(self) -> None:
        """Stop capture and release the camera device."""
        self._running = False

        with self._thread_lock:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)

        if self._camera:
            self._camera.release()
            self._camera = None

        logger.info("CameraService stopped")

    def cleanup(self) -> None:
        """Alias for stop() — used by callers that follow the cleanup protocol."""
        self.stop()

    # ------------------------------------------------------------------ #
    # Internal capture loop                                                #
    # ------------------------------------------------------------------ #

    def _capture_loop(self) -> None:
        """Background thread: read frames and invoke the registered callback."""
        try:
            while self._running:
                try:
                    ret, frame = self._camera.read()
                    if not ret:
                        logger.error("Camera read() returned no frame — stopping")
                        break

                    # Mirror the image so the user sees a selfie view.
                    frame = cv2.flip(frame, 1)

                    # Run hand detection + sign classification.
                    left_result, right_result = self._sign_recognizer.process_frame(frame)

                    if self._frame_callback:
                        self._frame_callback(frame, left_result, right_result)

                except Exception:
                    logger.exception("Error inside camera capture loop — continuing")

        except Exception:
            logger.exception("Fatal error in camera capture loop")
        finally:
            self._running = False
