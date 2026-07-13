"""
Application State Module.

Centralized, thread-safe state container for the entire application.
Eliminates scattered instance variables across widgets and controllers.

Follows the Single-Responsibility Principle: one place to read/write app state.
"""

import threading
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """
    Immutable-style, thread-safe holder for runtime application state.

    All reads and writes are serialised via an internal RLock so that
    background camera / speech threads cannot corrupt UI-visible state.

    Attributes:
        current_translation: Latest recognized text (sign or speech).
        speech_status: One of "Ready", "Listening", "Processing", "Speaking".
        camera_running: True when the camera capture loop is active.
        listening: True when the speech recognizer is active.
    """

    current_translation: str = ""
    speech_status: str = "Ready"
    camera_running: bool = False
    listening: bool = False

    # Internal lock — excluded from repr/comparison
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    # ------------------------------------------------------------------ #
    # Thread-safe property helpers                                         #
    # ------------------------------------------------------------------ #

    def set_translation(self, text: str) -> None:
        with self._lock:
            self.current_translation = text

    def set_speech_status(self, status: str) -> None:
        valid = {"Ready", "Listening", "Processing", "Speaking"}
        if status not in valid:
            logger.warning("Unknown speech status '%s', defaulting to 'Ready'", status)
            status = "Ready"
        with self._lock:
            self.speech_status = status

    def set_camera_running(self, running: bool) -> None:
        with self._lock:
            self.camera_running = running

    def set_listening(self, listening: bool) -> None:
        with self._lock:
            self.listening = listening

    def snapshot(self) -> dict:
        """Return a consistent copy of all fields for logging/debugging."""
        with self._lock:
            return {
                "current_translation": self.current_translation,
                "speech_status": self.speech_status,
                "camera_running": self.camera_running,
                "listening": self.listening,
            }
