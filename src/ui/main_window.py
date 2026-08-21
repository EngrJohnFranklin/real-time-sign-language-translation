"""
Main Window Module — View Layer.

Provides the top-level CustomTkinter application window.
This module owns only UI layout and widget wiring; all business
logic is delegated to the controller layer.

Layout (2-column grid):
  Row 0  — Live Camera (left) | Sign-Language Avatar / video (right)
  Row 1  — Current Translation (left) | Speech Status (right)
  Row 2  — Control buttons (full width)
  Row 3  — Status bar (full width)

How it works:
  1. MainWindow creates all domain services (CameraService, SpeechService …).
  2. It creates the three controllers, passing itself as the *view*.
  3. Controllers call back into MainWindow via the small *view protocol*
     (``update_translation``, ``update_speech_status``, ``set_status`` …).
  4. UI events (button clicks) delegate immediately to the controllers.

Public import path (backward-compatible)::

    from ui.main_window import MainWindow
    # or via the legacy shim:
    from ui.gui import MainWindow
"""

import logging
import os
import sys

import customtkinter as ctk

# ---------------------------------------------------------------------------
# Ensure src/ is on the path when this module is imported standalone
# ---------------------------------------------------------------------------
_src_dir = os.path.join(os.path.dirname(__file__), "..")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from app.app_state import AppState
from models.sign_detector import SignRecognizer
from translation.speech_handler import (
    SpeechHandler,
    SpeechLanguage,
    VoskSpeechRecognizer,
)
from translation.sign_to_text import SignToTextConverter
from services.camera_service import CameraService
from services.recognition_service import RecognitionService
from services.speech_service import SpeechService
from services.video_service import VideoService
from controllers.camera_controller import CameraController
from controllers.speech_controller import SpeechController
from controllers.sign_controller import SignController

logger = logging.getLogger(__name__)


class MainWindow(ctk.CTk):
    """
    Root application window — pure layout + controller wiring.

    All user-visible behaviour is implemented in the controller layer.
    This class acts as the *View* in the MVC split.
    """

    def __init__(self) -> None:
        super().__init__()

        self.title("Real-Time Sign Language Translation System")
        self.geometry("1400x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- Shared state ---
        self._state = AppState()

        # --- Domain services (infrastructure) ---
        self._sign_recognizer: SignRecognizer
        self._speech_handler: SpeechHandler
        self._sign_to_text: SignToTextConverter

        # --- Application services (facades) ---
        self._camera_svc: CameraService
        self._recognition_svc: RecognitionService
        self._speech_svc: SpeechService
        self._video_svc: VideoService

        # --- Controllers ---
        self._camera_ctrl: CameraController
        self._speech_ctrl: SpeechController
        self._sign_ctrl: SignController

        # --- View references (status bar is window-level; per-mode views hold the rest) ---
        self._view_container: ctk.CTkFrame
        self._status_bar: ctk.CTkLabel
        self._active_view = None

        try:
            self._init_services()
            self._init_controllers()
            self._build_layout()
            self._warn_if_filipino_model_unavailable()
            self.protocol("WM_DELETE_WINDOW", self._on_window_close)
            logger.info("MainWindow initialised successfully")
        except Exception:
            logger.exception("MainWindow initialisation failed")
            raise

    # ================================================================== #
    # Initialisation                                                       #
    # ================================================================== #

    def _init_services(self) -> None:
        """Instantiate all domain and application services."""
        logger.info("Initialising domain services…")
        self._sign_recognizer = SignRecognizer()
        self._speech_handler = SpeechHandler(
            language=SpeechLanguage.ENGLISH,
            tts_rate=150,
            tts_volume=0.8,
        )
        self._sign_to_text = SignToTextConverter()

        self._camera_svc = CameraService(self._sign_recognizer, parent=self)
        self._recognition_svc = RecognitionService()
        self._speech_svc = SpeechService(self._speech_handler)
        self._video_svc = VideoService(None)  # kept for SpeechController compatibility
        logger.info("Domain services ready")

    def _init_controllers(self) -> None:
        """Wire controllers — view callbacks are bound later after widgets exist."""
        self._sign_ctrl = SignController(
            self._sign_to_text, self._speech_svc, self._state, self
        )
        self._camera_ctrl = CameraController(
            self._camera_svc,
            self._recognition_svc,
            self._state,
            self,
            sign_recognized_callback=self._sign_ctrl.on_sign_recognized,
        )
        self._speech_ctrl = SpeechController(
            self._speech_svc, self._video_svc, self._state, self
        )

    def _warn_if_filipino_model_unavailable(self) -> None:
        """Show a visible warning when local Filipino speech recognition is unavailable."""
        if not VoskSpeechRecognizer.is_model_available(SpeechLanguage.FILIPINO):
            self.update_status(
                "Warning: Filipino recognition is unavailable until a Filipino "
                "Vosk model is installed."
            )

    # ================================================================== #
    # Layout                                                               #
    # ================================================================== #

    def _build_layout(self) -> None:
        """Set up the navigation container and show the Home screen."""
        self._view_container = ctk.CTkFrame(self, fg_color="gray30")
        self._view_container.pack(fill="both", expand=True)

        self._status_bar = ctk.CTkLabel(
            self, text="Ready", fg_color="gray30", text_color="lightgreen",
            font=("Arial", 10), corner_radius=8,
        )
        self._status_bar.pack(fill="x", padx=10, pady=(0, 5))

        self._active_view = None
        self.show_home()

    def _switch_view(self, build_view) -> None:
        """Destroy the current view and build a new one inside the container."""
        if self._active_view is not None:
            self._active_view.destroy()
            self._active_view = None
        self._active_view = build_view()
        self._active_view.pack(fill="both", expand=True)

    def show_home(self) -> None:
        """Stop active mode resources, then show the Home screen."""
        if self._state.camera_running:
            self._camera_ctrl._stop_camera()
        if self._state.listening:
            self._speech_ctrl._stop_listening()
        self._sign_ctrl.reset()

        from ui.views.home_view import HomeView
        self._switch_view(lambda: HomeView(
            self._view_container,
            on_sign_selected=self.show_sign_view,
            on_speech_selected=self.show_speech_view,
        ))
        self.update_status("Ready")

    def show_sign_view(self) -> None:
        from ui.views.sign_view import SignView

        def _build():
            view = SignView(self._view_container, on_back=self.show_home)
            view.set_camera_command(self._on_camera_clicked)
            return view

        self._switch_view(_build)
        self.update_status("Camera ready")
        if not self._state.camera_running:
            self._camera_ctrl._start_camera()

    def show_speech_view(self) -> None:
        from ui.views.speech_view import SpeechView

        def _build():
            view = SpeechView(self._view_container, on_back=self.show_home)
            view.set_speak_command(self._on_speak_clicked)
            return view

        self._switch_view(_build)
        self.update_status("Speech ready")
        # Camera is intentionally NOT started in speech mode.

    # ================================================================== #
    # UI event handlers — delegate immediately to controllers             #
    # ================================================================== #

    def _on_camera_clicked(self) -> None:
        try:
            self._camera_ctrl.on_camera_button_clicked()
        except Exception:
            logger.exception("Error in camera button handler")
            self.update_status("Camera error — see log")

    def _on_speak_clicked(self) -> None:
        try:
            self._speech_ctrl.on_speak_button_clicked()
        except Exception:
            logger.exception("Error in speak button handler")
            self.update_status("Speech error — see log")

    def _on_window_close(self) -> None:
        logger.info("Shutting down…")
        self.update_status("Shutting down…")
        try:
            self._camera_svc.cleanup()
            self._speech_svc.cleanup()
            self._video_svc.cleanup()
        except Exception:
            logger.exception("Error during shutdown cleanup")
        self.destroy()

    # ================================================================== #
    # View protocol — forward to the active per-mode view                 #
    # ================================================================== #

    def update_translation(self, text: str) -> None:
        try:
            view = self._active_view
            if view and hasattr(view, "update_translation"):
                view.update_translation(text)
        except Exception:
            logger.exception("update_translation error")

    def update_speech_status(self, status: str) -> None:
        try:
            view = self._active_view
            if view and hasattr(view, "update_speech_status"):
                view.update_speech_status(status)
        except Exception:
            logger.exception("update_speech_status error")

    def update_status(self, message: str) -> None:
        """Update the bottom status bar."""
        try:
            if self._status_bar.winfo_exists():
                self._status_bar.configure(text=message)
        except Exception:
            logger.exception("update_status error")

    def set_camera_button_text(self, text: str) -> None:
        try:
            view = self._active_view
            if view and hasattr(view, "set_camera_button_text"):
                view.set_camera_button_text(text)
        except Exception:
            logger.exception("set_camera_button_text error")

    def set_speak_button_text(self, text: str) -> None:
        try:
            view = self._active_view
            if view and hasattr(view, "set_speak_button_text"):
                view.set_speak_button_text(text)
        except Exception:
            logger.exception("set_speak_button_text error")

    def update_frame(self, frame, left_raw, right_raw, left_stable, right_stable) -> None:
        try:
            view = self._active_view
            if view and hasattr(view, "update_frame"):
                view.update_frame(frame, left_raw, right_raw, left_stable, right_stable)
        except Exception:
            logger.exception("update_frame error")

    def set_status(self, text: str) -> None:
        """Convenience alias used by CameraController for status updates."""
        self.update_status(text)
        try:
            view = self._active_view
            if view and hasattr(view, "set_status"):
                view.set_status(text)
        except Exception:
            logger.exception("set_status error")
