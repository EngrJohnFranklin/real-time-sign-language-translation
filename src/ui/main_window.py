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

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure src/ is on the path when this module is imported standalone
# ---------------------------------------------------------------------------
_src_dir = os.path.join(os.path.dirname(__file__), "..")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from app.app_state import AppState
from models.sign_detector import SignRecognizer
from translation.speech_handler import SpeechHandler, SpeechLanguage
from translation.sign_to_text import SignToTextConverter
from services.camera_service import CameraService
from services.recognition_service import RecognitionService
from services.speech_service import SpeechService
from services.video_service import VideoService
from controllers.camera_controller import CameraController
from controllers.speech_controller import SpeechController
from controllers.sign_controller import SignController
from ui.panels.camera_panel import CameraPanel
from ui.video_player import VideoPlayerPanel

logger = logging.getLogger(__name__)

_VIDEO_DISPLAY_HEIGHT = 150  # pixels for the avatar / sign animation label


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

        # --- View widget references ---
        self._camera_panel: CameraPanel
        self._camera_button: ctk.CTkButton
        self._speak_button: ctk.CTkButton
        self._translation_display: ctk.CTkLabel
        self._speech_status_display: ctk.CTkLabel
        self._video_display_label: ctk.CTkLabel
        self._video_player_panel: VideoPlayerPanel
        self._status_bar: ctk.CTkLabel

        try:
            self._init_services()
            self._init_controllers()
            self._build_layout()
            self._init_video_player()
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

        self._camera_svc = CameraService(self._sign_recognizer)
        self._recognition_svc = RecognitionService()
        self._speech_svc = SpeechService(self._speech_handler)
        # VideoService is wired after the panel is created (_init_video_player)
        self._video_svc = VideoService(None)
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

    def _init_video_player(self) -> None:
        """
        Create VideoPlayerPanel inside a hidden container and wire the
        frame callback so animation frames appear in the avatar label.
        """
        video_folder = self._find_video_folder()
        if not video_folder:
            logger.warning("No videos folder found — speech-to-sign unavailable")
            return

        try:
            hidden = ctk.CTkFrame(self)
            self._video_player_panel = VideoPlayerPanel(
                hidden, video_folder=video_folder
            )
            self._video_player_panel.video_player.set_frame_callback(
                self._display_video_frame
            )
            # Re-wire the video service with the real panel
            self._video_svc = VideoService(self._video_player_panel)
            # Propagate to speech controller
            self._speech_ctrl._video = self._video_svc
            logger.info("VideoPlayerPanel initialised: %s", video_folder)
        except Exception:
            logger.exception("Could not initialise VideoPlayerPanel")

    @staticmethod
    def _find_video_folder() -> str:
        """Return the path to the videos directory or empty string."""
        candidates = [
            "videos",
            os.path.join(os.path.dirname(__file__), "..", "..", "videos"),
            os.path.join(os.path.dirname(__file__), "..", "videos"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""

    # ================================================================== #
    # Layout                                                               #
    # ================================================================== #

    def _build_layout(self) -> None:
        """Construct the full 2-column responsive grid layout."""
        container = ctk.CTkFrame(self, fg_color="gray30")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        container.grid_rowconfigure(0, weight=3)
        container.grid_rowconfigure(1, weight=1)
        container.grid_rowconfigure(2, weight=0)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        # ---- Row 0: Camera & Avatar ----
        self._build_camera_cell(container, row=0, col=0)
        self._build_avatar_cell(container, row=0, col=1)

        # ---- Row 1: Translation & Speech Status ----
        self._build_translation_cell(container, row=1, col=0)
        self._build_speech_status_cell(container, row=1, col=1)

        # ---- Row 2: Buttons ----
        self._build_button_row(container, row=2)

        # ---- Status bar ----
        self._status_bar = ctk.CTkLabel(
            self,
            text="Ready",
            fg_color="gray30",
            text_color="lightgreen",
            font=("Arial", 10),
            corner_radius=8,
        )
        self._status_bar.pack(fill="x", padx=10, pady=(0, 5))

    def _build_camera_cell(self, parent, row: int, col: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=12)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(
            frame,
            text="📷 Live Camera",
            font=("Arial", 12, "bold"),
            text_color="lightblue",
        ).pack(pady=8)

        self._camera_panel = CameraPanel(frame)
        self._camera_panel.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_avatar_cell(self, parent, row: int, col: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=12)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(
            frame,
            text="🎬 Sign Language Avatar",
            font=("Arial", 12, "bold"),
            text_color="lightyellow",
        ).pack(pady=8)

        self._video_display_label = ctk.CTkLabel(
            frame,
            text="[Animation will display here]",
            fg_color="gray10",
            text_color="gray60",
            font=("Arial", 11),
            corner_radius=8,
        )
        self._video_display_label.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_translation_cell(self, parent, row: int, col: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=12)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(
            frame,
            text="📝 Current Translation",
            font=("Arial", 11, "bold"),
            text_color="lightgreen",
        ).pack(pady=5)

        self._translation_display = ctk.CTkLabel(
            frame,
            text="",
            font=("Courier", 13, "bold"),
            text_color="white",
            fg_color="gray10",
            corner_radius=8,
            wraplength=280,
        )
        self._translation_display.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_speech_status_cell(self, parent, row: int, col: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color="gray20", corner_radius=12)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(
            frame,
            text="📊 Speech Status",
            font=("Arial", 11, "bold"),
            text_color="lightblue",
        ).pack(pady=5)

        self._speech_status_display = ctk.CTkLabel(
            frame,
            text="🟢 Ready",
            font=("Arial", 16, "bold"),
            text_color="lightgreen",
            fg_color="gray10",
            corner_radius=8,
        )
        self._speech_status_display.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_button_row(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color="gray30")
        frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        self._camera_button = ctk.CTkButton(
            frame,
            text="▶ Start Camera",
            command=self._on_camera_clicked,
            font=("Arial", 12, "bold"),
            height=45,
            corner_radius=10,
            fg_color="#1f6aa5",
        )
        self._camera_button.grid(row=0, column=0, sticky="ew", padx=5)

        self._speak_button = ctk.CTkButton(
            frame,
            text="🎤 Speak",
            command=self._on_speak_clicked,
            font=("Arial", 12, "bold"),
            height=45,
            corner_radius=10,
            fg_color="#1f6aa5",
        )
        self._speak_button.grid(row=0, column=1, sticky="ew", padx=5)

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
    # View protocol — called by controllers to update widgets             #
    # ================================================================== #

    def update_translation(self, text: str) -> None:
        """Display *text* in the Current Translation label."""
        try:
            if self._translation_display.winfo_exists():
                self._translation_display.configure(text=text)
        except Exception:
            logger.exception("update_translation error")

    def update_speech_status(self, status: str) -> None:
        """Update the Speech Status label with emoji and colour."""
        _config = {
            "Ready":      ("🟢", "lightgreen"),
            "Listening":  ("🔴", "orange"),
            "Processing": ("🟠", "#FFD700"),
            "Speaking":   ("🔵", "lightyellow"),
        }
        emoji, color = _config.get(status, ("🟢", "lightgreen"))
        try:
            if self._speech_status_display.winfo_exists():
                self._speech_status_display.configure(
                    text=f"{emoji} {status}", text_color=color
                )
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
        """Update the camera toggle button label."""
        try:
            self._camera_button.configure(text=text)
        except Exception:
            logger.exception("set_camera_button_text error")

    def set_speak_button_text(self, text: str) -> None:
        """Update the speak toggle button label."""
        try:
            self._speak_button.configure(text=text)
        except Exception:
            logger.exception("set_speak_button_text error")

    # CameraPanel view protocol (forwarded from CameraController)
    def update_frame(self, frame, left_raw, right_raw, left_stable, right_stable) -> None:
        """Forward frame update to the camera panel."""
        self._camera_panel.update_frame(frame, left_raw, right_raw, left_stable, right_stable)

    def set_status(self, text: str) -> None:
        """Convenience alias used by CameraController for status updates."""
        self.update_status(text)
        self._camera_panel.set_status(text)

    # ================================================================== #
    # Video frame display                                                  #
    # ================================================================== #

    def _display_video_frame(self, frame: np.ndarray) -> None:
        """Receive a video frame from the playback thread and schedule UI update."""
        if frame is None or frame.size == 0:
            return
        try:
            if not self._video_display_label.winfo_exists():
                return
            h, w = frame.shape[:2]
            if h == 0:
                return
            display_w = int(_VIDEO_DISPLAY_HEIGHT * w / h)
            resized = cv2.resize(frame, (display_w, _VIDEO_DISPLAY_HEIGHT))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            ctk_img = ctk.CTkImage(
                light_image=Image.fromarray(rgb),
                size=(display_w, _VIDEO_DISPLAY_HEIGHT),
            )
            self.after(
                0,
                lambda img=ctk_img: self._apply_video_frame(img),
            )
        except Exception:
            logger.exception("_display_video_frame error")

    def _apply_video_frame(self, ctk_image) -> None:
        """Apply a prepared CTkImage to the video label (main thread only)."""
        try:
            if self._video_display_label.winfo_exists():
                self._video_display_label.configure(image=ctk_image, text="")
                self._video_display_label.image = ctk_image
        except Exception:
            logger.exception("_apply_video_frame error")
