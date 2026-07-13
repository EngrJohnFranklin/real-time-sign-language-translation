"""
Speech Controller Module.

Handles all speech-related user events and routes data between the
SpeechService, VideoService, and the UI.

Responsibilities:
  - Toggle the microphone on / off in response to the Speak button.
  - Handle SpeechResult callbacks: update the translation display and
    trigger text-to-sign video playback for final results.
  - Update AppState to reflect the current speech status.

Knows about services and state — does NOT contain audio-engine logic
or CustomTkinter widget construction.
"""

import logging
from typing import Callable, Optional

from app.app_state import AppState
from services.speech_service import SpeechService
from services.video_service import VideoService
from translation.speech_handler import SpeechResult

logger = logging.getLogger(__name__)


class SpeechController:
    """
    Mediates between the Speak/Stop button, SpeechService, VideoService,
    and the main window view.

    The *view* is any object that exposes:
      - ``update_translation(text: str)``
      - ``update_speech_status(status: str)``
      - ``update_status(message: str)``
      - ``set_speak_button_text(text: str)``
      - ``after(delay_ms: int, callback)``  (standard Tk/CTk method)
    """

    def __init__(
        self,
        speech_service: SpeechService,
        video_service: VideoService,
        state: AppState,
        view,
    ) -> None:
        self._speech = speech_service
        self._video = video_service
        self._state = state
        self._view = view

    # ------------------------------------------------------------------ #
    # Public event handlers (called by the view)                          #
    # ------------------------------------------------------------------ #

    def on_speak_button_clicked(self) -> None:
        """Toggle speech recognition on or off."""
        if self._state.listening:
            self._stop_listening()
        else:
            self._start_listening()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _start_listening(self) -> None:
        self._state.set_listening(True)
        self._state.set_speech_status("Listening")
        self._view.update_speech_status("Listening")
        self._view.set_speak_button_text("⏹ Stop")
        self._view.update_status("Listening for speech…")

        def _on_result(result: SpeechResult) -> None:
            # This callback runs on the recognition background thread.
            # All UI interactions must be dispatched to the main thread.
            self._view.after(
                0, lambda r=result: self._handle_speech_result(r)
            )

        self._speech.start_listening(_on_result)
        logger.info("SpeechController: recognition started")

    def _stop_listening(self) -> None:
        self._speech.stop_listening()
        self._state.set_listening(False)
        self._state.set_speech_status("Ready")
        self._view.update_speech_status("Ready")
        self._view.set_speak_button_text("🎤 Speak")
        self._view.update_status("Speech recognition stopped")
        logger.info("SpeechController: recognition stopped")

    def _handle_speech_result(self, result: SpeechResult) -> None:
        """Process a speech recognition result on the main thread."""
        try:
            cleaned = " ".join(result.text.strip().split())
            if not cleaned:
                return

            if result.is_final:
                self._state.set_translation(cleaned)
                self._view.update_translation(cleaned)
                self._view.update_speech_status("Processing")
                self._play_as_signs(cleaned)
                # Return to Ready after animations are queued
                self._view.after(500, lambda: self._view.update_speech_status("Ready"))
            else:
                self._view.update_speech_status("Listening")

        except Exception:
            logger.exception("SpeechController._handle_speech_result error")

    def _play_as_signs(self, text: str) -> None:
        """Queue sign-language animations for the given text."""
        if not self._video.available:
            return
        if self._video.play_text(text):
            self._view.update_status(f"Playing sign animations for: {text[:40]}…")
