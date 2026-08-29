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
import time

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
        self._last_final_text = ""
        self._last_final_time = 0.0

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

    def _monitor_cooldown(self) -> None:
        """Continuously monitor cooldown state and update status."""
        if not self._state.listening:
            return

        # Check if still in cooldown
        remaining = self._speech.get_cooldown_remaining()
        if remaining > 0.0:
            status_text = f"Please wait… ({remaining:.1f}s)"
            self._view.update_speech_status(status_text)
            # Check again after 200ms
            self._view.after(200, self._monitor_cooldown)
        else:
            # Cooldown is over, back to listening
            if self._state.listening:
                self._view.update_speech_status("Listening")

    def _stop_listening(self) -> None:
        self._speech.stop_listening()
        self._state.set_listening(False)
        self._state.set_recognition_state("ready")
        self._state.set_speech_status("Ready")
        self._view.update_speech_status("Ready")
        self._view.set_speak_button_text("🎤 Speak")
        self._view.update_status("Speech recognition stopped")
        logger.info("SpeechController: recognition stopped")

    def _handle_speech_result(self, result: SpeechResult) -> None:
        """Process a speech recognition result on the main thread.

        The recognizer (``VoskSpeechRecognizer``) already enforces a strict
        one-word-then-cooldown cycle upstream, so a final result here is a
        fully finalized single word — commit it immediately, no debounce.
        """
        try:
            cleaned = " ".join(result.text.strip().split())
            if not cleaned:
                return

            if result.is_final:
                if self._state.snapshot()["recognition_state"] == "result_active":
                    return

                # Check for duplicate recognition
                now = time.time()
                if (
                    cleaned.lower() == self._last_final_text.lower()
                    and (now - self._last_final_time) < 2.0
                ):
                    # Same word within 2 seconds — likely the word is still being
                    # processed for video playback, show feedback to user
                    self._view.update_speech_status("Already recognized")
                    self._view.after(1000, lambda: self._view.update_speech_status("Processing"))
                    return

                self._last_final_text = cleaned
                self._last_final_time = now
                self._state.set_recognition_state("result_active")
                self._state.set_translation(cleaned)
                self._view.update_translation(cleaned)

                self._view.update_speech_status("Processing")
                self._play_as_signs(cleaned)
                self._view.after(3000, self._finish_result_cycle)
            else:
                self._view.update_speech_status("Listening")

        except Exception:
            logger.exception("SpeechController._handle_speech_result error")

    def _finish_result_cycle(self) -> None:
        """Clear the accepted result and reopen recognition as one transition."""
        if not self._state.listening:
            return
        self._speech.release_cooldown()
        self._state.set_translation("")
        self._state.set_recognition_state("ready")
        self._state.set_speech_status("Ready")
        self._view.update_translation("")
        self._view.clear_sign_image()
        self._view.update_speech_status("Ready")

    def _play_as_signs(self, text: str) -> None:
        """Queue sign-language animations for the given text."""
        if not self._video.available:
            return
        if self._video.play_text(text):
            self._view.update_status(f"Playing sign animations for: {text[:40]}…")
