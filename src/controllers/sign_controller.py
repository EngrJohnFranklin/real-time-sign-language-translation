"""
Sign Controller Module — Real-Time Translation Logic.

Handles the confirmed-sign → real-time translation → TTS pipeline.

Responsibilities:
  - Receive confirmed sign predictions from the camera controller
  - Track current sign and detect changes (using RealTimeTranslator)
  - Ignore duplicate detections while same sign is held
  - Update display with only the current sign (natural language meaning)
  - Trigger TTS only once per newly recognized sign
  - Automatically clear display after 3 seconds of inactivity
  - Update AppState with latest translation

Features:
  - Natural language output (e.g., "Hello" instead of "Shaka")
  - Single current sign display (not accumulated text)
  - Timeout-based auto-clear
  - Smart TTS (only once per sign, respects confidence threshold)
  - Hand side tracking

This controller knows nothing about camera hardware, audio engines,
or UI widgets beyond the minimal view protocol it depends on.
"""

import logging
from typing import Optional

from app.app_state import AppState
from services.speech_service import SpeechService
from translation.sign_to_text import SignToTextConverter
from translation.realtime_translator import RealTimeTranslator

logger = logging.getLogger(__name__)


class SignController:
    """
    Manages real-time sign → natural language translation with TTS.

    The *view* is any object that exposes:
      - ``update_translation(text: str)``
      - ``update_status(message: str)``  (optional)
      - ``after(ms, callback)`` for scheduling timeout checks
    """

    def __init__(
        self,
        sign_to_text: SignToTextConverter,
        speech_service: SpeechService,
        state: AppState,
        view,
    ) -> None:
        self._converter = sign_to_text  # Kept for backward compatibility
        self._speech = speech_service
        self._state = state
        self._view = view
        self._translator = RealTimeTranslator()
        self._timeout_task_id: Optional[str] = None  # ID for scheduled timeout check

    # ------------------------------------------------------------------ #
    # Public event handlers (called by CameraController via MainWindow)   #
    # ------------------------------------------------------------------ #

    def on_sign_recognized(
        self, sign_name: str, hand_side: str, confidence: float = 0.8
    ) -> None:
        """
        Handle a newly confirmed, stable sign prediction.

        Must be called on the **main (UI) thread** — CameraController
        ensures this via ``view.after(0, …)``.

        Args:
            sign_name:  Predicted sign label (e.g. "Thumbs Up").
            hand_side:  "Left" or "Right".
            confidence: Model confidence score (0.0–1.0).
        """
        try:
            # Update translator with new sign
            display_text, should_speak = self._translator.update(
                sign_name, hand_side, confidence
            )
            
            if display_text is None:
                # Duplicate sign - no change needed
                return
            
            # Update display with current sign's natural language meaning
            self._state.set_translation(display_text)
            self._view.update_translation(display_text)
            
            logger.info(
                f"Sign updated: {sign_name} → '{display_text}' "
                f"({hand_side}, {confidence:.0%})"
            )
            
            # Trigger TTS if this is a new sign with high enough confidence
            if should_speak and display_text:
                self._speak(display_text)
            
            # Schedule timeout check to clear display if no new sign for 3 seconds
            self._schedule_timeout_check()

        except Exception:
            logger.exception("SignController.on_sign_recognized error for '%s'", sign_name)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _speak(self, text: str) -> None:
        """Asynchronously speak the confirmed sign text."""
        normalized = " ".join(text.strip().split())
        if normalized:
            logger.info(f"Speaking: '{normalized}'")
            self._speech.speak_async(normalized)

    def _schedule_timeout_check(self) -> None:
        """Schedule a timeout check to clear display if no new sign detected."""
        # Cancel previous timeout if still pending
        if self._timeout_task_id:
            try:
                self._view.after_cancel(self._timeout_task_id)
            except (Exception, AttributeError):
                logger.debug("Could not cancel previous timeout task")
                pass
        
        # Schedule check in 500ms, then repeat every 500ms
        if hasattr(self._view, 'after'):
            self._timeout_task_id = self._view.after(500, self._check_timeout)
        else:
            logger.warning("View does not support scheduled callbacks (after method)")

    def _check_timeout(self) -> None:
        """Check if display should be cleared due to timeout."""
        try:
            timeout_result = self._translator.check_timeout()
            
            if timeout_result is not None:  # Timeout reached
                self._state.set_translation(timeout_result)
                self._view.update_translation(timeout_result)
                logger.info("Display cleared due to timeout")
                self._timeout_task_id = None
                return
            
            # Reschedule next check if view supports it
            if hasattr(self._view, 'after'):
                self._timeout_task_id = self._view.after(500, self._check_timeout)
            
        except Exception:
            logger.exception("SignController._check_timeout error")
            self._timeout_task_id = None
