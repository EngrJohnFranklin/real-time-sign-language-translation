"""
Speech Service Module.

Thin facade over SpeechHandler that exposes a minimal, stable API
to controllers without coupling them to the internal Vosk / pyttsx3
implementation details.

Responsibilities:
  - Start / stop speech recognition (STT).
  - Speak text asynchronously (TTS).
  - Delegate lifecycle management to the underlying SpeechHandler.

All audio engine complexity lives in translation/speech_handler.py.
"""

import logging
from typing import Callable, Optional

from translation.speech_handler import SpeechHandler, SpeechResult, SpeechLanguage

logger = logging.getLogger(__name__)


class SpeechService:
    """
    Application-level facade for speech input and output.

    Wraps :class:`~translation.speech_handler.SpeechHandler` so that
    controllers interact with a clean domain API rather than the
    audio-engine internals.

    Usage::

        service = SpeechService(handler)
        service.start_listening(callback)
        service.speak_async("Hello")
        service.stop_listening()
        service.cleanup()
    """

    def __init__(self, handler: SpeechHandler) -> None:
        self._handler = handler
        self._speech_sequence = 0  # Track latest requested speech to prevent stale playback

    # ------------------------------------------------------------------ #
    # Speech-to-Text                                                       #
    # ------------------------------------------------------------------ #

    def start_listening(
        self, callback: Callable[[SpeechResult], None]
    ) -> None:
        """
        Start continuous speech recognition.

        Args:
            callback: Called from the recognition background thread for every
                :class:`~translation.speech_handler.SpeechResult`.
                UI callers MUST schedule widget updates on the main thread.
        """
        try:
            self._handler.listen_and_recognize(callback=callback)
            logger.debug("SpeechService: recognition started")
        except Exception:
            logger.exception("SpeechService failed to start listening")

    def stop_listening(self) -> None:
        """Stop the speech recognition loop."""
        try:
            self._handler.stop_listening()
            logger.debug("SpeechService: recognition stopped")
        except Exception:
            logger.exception("SpeechService failed to stop listening")

    # ------------------------------------------------------------------ #
    # Text-to-Speech                                                       #
    # ------------------------------------------------------------------ #

    def speak_async(self, text: str) -> None:
        """
        Synthesize and speak *text* in a background thread.

        If a newer speak_async() call arrives before the previous one starts,
        the older speech is skipped to keep TTS in sync with the latest
        confirmed recognition.

        Args:
            text: The text to convert to speech.
        """
        if not text or not text.strip():
            return
        
        # Increment sequence number — old pending threads will skip if outdated
        self._speech_sequence += 1
        current_sequence = self._speech_sequence
        
        try:
            self._handler.speak_async(text, sequence_id=current_sequence, 
                                     is_current=lambda: current_sequence == self._speech_sequence)
        except Exception:
            logger.exception("SpeechService failed to speak '%s'", text[:40])

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        """Release all audio resources."""
        try:
            self._handler.cleanup()
        except Exception:
            logger.exception("SpeechService cleanup error")
