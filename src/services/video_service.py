"""
Video Service Module.

Thin facade over VideoPlayerPanel that exposes a minimal, stable API
to controllers without coupling them to the CustomTkinter widget internals.

Responsibilities:
  - Convert spoken / recognized text into a sequence of sign-language
    animation clips and enqueue them for playback.
  - Register the per-frame display callback that sends frames to the
    UI video label.
  - Delegate all playback queue management to VideoPlayerPanel.

All video engine complexity lives in ui/video_player.py.
"""

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VideoService:
    """
    Application-level facade for text-to-sign video playback.

    Wraps :class:`~ui.video_player.VideoPlayerPanel` so that controllers
    interact with a clean domain API rather than CustomTkinter widget methods.

    The *frame_callback* receives raw OpenCV frames from the playback thread
    and is responsible for scheduling UI updates on the main thread.

    Usage::

        service = VideoService(player_panel, on_frame)
        service.play_text("hello world")
        service.cleanup()
    """

    def __init__(
        self,
        player_panel,  # ui.video_player.VideoPlayerPanel — typed loosely to avoid circular import
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self._panel = player_panel
        if frame_callback and player_panel is not None:
            try:
                player_panel.video_player.set_frame_callback(frame_callback)
            except Exception:
                logger.exception("VideoService: failed to register frame callback")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        """True when the underlying VideoPlayerPanel was successfully created."""
        return self._panel is not None

    def play_text(self, text: str) -> bool:
        """
        Queue sign-language animation clips for *text* and start playback.

        Words are resolved to whole-word clips first; individual characters
        fall back to alphabet clips.

        Args:
            text: Free-form text (words and/or letters).

        Returns:
            ``True`` when at least one clip was queued, ``False`` otherwise.
        """
        if not self._panel or not text:
            return False
        try:
            return self._panel.play_text_as_signs(text)
        except Exception:
            logger.exception("VideoService.play_text failed for '%s'", text[:40])
            return False

    def cleanup(self) -> None:
        """Release video resources."""
        if not self._panel:
            return
        try:
            self._panel.cleanup()
        except Exception:
            logger.exception("VideoService cleanup error")
