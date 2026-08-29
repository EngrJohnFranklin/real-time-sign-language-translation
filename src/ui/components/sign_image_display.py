"""Reusable static sign-reference image display widget."""

import logging
import threading
import tkinter as tk
from pathlib import Path
from typing import Dict, Optional, Tuple

import customtkinter as ctk
from PIL import Image

from utils.paths import get_project_root

logger = logging.getLogger(__name__)


class SignImageDisplay(ctk.CTkFrame):
    """Display the matching static sign image or a friendly placeholder."""

    def __init__(
        self,
        parent,
        word: str = "",
        max_size: Tuple[int, int] = (300, 300),
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._max_size = max_size
        self._image: Optional[ctk.CTkImage] = None
        # Cache CTkImage per resolved path so each underlying Tcl "pyimageN"
        # is registered exactly once and reused for the widget's lifetime.
        # Recreating a CTkImage for a word the label is already showing (then
        # dropping the old reference mid-render) is what triggered
        # `image "pyimageN" doesn't exist`.
        self._image_cache: Dict[str, ctk.CTkImage] = {}
        self._image_dir = get_project_root() / "assets" / "sign_images"
        self._label = ctk.CTkLabel(self, text="")
        self._label.pack(fill="both", expand=True, padx=8, pady=8)
        self.update_word(word)

    @staticmethod
    def _filename_for_word(word: str) -> str:
        normalized = " ".join((word or "").strip().lower().split())
        return f"{normalized.replace(' ', '_')}.png"

    def _find_image_path(self, word: str) -> Optional[Path]:
        phrase_path = self._image_dir / self._filename_for_word(word)
        if phrase_path.exists():
            return phrase_path

        # For multi-word phrases (e.g., "i love you", "thank you"),
        # do NOT silently fall back to the first word. Return None to signal
        # to the caller that the full phrase image is missing.
        # Single words (e.g. "hello") will just not find anything anyway.
        if " " in (word or "").strip():
            return None

        # Single word: try individual character fallback only if needed
        for token in (word or "").split():
            token_path = self._image_dir / self._filename_for_word(token)
            if token_path.exists():
                return token_path

        return None

    def update_word(self, word: str) -> None:
        """Show an image for *word*, falling back to a placeholder.

        Thread-safe: Tk/CustomTkinter is strictly single-threaded, so when
        this is invoked from a worker thread (e.g. the Vosk recognition
        thread via ``update_translation``) the update is re-dispatched to
        the main GUI thread with ``after()`` instead of touching Tcl
        directly — otherwise the label can reference an image name that
        was never registered (``image "pyimageN" doesn't exist``).
        """
        if threading.current_thread() is not threading.main_thread():
            try:
                self.after(0, lambda w=word: self.update_word(w))
            except (tk.TclError, RuntimeError):
                # Window already destroyed — nothing sensible to do.
                logger.debug(
                    "SignImageDisplay: dropped off-thread update for '%s'", word
                )
            return

        if not self.winfo_exists():
            return

        image_path = self._find_image_path(word)
        if image_path is None:
            self._show_placeholder()
            if word and " " in word.strip():
                # Multi-word phrase with no image file — log clearly
                logger.warning(
                    "No sign image found for phrase '%s' in %s",
                    word, self._image_dir
                )
            elif word:
                logger.warning(
                    "Image lookup MISS for word '%s' in %s",
                    word,
                    self._image_dir,
                )
            return

        logger.info("Image lookup HIT for '%s' -> %s", word, image_path)
        cache_key = str(image_path)
        try:
            # Reuse a cached, already-registered image if we have shown this
            # file before — avoids destroying/re-registering the Tcl image
            # while the label may still be rendering it.
            new_image = self._image_cache.get(cache_key)
            if new_image is None:
                with Image.open(image_path) as source_image:
                    image = source_image.convert("RGBA")
                image.thumbnail(self._max_size, Image.Resampling.LANCZOS)
                new_image = ctk.CTkImage(
                    light_image=image,
                    dark_image=image,
                    size=image.size,
                )
                self._image_cache[cache_key] = new_image
            # Do NOT drop the previous reference here: the label may still be
            # displaying the old image. Cached images stay alive for the
            # widget's lifetime, so their "pyimageN" never becomes invalid
            # mid-render. self._image is updated only for the placeholder path.
            self._apply_image(new_image)
        except (OSError, ValueError) as error:
            self._show_placeholder()
            logger.warning("Could not load sign image %s: %s", image_path, error)

    def clear(self) -> None:
        """Clear the currently displayed sign image."""
        self._show_placeholder()

    def _apply_image(self, image: ctk.CTkImage) -> None:
        """Configure the label with *image*, degrading gracefully on Tcl errors."""
        try:
            self._label.configure(image=image, text="")
            self._image = image
        except tk.TclError as error:
            # The Tcl image is gone/invalid (e.g. interpreter teardown or a
            # stale reference) — show text instead of crashing the app.
            logger.warning("Sign image no longer valid (%s); using text fallback", error)
            self._image = None
            self._show_placeholder()

    def _show_placeholder(self) -> None:
        """Clear the image and show the text placeholder without ever raising."""
        try:
            if self._label.winfo_exists():
                # Clear the label's image FIRST; only then drop our reference,
                # so the label never renders a name that was just deleted.
                self._label.configure(image=None, text="No sign image available")
                self._image = None
        except tk.TclError:
            # Widget/interpreter already destroyed (shutdown race) — ignore.
            logger.debug("SignImageDisplay: placeholder update skipped (widget gone)")