"""Camera device selection helper.

Shared by every entry point that opens a webcam (CameraService,
collect_training_data.py, verify_gesture_landmarks.py, etc.) so camera
selection behaves the same way everywhere via a CustomTkinter dialog,
consistent with the rest of the app's UI.
"""

import logging
from typing import List, Optional

import cv2
import customtkinter as ctk

logger = logging.getLogger(__name__)

MAX_CAMERA_INDEX_TO_PROBE = 5


def detect_available_cameras(max_index: int = MAX_CAMERA_INDEX_TO_PROBE) -> List[int]:
    """Return indices that successfully open with cv2.VideoCapture."""
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available.append(index)
        cap.release()
    return available


class _CameraSelectionDialog(ctk.CTkToplevel):
    """Modal dialog listing detected cameras; blocks until the user confirms."""

    def __init__(self, master, camera_indices: List[int]) -> None:
        super().__init__(master)
        self.title("Select Camera")
        self.geometry("320x180")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        self._options = [f"Camera {i}" for i in camera_indices]
        self._indices = camera_indices
        self.selected_index = camera_indices[0]

        ctk.CTkLabel(
            self, text="Multiple cameras detected.\nChoose one to use:"
        ).pack(padx=20, pady=(20, 10))

        self._menu = ctk.CTkOptionMenu(
            self, values=self._options, command=self._on_select
        )
        self._menu.set(self._options[0])
        self._menu.pack(padx=20, pady=10, fill="x")

        ctk.CTkButton(self, text="Start", command=self._on_confirm).pack(
            padx=20, pady=(10, 20)
        )

        self.protocol("WM_DELETE_WINDOW", self._on_confirm)
        self.transient(master)
        self.grab_set()
        self.after(50, self.focus_force)

    def _on_select(self, choice: str) -> None:
        self.selected_index = self._indices[self._options.index(choice)]

    def _on_confirm(self) -> None:
        self.grab_release()
        self.destroy()


class _NoCameraDialog(ctk.CTkToplevel):
    """Modal error dialog shown when no cameras are detected."""

    def __init__(self, master) -> None:
        super().__init__(master)
        self.title("No Camera Found")
        self.geometry("320x150")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        ctk.CTkLabel(
            self,
            text="No camera devices were detected.\nConnect a webcam and try again.",
            wraplength=280,
        ).pack(padx=20, pady=(20, 10))

        ctk.CTkButton(self, text="OK", command=self._on_ok).pack(pady=(10, 20))

        self.transient(master)
        self.grab_set()
        self.after(50, self.focus_force)

    def _on_ok(self) -> None:
        self.grab_release()
        self.destroy()


def select_camera_index(
    parent: Optional[ctk.CTk] = None,
    max_index: int = MAX_CAMERA_INDEX_TO_PROBE,
) -> Optional[int]:
    """
    Detect cameras and return the index to use via a CustomTkinter dialog.

    Args:
        parent: Existing CTk/CTkToplevel root to host the dialog as a child.
            If None, a temporary hidden CTk root is created and destroyed
            afterward (for non-GUI callers such as standalone scripts).
        max_index: Highest camera index (exclusive) to probe.

    Returns:
        The chosen camera index, or None if no camera was detected.
    """
    available = detect_available_cameras(max_index)

    owns_root = parent is None
    root = ctk.CTk() if owns_root else parent
    if owns_root:
        root.withdraw()

    try:
        if not available:
            dialog = _NoCameraDialog(root)
            root.wait_window(dialog)
            return None

        if len(available) == 1:
            logger.info(f"Using camera {available[0]} (only one detected)")
            return available[0]

        dialog = _CameraSelectionDialog(root, available)
        root.wait_window(dialog)
        return dialog.selected_index
    finally:
        if owns_root:
            root.destroy()
