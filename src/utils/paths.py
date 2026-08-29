"""Project-root path resolution.

Resolves paths by walking upward to a marker file (pyproject.toml) instead of
a fixed .parent chain, so they stay correct regardless of which module calls
them or how deep the calling file lives.
"""

import pathlib
import sys
from typing import Optional

_MARKER_FILES = ("pyproject.toml", "setup.py", "requirements.txt")


def get_project_root(start: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Return the source root or the PyInstaller bundle resource directory."""
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys._MEIPASS)

    current = (start or pathlib.Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for _ in range(10):
        if any((current / marker).exists() for marker in _MARKER_FILES):
            return current
        if current.parent == current:
            break
        current = current.parent
    return pathlib.Path(__file__).resolve().parent.parent.parent


def get_model_path(filename: str = "sign_model.pkl") -> pathlib.Path:
    """Absolute path to <project_root>/data/models/<filename>."""
    return get_project_root() / "data" / "models" / filename
