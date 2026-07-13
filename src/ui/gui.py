"""
Backward-Compatibility Shim for ui.gui.

All public symbols previously defined in this module are now maintained
in their canonical homes:

  MainWindow  -> ui.main_window.MainWindow
  CameraPanel -> ui.panels.camera_panel.CameraPanel

Importing from ``ui.gui`` continues to work unchanged so that external
code and the entry-point (``src/main.py``) require no modification.
"""

# Re-export canonical implementations
from ui.main_window import MainWindow          # noqa: F401  (public re-export)
from ui.panels.camera_panel import CameraPanel  # noqa: F401  (public re-export)

__all__ = ["MainWindow", "CameraPanel"]
