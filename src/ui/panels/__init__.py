"""
UI Panels sub-package.

Contains self-contained view panels that own only rendering logic
and delegate all business logic to the controller layer.
"""

from ui.panels.camera_panel import CameraPanel

__all__ = ["CameraPanel"]
