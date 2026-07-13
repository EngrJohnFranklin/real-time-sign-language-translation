"""
Controllers package.

Provides the event-handler / presenter layer that sits between the
UI views and the domain services.

Each controller owns exactly one slice of the user workflow:

- CameraController  — camera toggle, frame routing, sign callbacks
- SpeechController  — speech listen/stop, speech-result routing
- SignController    — confirmed-sign → text → TTS pipeline
"""

from controllers.camera_controller import CameraController
from controllers.speech_controller import SpeechController
from controllers.sign_controller import SignController

__all__ = ["CameraController", "SpeechController", "SignController"]
