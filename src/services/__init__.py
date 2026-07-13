"""
Services package.

Provides domain-focused service classes extracted from the original GUI God Class.
Each service owns exactly one infrastructure concern:

- CameraService      — OpenCV camera capture + MediaPipe sign detection
- RecognitionService — Temporal smoothing for stable sign predictions
- SpeechService      — Facade over SpeechHandler (STT + TTS)
- VideoService       — Facade over VideoPlayerPanel (text-to-sign animation)
"""

from services.camera_service import CameraService
from services.recognition_service import RecognitionService
from services.speech_service import SpeechService
from services.video_service import VideoService

__all__ = [
    "CameraService",
    "RecognitionService",
    "SpeechService",
    "VideoService",
]
