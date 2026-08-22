"""Focused tests for word-level recognition integration."""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_template_loader_excludes_recording_outliers(tmp_path, monkeypatch):
    from translation import word_template_loader as loader

    hello_dir = tmp_path / "hello"
    hello_dir.mkdir()
    np.save(hello_dir / "sample_01.npy", np.zeros((10, 126)))
    np.save(hello_dir / "sample_02.npy", np.ones((10, 126)))
    monkeypatch.setattr(loader, "WORD_TEMPLATES_DIR", tmp_path)

    manager = loader.WordTemplateManager()
    assert manager.load_all_templates()
    assert len(manager.get_templates_for_word("hello")) == 1


def test_template_loader_uses_project_data_directory():
    from translation.word_template_loader import WORD_TEMPLATES_DIR

    assert WORD_TEMPLATES_DIR == PROJECT_ROOT / "data" / "word_templates"


def test_translator_displays_and_speaks_word_labels():
    from translation.realtime_translator import RealTimeTranslator

    display_text, should_speak = RealTimeTranslator().update("thank_you", "Both", 0.9)

    assert display_text == "Thank You"
    assert should_speak is True


def test_word_recognition_waits_three_seconds_then_tries_motion_first(monkeypatch):
    from controllers import camera_controller

    clock = [100.0]
    monkeypatch.setattr(camera_controller.time, "monotonic", lambda: clock[0])
    CameraController = camera_controller.CameraController

    class FakeWordRecognizer:
        def __init__(self):
            self.calls = []

        def recognize_static_from_frame(self, _features):
            self.calls.append("static")
            return "love", 0.99

        def recognize_motion_from_sequence(self, _sequence):
            self.calls.append("motion")
            return None, 0.0

    recognizer = FakeWordRecognizer()

    controller = CameraController(
        camera_service=None,
        recognition_service=None,
        state=None,
        view=None,
        word_recognizer=recognizer,
    )

    for _ in range(10):
        assert controller._advance_word_session(np.zeros(126, dtype=np.float32)) is None
    assert recognizer.calls == []

    clock[0] += controller.DETECTION_WINDOW_SECONDS
    assert controller._advance_word_session(np.zeros(126, dtype=np.float32)) == ("love", 0.99)
    assert recognizer.calls == ["motion", "static"]


def test_letter_callbacks_are_disabled_in_words_only_mode():
    from controllers.camera_controller import CameraController

    class FakeRecognitionService:
        def update(self, _left, _right):
            return "left", True, "right", True

    class FakeView:
        def __init__(self):
            self.callbacks = []

        def after(self, _delay, callback):
            self.callbacks.append(callback)

        def update_frame(self, *_args):
            pass

    callback_labels = []
    view = FakeView()
    controller = CameraController(
        camera_service=None,
        recognition_service=FakeRecognitionService(),
        state=None,
        view=view,
        sign_recognized_callback=lambda label, *_args: callback_labels.append(label),
    )

    controller._on_frame_received(np.zeros((1, 1, 3)), object(), object(), None)
    for callback in view.callbacks:
        callback()

    assert callback_labels == []


def test_words_only_mode_skips_letter_classifier(monkeypatch):
    from models import sign_detector

    class FailingClassifier:
        def predict_both_hands(self, *_args):
            raise AssertionError("Letter classifier must not run in words-only mode")

    class FakeHands:
        def process(self, _frame):
            landmarks = [type("Landmark", (), {"x": 0.1 + index * 0.01, "y": 0.2, "z": 0.0})()
                         for index in range(21)]
            hand = type("Hand", (), {"landmark": landmarks})()
            handedness = type(
                "Handedness",
                (),
                {"classification": [type("Classification", (), {"label": "Left"})()]},
            )()
            return type(
                "Results",
                (),
                {"multi_hand_landmarks": [hand], "multi_handedness": [handedness]},
            )()

    monkeypatch.setattr(sign_detector.cv2, "cvtColor", lambda frame, _code: frame)
    recognizer = sign_detector.SignRecognizer.__new__(sign_detector.SignRecognizer)
    recognizer.hands = FakeHands()
    recognizer.xgboost_classifier = FailingClassifier()

    left, right, features = recognizer.process_frame_with_features(
        np.ones((2, 2, 3), dtype=np.uint8), enable_letter_recognition=False
    )

    assert left is not None and left.sign_type == sign_detector.SignType.UNKNOWN
    assert left.landmarks and len(left.landmarks) == 21
    assert right is None
    assert features.shape == (126,)