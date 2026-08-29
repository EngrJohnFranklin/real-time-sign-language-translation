"""
UI Tests
Unit tests for GUI components and UI widgets
"""

import unittest
from unittest.mock import MagicMock, patch
import customtkinter as ctk

from ui.views.home_view import HomeView
from ui.views.sign_view import SignView
from ui.views.speech_view import SpeechView


class TestHomeView(unittest.TestCase):
    """Unit tests for the redesigned HomeView screen."""

    @classmethod
    def setUpClass(cls):
        ctk.set_appearance_mode("dark")
        cls.root = ctk.CTk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def test_home_view_init_and_callbacks(self):
        """Verify that HomeView initializes and invokes callbacks correctly."""
        on_sign_mock = MagicMock()
        on_speech_mock = MagicMock()

        view = HomeView(
            self.root,
            on_sign_selected=on_sign_mock,
            on_speech_selected=on_speech_mock,
        )

        self.assertIsNotNone(view._sign_button)
        self.assertIsNotNone(view._speech_button)

        # Trigger sign language button command
        view._sign_button._command()
        on_sign_mock.assert_called_once()

        # Trigger speech button command
        view._speech_button._command()
        on_speech_mock.assert_called_once()

        view.destroy()


class TestMainWindowNavigation(unittest.TestCase):
    """Verify MainWindow navigation between HomeView, SignView, and SpeechView."""

    @classmethod
    def setUpClass(cls):
        ctk.set_appearance_mode("dark")

    @patch("ui.main_window.CameraService")
    @patch("ui.main_window.SpeechService")
    @patch("ui.main_window.RecognitionService")
    @patch("ui.main_window.SignRecognizer")
    @patch("ui.main_window.WordRecognizer")
    @patch("ui.main_window.SpeechHandler")
    @patch("ui.main_window.SignToTextConverter")
    def test_navigation_triggers(
        self,
        mock_sign_to_text,
        mock_speech_handler,
        mock_word_recognizer,
        mock_sign_recognizer,
        mock_rec_svc,
        mock_speech_svc,
        mock_camera_svc,
    ):
        from ui.main_window import MainWindow

        win = MainWindow()
        win.withdraw()

        # Initially on HomeView
        self.assertIsInstance(win._active_view, HomeView)

        # Trigger Sign View
        win.show_sign_view()
        self.assertIsInstance(win._active_view, SignView)

        # Return to Home View
        win.show_home()
        self.assertIsInstance(win._active_view, HomeView)

        # Trigger Speech View
        win.show_speech_view()
        self.assertIsInstance(win._active_view, SpeechView)

        # Return to Home View
        win.show_home()
        self.assertIsInstance(win._active_view, HomeView)

        win.destroy()


if __name__ == "__main__":
    unittest.main()


