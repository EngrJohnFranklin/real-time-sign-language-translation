"""
Main GUI Application for Real-Time Sign Language Translation.

This module provides the main user interface using CustomTkinter with a dark modern theme.
Integrates sign detection, speech recognition, and text-to-speech modules.

Simplified Layout:
- Left panel: Live webcam feed with hand landmarks
- Right top: Recognized sign text
- Right bottom: Speech-to-text output
- Control buttons: Start Camera, Speak
"""

import os
import sys
import threading
import logging
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

# Add src directory to path for module imports
src_dir = os.path.join(os.path.dirname(__file__), '..')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import application modules
from models.sign_detector import SignRecognizer, SignType
from translation.speech_handler import SpeechHandler, SpeechResult, SpeechLanguage
from translation.sign_to_text import SignToTextConverter
from ui.video_player import VideoPlayerPanel

logger = logging.getLogger(__name__)


class CameraPanel(ctk.CTkFrame):
    """
    Left panel displaying live webcam feed with hand landmark overlay.
    
    Handles camera initialization, frame capture, and hand landmark visualization.
    Runs camera capture in background thread to prevent UI blocking.
    """

    CONFIDENCE_THRESHOLD = 0.75
    STABILITY_FRAMES = 4
    LABEL_HOLD_FRAMES = 6
    
    def __init__(self, parent, sign_recognizer: SignRecognizer, **kwargs):
        """
        Initialize camera panel.
        
        Args:
            parent: Parent CustomTkinter widget.
            sign_recognizer: SignRecognizer instance for hand detection.
            **kwargs: Additional arguments for CTkFrame.
        """
        super().__init__(parent, **kwargs)
        
        self.sign_recognizer = sign_recognizer
        self.camera = None
        self.is_running = False
        self.camera_thread = None
        self.thread_lock = threading.Lock()  # Prevent thread race conditions
        self.current_frame = None
        self.frame_label = None
        self.status_label = None
        self.left_hand_result = None
        self.right_hand_result = None
        self.recognized_sign_callback = None
        self.left_prediction_history = deque(maxlen=self.STABILITY_FRAMES)
        self.right_prediction_history = deque(maxlen=self.STABILITY_FRAMES)
        self.left_stable_result = None
        self.right_stable_result = None
        self.left_hold_counter = 0
        self.right_hold_counter = 0
        
        try:
            self._create_ui()
            logger.info("CameraPanel initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing CameraPanel: {e}")
            raise
    
    def _create_ui(self):
        """Create UI elements for camera panel."""
        try:
            # Title label
            title = ctk.CTkLabel(self, text="📷 Live Camera Feed", font=("Arial", 16, "bold"))
            title.pack(pady=10)
            
            # Camera display
            self.frame_label = ctk.CTkLabel(self, text="Camera Inactive", fg_color="gray20", height=400)
            self.frame_label.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Status label
            self.status_label = ctk.CTkLabel(
                self,
                text="Status: Ready",
                fg_color="gray30",
                text_color="lightblue"
            )
            self.status_label.pack(fill="x", padx=10, pady=5)
        
        except Exception as e:
            logger.error(f"Error creating camera panel UI: {e}")
            raise
    
    def start_camera(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if camera started successfully, False otherwise.
        """
        if self.is_running:
            logger.warning("Camera already running")
            return False
        
        try:
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                logger.error("Cannot open camera")
                self.status_label.configure(text="Status: Camera Error ❌")
                return False
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            with self.thread_lock:
                self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
                self.camera_thread.start()
            
            self.status_label.configure(text="Status: Camera Running ✓")
            logger.info("Camera started successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            self.status_label.configure(text=f"Status: Error - {str(e)[:30]}")
            return False
    
    def stop_camera(self):
        """Stop camera capture and clean up resources."""
        try:
            self.is_running = False
            
            with self.thread_lock:
                if self.camera_thread and self.camera_thread.is_alive():
                    self.camera_thread.join(timeout=2.0)
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            self.status_label.configure(text="Status: Camera Stopped")
            logger.info("Camera stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    def _camera_loop(self):
        """
        Main camera capture loop running in background thread.
        
        Continuously captures frames, processes them for hand detection,
        and updates the display. Should not be called directly.
        """
        try:
            while self.is_running:
                try:
                    ret, frame = self.camera.read()
                    
                    if not ret:
                        logger.error("Failed to read camera frame")
                        break
                    
                    # Flip frame for mirror effect (selfie view)
                    frame = cv2.flip(frame, 1)
                    self.current_frame = frame

                    # Every frame is sent to the existing SignRecognizer (MediaPipe-based).
                    left_result, right_result = self.sign_recognizer.process_frame(frame)
                    self.left_hand_result = left_result
                    self.right_hand_result = right_result

                    left_display_result, left_is_new = self._update_temporal_state(left_result, "Left")
                    right_display_result, right_is_new = self._update_temporal_state(right_result, "Right")

                    # Draw hand landmarks on frame
                    if left_result and left_result.landmarks:
                        self._draw_hand_landmarks(frame, left_result.landmarks, (0, 255, 0))

                    if right_result and right_result.landmarks:
                        self._draw_hand_landmarks(frame, right_result.landmarks, (0, 165, 255))

                    # Draw detected sign labels only when above confidence threshold.
                    if left_display_result:
                        cv2.putText(
                            frame,
                            f"L: {left_display_result.sign_type.value} ({left_display_result.confidence:.2f})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                    if right_display_result:
                        cv2.putText(
                            frame,
                            f"R: {right_display_result.sign_type.value} ({right_display_result.confidence:.2f})",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 165, 255),
                            2
                        )

                    # Update recognized sign panel with threshold-filtered predictions.
                    if self.recognized_sign_callback:
                        if left_display_result and left_is_new:
                            self.recognized_sign_callback(left_display_result.sign_type.value, "Left")
                        if right_display_result and right_is_new:
                            self.recognized_sign_callback(right_display_result.sign_type.value, "Right")
                    
                    # Update display in main thread
                    if frame is not None:
                        self._update_frame_display(frame)
                
                except Exception as e:
                    logger.error(f"Error in camera loop: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Critical error in camera loop: {e}")
            self.is_running = False
    
    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks: list, color: Tuple[int, int, int]):
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: OpenCV frame to draw on.
            landmarks: List of landmark positions (normalized coordinates).
            color: RGB color tuple for drawing.
        """
        try:
            h, w = frame.shape[:2]
            
            # Draw circles at each landmark
            for landmark in landmarks:
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                cv2.circle(frame, (x, y), 5, color, -1)
            
            # Draw connections between key points (optional)
            # This creates skeleton-like visualization
            if len(landmarks) >= 21:
                # Connection pairs for hand skeleton
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
                    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
                    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
                ]
                
                for start, end in connections:
                    if start < len(landmarks) and end < len(landmarks):
                        x1 = int(landmarks[start][0] * w)
                        y1 = int(landmarks[start][1] * h)
                        x2 = int(landmarks[end][0] * w)
                        y2 = int(landmarks[end][1] * h)
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")

    def _should_display_result(self, result) -> bool:
        """
        Check whether a sign result should be shown to the user.

        Args:
            result: SignResult instance or None.

        Returns:
            True when prediction is not UNKNOWN and meets confidence threshold.
        """
        if not result:
            return False
        if result.sign_type == SignType.UNKNOWN:
            return False
        return result.confidence >= self.CONFIDENCE_THRESHOLD

    def _update_temporal_state(self, result, hand_side: str):
        """
        Apply temporal smoothing per hand to reduce flickering predictions.
        
        Requires same prediction to appear in at least 3 out of 4 consecutive frames 
        to reduce sensitivity to occasional detection noise while maintaining responsiveness.

        Args:
            result: Current frame SignResult or None.
            hand_side: Left or Right hand identifier.

        Returns:
            Tuple of (result_to_display, is_newly_accepted).
        """
        if hand_side == "Left":
            history = self.left_prediction_history
            stable_result = self.left_stable_result
            hold_counter = self.left_hold_counter
        else:
            history = self.right_prediction_history
            stable_result = self.right_stable_result
            hold_counter = self.right_hold_counter

        is_valid_candidate = self._should_display_result(result)
        current_label = result.sign_type.value if is_valid_candidate else None
        history.append(current_label)

        is_newly_accepted = False
        
        # Check if prediction is stable: same label appears in at least 3 out of 4 frames
        if len(history) == self.STABILITY_FRAMES:
            label_counts = {}
            for label in history:
                if label is not None:
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            # Find if any label appears at least 3 times
            most_common_label = max(label_counts.items(), key=lambda x: x[1])[0] if label_counts else None
            most_common_count = label_counts.get(most_common_label, 0) if most_common_label else 0
            
            if most_common_count >= 3 and most_common_label is not None:
                # Prediction is stable (appears 3+ times in last 4 frames)
                previous_label = stable_result.sign_type.value if stable_result else None
                # Update stable result to current frame if it matches the stable prediction
                if current_label == most_common_label:
                    stable_result = result
                hold_counter = self.LABEL_HOLD_FRAMES
                is_newly_accepted = previous_label != most_common_label
            else:
                # Not stable enough, start hold-down phase
                if hold_counter > 0:
                    hold_counter -= 1
                else:
                    stable_result = None
        elif stable_result and hold_counter > 0:
            # Continue holding previous stable result
            hold_counter -= 1
        else:
            # No stable result to hold
            stable_result = None
            hold_counter = 0

        if hand_side == "Left":
            self.left_stable_result = stable_result
            self.left_hold_counter = hold_counter
        else:
            self.right_stable_result = stable_result
            self.right_hold_counter = hold_counter

        return stable_result, is_newly_accepted
    
    def _update_frame_display(self, frame: np.ndarray):
        """
        Update the frame display in the UI.
        
        Args:
            frame: OpenCV frame to display.
        """
        if frame is None:
            logger.warning("Cannot display frame: frame is None")
            return
        
        try:
            # Ensure frame has valid dimensions
            if frame.size == 0:
                logger.warning("Cannot display frame: empty frame data")
                return
            # Resize frame to fit display
            h, w = frame.shape[:2]
            display_h = 400
            display_w = int(display_h * w / h)
            frame_resized = cv2.resize(frame, (display_w, display_h))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to CTkImage
            ctk_image = ctk.CTkImage(light_image=pil_image, size=(display_w, display_h))
            
            # Schedule UI update in main thread
            self.after(0, lambda: self.frame_label.configure(image=ctk_image, text=""))
            self.frame_label.image = ctk_image
        
        except Exception as e:
            logger.error(f"Error updating frame display: {e}")
    
    def set_recognized_sign_callback(self, callback):
        """
        Set callback function for recognized signs.
        
        Args:
            callback: Function to call with (sign_name, hand_side) when sign recognized.
        """
        self.recognized_sign_callback = callback
    
    def cleanup(self):
        """Clean up camera resources."""
        try:
            self.stop_camera()
            logger.info("CameraPanel cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class MainWindow(ctk.CTk):
    """
    Main application window with integrated sign language translation system.
    
    Combines camera input, sign recognition, speech input/output, and video playback
    in a cohesive user interface with dark modern theme.
    """
    
    def __init__(self):
        """Initialize main application window."""
        super().__init__()
        
        self.title("Real-Time Sign Language Translation System")
        self.geometry("1400x800")
        
        # Set dark theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Module instances
        self.sign_recognizer = None
        self.speech_handler = None
        self.sign_to_text_converter = None
        
        # UI components
        self.camera_panel = None
        self.sign_text_display = None
        self.speech_text_display = None
        self.video_player_panel = None
        
        # State variables
        self.current_speech_text = ""
        self.current_partial_speech_text = ""
        self.final_speech_lines = []
        self.last_auto_spoken_text = ""
        self.current_language = SpeechLanguage.ENGLISH
        
        try:
            self._initialize_modules()
            self._create_ui()
            self._initialize_video_player()
            self._setup_callbacks()
            logger.info("MainWindow initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MainWindow: {e}")
            raise
    
    def _initialize_modules(self):
        """Initialize all application modules."""
        try:
            logger.info("Initializing sign recognizer...")
            self.sign_recognizer = SignRecognizer()
            
            logger.info(f"Initializing speech handler for language: {self.current_language.value}...")
            self.speech_handler = SpeechHandler(
                language=self.current_language,
                tts_rate=150,
                tts_volume=0.8
            )

            logger.info("Initializing sign-to-text converter...")
            self.sign_to_text_converter = SignToTextConverter()
            
            logger.info("All modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            self._show_error_dialog(f"Initialization Error: {str(e)}")
            raise
    
    def _create_ui(self):
        """Create simplified main UI layout."""
        try:
            # Main container
            main_container = ctk.CTkFrame(self)
            main_container.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Left panel - Camera feed
            left_panel = ctk.CTkFrame(main_container, fg_color="gray25")
            left_panel.pack(side="left", fill="both", expand=True, padx=5)
            
            self.camera_panel = CameraPanel(left_panel, self.sign_recognizer)
            self.camera_panel.pack(fill="both", expand=True)
            
            # Right panel - Text outputs only
            right_panel = ctk.CTkFrame(main_container)
            right_panel.pack(side="right", fill="both", expand=True, padx=5)
            
            # Top section - Recognized signs
            signs_frame = ctk.CTkFrame(right_panel, fg_color="gray25")
            signs_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            ctk.CTkLabel(signs_frame, text="🤖 Recognized Signs", font=("Arial", 14, "bold")).pack(pady=5)
            
            self.sign_text_display = ctk.CTkTextbox(signs_frame, font=("Courier", 11))
            self.sign_text_display.pack(fill="both", expand=True, padx=5, pady=5)
            self.sign_text_display.configure(state="disabled")
            
            # Bottom section - Speech recognition
            speech_frame = ctk.CTkFrame(right_panel, fg_color="gray25")
            speech_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            ctk.CTkLabel(speech_frame, text="🎤 Speech-to-Text", font=("Arial", 14, "bold")).pack(pady=5)
            
            self.speech_text_display = ctk.CTkTextbox(speech_frame, font=("Courier", 11))
            self.speech_text_display.pack(fill="both", expand=True, padx=5, pady=5)
            self.speech_text_display.configure(state="disabled")
            
            # Control buttons section - Only 2 primary buttons
            button_frame = ctk.CTkFrame(self)
            button_frame.pack(fill="x", padx=5, pady=10)
            
            # Start Camera button
            self.camera_button = ctk.CTkButton(
                button_frame,
                text="▶ Start Camera",
                command=self._on_camera_clicked,
                width=150,
                height=40,
                font=("Arial", 12, "bold")
            )
            self.camera_button.pack(side="left", padx=5)
            
            # Speak button
            self.speak_button = ctk.CTkButton(
                button_frame,
                text="🎤 Speak",
                command=self._on_speak_clicked,
                width=150,
                height=40,
                font=("Arial", 12, "bold")
            )
            self.speak_button.pack(side="left", padx=5)
            
            # Language selector
            ctk.CTkLabel(button_frame, text="Language:", font=("Arial", 10)).pack(side="left", padx=(10, 0))
            self.language_selector = ctk.CTkComboBox(
                button_frame,
                values=["English", "Filipino"],
                command=self._on_language_changed,
                state="readonly",
                width=100,
                height=40,
                font=("Arial", 10)
            )
            self.language_selector.set("English")
            self.language_selector.pack(side="left", padx=5)
            
            # Status bar
            self.status_bar = ctk.CTkLabel(
                self,
                text="Ready",
                fg_color="gray30",
                text_color="lightgreen",
                font=("Arial", 10)
            )
            self.status_bar.pack(fill="x", padx=5, pady=5)
        
        except Exception as e:
            logger.error(f"Error creating UI: {e}")
            raise
    
    def _initialize_video_player(self):
        """Initialize video player for speech-to-sign conversion (no UI display)."""
        try:
            logger.info("Initializing video player for sign language conversion...")
            # Create a hidden container for VideoPlayerPanel
            hidden_container = ctk.CTkFrame(self)
            # Don't pack it - we only need the VideoPlayerPanel for its conversion logic
            
            # Find video folder
            video_folder = None
            for possible_path in [
                "videos",
                os.path.join(os.path.dirname(__file__), "..", "..", "videos"),
                os.path.join(os.path.dirname(__file__), "..", "videos")
            ]:
                if os.path.exists(possible_path):
                    video_folder = possible_path
                    logger.info(f"Found videos folder: {video_folder}")
                    break
            
            if video_folder:
                self.video_player_panel = VideoPlayerPanel(hidden_container, video_folder=video_folder)
                logger.info("Video player initialized successfully")
            else:
                logger.warning("No videos folder found - speech-to-sign conversion will be unavailable")
        except Exception as e:
            logger.warning(f"Could not initialize video player: {e}")
            # Don't raise - video player is optional
    
    def _setup_callbacks(self):
        """Setup callbacks for modules."""
        try:
            # Camera callback
            self.camera_panel.set_recognized_sign_callback(self._on_sign_recognized)
            
            # Protocol for window close - use default close behavior
            self.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        except Exception as e:
            logger.error(f"Error setting up callbacks: {e}")
    
    def _on_camera_clicked(self):
        """Handle Start Camera button click."""
        try:
            if self.camera_panel.is_running:
                self.camera_panel.stop_camera()
                self.camera_button.configure(text="▶ Start Camera")
                self._update_status("Camera stopped")
            else:
                if self.camera_panel.start_camera():
                    self.camera_button.configure(text="⏹ Stop Camera")
                    self._update_status("Camera running")
                else:
                    self._update_status("Failed to start camera")
        except Exception as e:
            logger.error(f"Error toggling camera: {e}")
            self._update_status(f"Error: {str(e)[:40]}")
    
    def _on_speak_clicked(self):
        """Handle Speak button click for microphone listening."""
        try:
            def on_speech_result(result: SpeechResult):
                self.after(0, lambda r=result: self._update_speech_text(r.text, r.is_final))

            if hasattr(self, '_listening') and self._listening:
                self.speech_handler.stop_listening()
                self.speak_button.configure(text="🎤 Speak")
                self._listening = False
                self._update_status("Speech recognition stopped")
            else:
                self.speech_handler.listen_and_recognize(callback=on_speech_result)
                self.speak_button.configure(text="⏹ Stop Speaking")
                self._listening = True
                self._update_status("Listening for speech...")
        except Exception as e:
            logger.error(f"Error toggling speak listening: {e}")
            self._update_status(f"Speech Error: {str(e)[:40]}")
    
    def _on_language_changed(self, language_name: str):
        """
        Handle language selection change.
        
        Args:
            language_name: Selected language name ("English" or "Filipino").
        """
        try:
            # Stop listening if currently active
            if hasattr(self, '_listening') and self._listening:
                self.speech_handler.stop_listening()
                self.speak_button.configure(text="🎤 Speak")
                self._listening = False
            
            # Map language name to enum
            if language_name == "Filipino":
                self.current_language = SpeechLanguage.FILIPINO
            else:
                self.current_language = SpeechLanguage.ENGLISH
            
            # Reinitialize speech handler with new language
            logger.info(f"Switching to {language_name} speech recognition...")
            if self.speech_handler:
                self.speech_handler.cleanup()
            
            self.speech_handler = SpeechHandler(
                language=self.current_language,
                tts_rate=150,
                tts_volume=0.8
            )
            
            self._update_status(f"Language changed to {language_name}")
        except Exception as e:
            logger.error(f"Error changing language: {e}")
            self._update_status(f"Language Error: {str(e)[:40]}")
    
    def _on_window_close(self):
        """Handle window close event."""
        try:
            logger.info("Shutting down application...")
            self._update_status("Shutting down...")
            
            # Cleanup modules
            if self.camera_panel:
                self.camera_panel.cleanup()
            
            if self.speech_handler:
                self.speech_handler.cleanup()
            
            if self.video_player_panel:
                self.video_player_panel.cleanup()
            
            logger.info("Cleanup completed")
            self.destroy()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.destroy()
    
    def _on_sign_recognized(self, sign_name: str, hand_side: str):
        """
        Handle recognized sign callback.
        
        Args:
            sign_name: Name of recognized sign.
            hand_side: Which hand detected the sign (Left/Right).
        """
        try:
            if not self.sign_to_text_converter:
                return

            appended_word = self.sign_to_text_converter.add_confirmed_prediction(sign_name)
            if appended_word:
                self._append_sign_text(appended_word)
                self._speak_confirmed_sign_text(appended_word)
        except Exception as e:
            logger.error(f"Error handling sign recognition: {e}")

    def _append_sign_text(self, word: str):
        """Append converted sign text to the existing output textbox."""
        try:
            if self.sign_text_display.winfo_exists():
                self.sign_text_display.configure(state="normal")

                existing = self.sign_text_display.get("1.0", "end-1c").strip()
                if existing:
                    self.sign_text_display.insert("end", f" {word}")
                else:
                    self.sign_text_display.insert("end", word)

                self.sign_text_display.configure(state="disabled")
                self.sign_text_display.see("end")
        except Exception as e:
            logger.error(f"Error updating sign display: {e}")

    def _speak_confirmed_sign_text(self, text: str):
        """Speak newly confirmed sign text asynchronously without repeating it."""
        try:
            if not text or not self.speech_handler:
                return

            normalized = " ".join(text.strip().split())
            if not normalized:
                return

            if normalized.lower() == self.last_auto_spoken_text.lower():
                return

            self.speech_handler.speak_async(normalized)
            self.last_auto_spoken_text = normalized
        except Exception as e:
            logger.error(f"Error auto-speaking confirmed sign text: {e}")
    
    def _update_speech_text(self, text: str, is_final: bool):
        """
        Update speech text display.
        
        Args:
            text: Recognized speech text.
            is_final: Whether this is a final result.
        """
        try:
            if self.speech_text_display.winfo_exists():
                cleaned = " ".join(text.strip().split())
                if not cleaned:
                    return

                if is_final:
                    if not self.final_speech_lines or self.final_speech_lines[-1].lower() != cleaned.lower():
                        self.final_speech_lines.append(cleaned)
                        self.current_speech_text = cleaned
                        # Play recognized speech as sign language animations
                        self._play_speech_text_as_signs(cleaned)
                    self.current_partial_speech_text = ""
                else:
                    self.current_partial_speech_text = cleaned

                lines = list(self.final_speech_lines)
                if self.current_partial_speech_text:
                    lines.append(f"... {self.current_partial_speech_text}")

                self.speech_text_display.configure(state="normal")
                self.speech_text_display.delete("1.0", "end")
                self.speech_text_display.insert("end", "\n".join(lines))
                self.speech_text_display.configure(state="disabled")
                self.speech_text_display.see("end")
        except Exception as e:
            logger.error(f"Error updating speech display: {e}")
    
    def _play_speech_text_as_signs(self, text: str):
        """
        Convert recognized speech text to sign language animations and queue for playback.
        
        Workflow:
        - Speech text → Words
        - For each word: If word video exists, play it. Else split into letters and play alphabet videos.
        
        Args:
            text: Recognized speech text to convert to signs.
        """
        try:
            if not text or not self.video_player_panel:
                return
            
            # Queue the text for playback
            self.video_player_panel.play_text_as_signs(text)
            self._update_status(f"Playing sign animations for: {text[:40]}...")
        except Exception as e:
            logger.error(f"Error converting speech to signs: {e}")
    
    def _update_status(self, message: str):
        """
        Update status bar.
        
        Args:
            message: Status message to display.
        """
        try:
            if self.status_bar.winfo_exists():
                self.status_bar.configure(text=message)
                logger.debug(f"Status: {message}")
        except Exception as e:
            logger.error(f"Error updating status: {e}")
    
    def _show_error_dialog(self, message: str):
        """
        Show error dialog.
        
        Args:
            message: Error message to display.
        """
        try:
            error_window = ctk.CTkToplevel(self)
            error_window.title("Error")
            error_window.geometry("400x150")
            
            label = ctk.CTkLabel(error_window, text=message, wraplength=380)
            label.pack(padx=10, pady=10, fill="both", expand=True)
            
            button = ctk.CTkButton(error_window, text="OK", command=error_window.destroy)
            button.pack(padx=10, pady=10)
        except Exception as e:
            logger.error(f"Error showing error dialog: {e}")


def main():
    """Main entry point for the application."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sign_language_app.log'),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Starting Real-Time Sign Language Translation System")
        
        # Create and run main window
        app = MainWindow()
        app.mainloop()
        
        logger.info("Application closed successfully")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
