"""
Main GUI Application for Real-Time Sign Language Translation.

This module provides the main user interface using CustomTkinter with a dark modern theme.
Integrates sign detection, speech recognition, text-to-speech, and video playback modules.

Layout:
- Left panel: Live webcam feed with hand landmarks
- Right top: Recognized sign text
- Right middle: Speech-to-text output
- Right bottom: Sign language video demonstrations
- Control buttons: Start Camera, Start Listening, Speak Text, Play Sign Video, Clear, Exit
"""

import os
import sys
import threading
import logging
from typing import Optional, Tuple
from pathlib import Path

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
from ui.video_player import VideoPlayer, VideoSign

logger = logging.getLogger(__name__)


class CameraPanel(ctk.CTkFrame):
    """
    Left panel displaying live webcam feed with hand landmark overlay.
    
    Handles camera initialization, frame capture, and hand landmark visualization.
    Runs camera capture in background thread to prevent UI blocking.
    """
    
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
        self.current_frame = None
        self.frame_label = None
        self.status_label = None
        self.left_hand_result = None
        self.right_hand_result = None
        self.recognized_sign_callback = None
        
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
                    
                    # Process frame with sign recognizer
                    left_result, right_result = self.sign_recognizer.process_frame(frame)
                    self.left_hand_result = left_result
                    self.right_hand_result = right_result
                    
                    # Draw hand landmarks on frame
                    if left_result and left_result.landmarks:
                        self._draw_hand_landmarks(frame, left_result.landmarks, (0, 255, 0))
                    
                    if right_result and right_result.landmarks:
                        self._draw_hand_landmarks(frame, right_result.landmarks, (0, 165, 255))
                    
                    # Draw recognized sign text
                    if left_result and left_result.sign_type != SignType.UNKNOWN:
                        cv2.putText(frame, f"L: {left_result.sign_type.value}",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if right_result and right_result.sign_type != SignType.UNKNOWN:
                        cv2.putText(frame, f"R: {right_result.sign_type.value}",
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    
                    # Call recognized sign callback
                    if self.recognized_sign_callback:
                        if left_result and left_result.sign_type != SignType.UNKNOWN:
                            self.recognized_sign_callback(left_result.sign_type.value, "Left")
                        if right_result and right_result.sign_type != SignType.UNKNOWN:
                            self.recognized_sign_callback(right_result.sign_type.value, "Right")
                    
                    # Update display in main thread
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
    
    def _update_frame_display(self, frame: np.ndarray):
        """
        Update the frame display in the UI.
        
        Args:
            frame: OpenCV frame to display.
        """
        try:
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
        self.video_player = None
        
        # UI components
        self.camera_panel = None
        self.sign_text_display = None
        self.speech_text_display = None
        self.video_player_panel = None
        
        # State variables
        self.current_recognized_signs = []
        self.current_speech_text = ""
        
        try:
            self._initialize_modules()
            self._create_ui()
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
            
            logger.info("Initializing speech handler...")
            self.speech_handler = SpeechHandler(
                language=SpeechLanguage.ENGLISH,
                tts_rate=150,
                tts_volume=0.8
            )
            
            logger.info("Initializing video player...")
            # Try to find videos folder in various locations
            video_folder = None
            for possible_path in [
                "videos",
                os.path.join(os.path.dirname(__file__), "..", "..", "videos"),
                os.path.join(os.path.dirname(__file__), "..", "videos")
            ]:
                if os.path.exists(possible_path):
                    video_folder = possible_path
                    break
            
            self.video_player = VideoPlayer(video_folder)
            
            logger.info("All modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            self._show_error_dialog(f"Initialization Error: {str(e)}")
            raise
    
    def _create_ui(self):
        """Create main UI layout."""
        try:
            # Main container
            main_container = ctk.CTkFrame(self)
            main_container.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Left panel - Camera feed
            left_panel = ctk.CTkFrame(main_container, fg_color="gray25")
            left_panel.pack(side="left", fill="both", expand=True, padx=5)
            
            self.camera_panel = CameraPanel(left_panel, self.sign_recognizer)
            self.camera_panel.pack(fill="both", expand=True)
            
            # Right panel - Information and controls
            right_panel = ctk.CTkFrame(main_container)
            right_panel.pack(side="right", fill="both", expand=True, padx=5)
            
            # Top section - Recognized signs
            signs_frame = ctk.CTkFrame(right_panel, fg_color="gray25")
            signs_frame.pack(fill="x", padx=5, pady=5)
            
            ctk.CTkLabel(signs_frame, text="🤖 Recognized Signs", font=("Arial", 14, "bold")).pack(pady=5)
            
            self.sign_text_display = ctk.CTkTextbox(signs_frame, height=80, font=("Courier", 11))
            self.sign_text_display.pack(fill="both", expand=True, padx=5, pady=5)
            self.sign_text_display.configure(state="disabled")
            
            # Middle section - Speech recognition
            speech_frame = ctk.CTkFrame(right_panel, fg_color="gray25")
            speech_frame.pack(fill="x", padx=5, pady=5)
            
            ctk.CTkLabel(speech_frame, text="🎤 Speech-to-Text", font=("Arial", 14, "bold")).pack(pady=5)
            
            self.speech_text_display = ctk.CTkTextbox(speech_frame, height=80, font=("Courier", 11))
            self.speech_text_display.pack(fill="both", expand=True, padx=5, pady=5)
            self.speech_text_display.configure(state="disabled")
            
            # Bottom section - Video player
            video_frame = ctk.CTkFrame(right_panel, fg_color="gray25")
            video_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            ctk.CTkLabel(video_frame, text="🎬 Sign Language Videos", font=("Arial", 14, "bold")).pack(pady=5)
            
            # Import VideoPlayerPanel here to avoid circular imports
            from ui.video_player import VideoPlayerPanel
            
            self.video_player_panel = VideoPlayerPanel(
                video_frame,
                video_folder=self.video_player.video_folder
            )
            self.video_player_panel.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Control buttons section
            button_frame = ctk.CTkFrame(self)
            button_frame.pack(fill="x", padx=5, pady=10)
            
            # Start Camera button
            self.camera_button = ctk.CTkButton(
                button_frame,
                text="▶ Start Camera",
                command=self._on_camera_clicked,
                width=120,
                height=40,
                font=("Arial", 11, "bold")
            )
            self.camera_button.pack(side="left", padx=5)
            
            # Start Listening button
            self.listen_button = ctk.CTkButton(
                button_frame,
                text="🎤 Listen",
                command=self._on_listen_clicked,
                width=120,
                height=40,
                font=("Arial", 11, "bold")
            )
            self.listen_button.pack(side="left", padx=5)
            
            # Speak Text button
            self.speak_button = ctk.CTkButton(
                button_frame,
                text="🔊 Speak",
                command=self._on_speak_clicked,
                width=120,
                height=40,
                font=("Arial", 11, "bold")
            )
            self.speak_button.pack(side="left", padx=5)
            
            # Clear button
            self.clear_button = ctk.CTkButton(
                button_frame,
                text="🗑 Clear",
                command=self._on_clear_clicked,
                width=120,
                height=40,
                font=("Arial", 11, "bold"),
                fg_color="orange"
            )
            self.clear_button.pack(side="left", padx=5)
            
            # Exit button
            self.exit_button = ctk.CTkButton(
                button_frame,
                text="❌ Exit",
                command=self._on_exit_clicked,
                width=120,
                height=40,
                font=("Arial", 11, "bold"),
                fg_color="red"
            )
            self.exit_button.pack(side="left", padx=5)
            
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
    
    def _setup_callbacks(self):
        """Setup callbacks for modules."""
        try:
            # Camera callback
            self.camera_panel.set_recognized_sign_callback(self._on_sign_recognized)
            
            # Protocol for window close
            self.protocol("WM_DELETE_WINDOW", self._on_exit_clicked)
        
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
    
    def _on_listen_clicked(self):
        """Handle Start Listening button click."""
        try:
            def on_speech_result(result: SpeechResult):
                self._update_speech_text(result.text, result.is_final)
            
            if hasattr(self, '_listening') and self._listening:
                self.speech_handler.stop_listening()
                self.listen_button.configure(text="🎤 Listen")
                self._listening = False
                self._update_status("Speech recognition stopped")
            else:
                self.speech_handler.listen_and_recognize(callback=on_speech_result)
                self.listen_button.configure(text="⏹ Stop Listening")
                self._listening = True
                self._update_status("Listening for speech...")
        except Exception as e:
            logger.error(f"Error toggling listening: {e}")
            self._update_status(f"Speech Error: {str(e)[:40]}")
    
    def _on_speak_clicked(self):
        """Handle Speak Text button click."""
        try:
            if self.speech_text_display.winfo_exists():
                self.speech_text_display.configure(state="normal")
                text = self.speech_text_display.get("1.0", "end-1c")
                self.speech_text_display.configure(state="disabled")
                
                if text.strip():
                    self._update_status("Speaking...")
                    self.speech_handler.speak_async(text)
                    self._update_status("Finished speaking")
                else:
                    self._update_status("No text to speak")
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
            self._update_status(f"TTS Error: {str(e)[:40]}")
    
    def _on_clear_clicked(self):
        """Handle Clear button click."""
        try:
            # Clear sign text
            if self.sign_text_display.winfo_exists():
                self.sign_text_display.configure(state="normal")
                self.sign_text_display.delete("1.0", "end")
                self.sign_text_display.configure(state="disabled")
            
            # Clear speech text
            if self.speech_text_display.winfo_exists():
                self.speech_text_display.configure(state="normal")
                self.speech_text_display.delete("1.0", "end")
                self.speech_text_display.configure(state="disabled")
            
            self.current_recognized_signs = []
            self.current_speech_text = ""
            self._update_status("Cleared all text")
        except Exception as e:
            logger.error(f"Error clearing text: {e}")
    
    def _on_exit_clicked(self):
        """Handle Exit button click."""
        try:
            logger.info("Shutting down application...")
            self._update_status("Shutting down...")
            
            # Cleanup modules
            if self.camera_panel:
                self.camera_panel.cleanup()
            
            if self.speech_handler:
                self.speech_handler.cleanup()
            
            if self.video_player:
                self.video_player.cleanup()
            
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
            # Add to sign list
            sign_entry = f"{hand_side}: {sign_name}"
            if sign_entry not in self.current_recognized_signs:
                self.current_recognized_signs.append(sign_entry)
                
                # Keep only last 10 signs
                if len(self.current_recognized_signs) > 10:
                    self.current_recognized_signs.pop(0)
                
                # Update display
                self._update_sign_display()
        except Exception as e:
            logger.error(f"Error handling sign recognition: {e}")
    
    def _update_sign_display(self):
        """Update the sign text display."""
        try:
            if self.sign_text_display.winfo_exists():
                self.sign_text_display.configure(state="normal")
                self.sign_text_display.delete("1.0", "end")
                
                for sign in self.current_recognized_signs:
                    self.sign_text_display.insert("end", f"• {sign}\n")
                
                self.sign_text_display.configure(state="disabled")
        except Exception as e:
            logger.error(f"Error updating sign display: {e}")
    
    def _update_speech_text(self, text: str, is_final: bool):
        """
        Update speech text display.
        
        Args:
            text: Recognized speech text.
            is_final: Whether this is a final result.
        """
        try:
            if self.speech_text_display.winfo_exists():
                self.speech_text_display.configure(state="normal")
                
                if is_final:
                    # Append final result with newline
                    self.speech_text_display.insert("end", f"{text}\n")
                    self.current_speech_text = text
                else:
                    # Update last line with partial result
                    content = self.speech_text_display.get("1.0", "end")
                    lines = content.split('\n')
                    if lines and not lines[-1].endswith('\n'):
                        # Replace partial result
                        self.speech_text_display.delete("1.0", "end")
                        self.speech_text_display.insert("end", '\n'.join(lines[:-1]))
                        if lines[:-1]:
                            self.speech_text_display.insert("end", "\n")
                        self.speech_text_display.insert("end", f"{text}")
                
                self.speech_text_display.configure(state="disabled")
                self.speech_text_display.see("end")
        except Exception as e:
            logger.error(f"Error updating speech display: {e}")
    
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
