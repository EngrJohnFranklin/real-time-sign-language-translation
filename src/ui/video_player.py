"""
Video Player Module for Sign Language Demonstration Videos.

This module provides a video player for displaying sign language demonstration videos
using OpenCV for video processing and CustomTkinter for GUI integration.
Supports playback controls (play, pause, stop, seek) and displays videos in a panel.
"""

import os
import threading
import logging
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np

logger = logging.getLogger(__name__)


class VideoSign(Enum):
    """Enumeration of video signs with their filenames."""
    HELLO = "hello.mp4"
    THANK_YOU = "thankyou.mp4"
    YES = "yes.mp4"
    NO = "no.mp4"
    HELP = "help.mp4"
    GOOD_MORNING = "goodmorning.mp4"
    SORRY = "sorry.mp4"
    PLEASE = "please.mp4"
    GOODBYE = "goodbye.mp4"
    I_LOVE_YOU = "iloveyou.mp4"


@dataclass
class VideoInfo:
    """Data class containing video metadata."""
    filename: str
    sign_name: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    file_path: str
    
    def __str__(self) -> str:
        """String representation of video info."""
        return f"{self.sign_name} ({self.width}x{self.height}) - {self.total_frames} frames @ {self.fps:.1f} FPS"


class PlaybackState(Enum):
    """Enumeration of video playback states."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class VideoPlayer:
    """
    Video player for sign language demonstration videos.
    
    Handles video loading, playback control, and frame display integration
    with CustomTkinter GUI. Videos are played in a separate thread to prevent
    blocking the UI.
    """
    
    # Default video folder path relative to project root
    DEFAULT_VIDEO_FOLDER = "videos"
    
    # Display configuration
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
    
    def __init__(self, video_folder: Optional[str] = None):
        """
        Initialize video player.
        
        Args:
            video_folder: Path to folder containing video files.
                         If None, uses DEFAULT_VIDEO_FOLDER in current directory.
                         
        Raises:
            RuntimeError: If video folder doesn't exist or no videos found.
        """
        self.video_folder = self._resolve_video_folder(video_folder)
        self.current_video_path = None
        self.video_capture = None
        self.current_frame = None
        self.video_info = None
        self.playback_state = PlaybackState.STOPPED
        self.current_frame_index = 0
        self.playback_thread = None
        self.is_playing = False
        self.is_running = False
        self.frame_callback = None
        self.status_callback = None
        
        try:
            self._verify_video_folder()
            logger.info(f"VideoPlayer initialized with folder: {self.video_folder}")
        except Exception as e:
            logger.error(f"Error initializing VideoPlayer: {e}")
            raise
    
    def _resolve_video_folder(self, video_folder: Optional[str]) -> str:
        """
        Resolve the path to the video folder.
        
        Args:
            video_folder: Provided video folder path, or None to use default.
            
        Returns:
            Absolute path to video folder.
        """
        try:
            if video_folder:
                return os.path.abspath(video_folder)
            else:
                # Try to find videos folder relative to current working directory
                default_path = os.path.abspath(self.DEFAULT_VIDEO_FOLDER)
                if os.path.exists(default_path):
                    return default_path
                
                # Try relative to this file's directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                project_videos = os.path.join(parent_dir, self.DEFAULT_VIDEO_FOLDER)
                
                if os.path.exists(project_videos):
                    return project_videos
                
                # Last resort: use default
                return default_path
        except Exception as e:
            logger.error(f"Error resolving video folder: {e}")
            raise
    
    def _verify_video_folder(self):
        """
        Verify video folder exists and contains video files.
        
        Logs warnings if folder doesn't exist, but does not raise exceptions
        to allow the application to continue without video functionality.
        """
        try:
            if not os.path.exists(self.video_folder):
                logger.warning(f"Video folder not found: {self.video_folder}. Video playback will not be available.")
                return
            
            if not os.path.isdir(self.video_folder):
                logger.warning(f"Path is not a directory: {self.video_folder}. Video playback will not be available.")
                return
            
            # Check for at least one video file
            video_files = [f for f in os.listdir(self.video_folder) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            if not video_files:
                logger.warning(f"No video files found in {self.video_folder}")
            else:
                logger.info(f"Found {len(video_files)} video files in {self.video_folder}")
        except Exception as e:
            logger.warning(f"Error verifying video folder: {e}. Video playback will not be available.")
    
    def get_available_videos(self) -> Dict[str, str]:
        """
        Get list of available video signs and their file paths.
        
        Returns:
            Dictionary mapping sign names to file paths.
        """
        try:
            available_videos = {}
            
            for video_sign in VideoSign:
                file_path = os.path.join(self.video_folder, video_sign.value)
                if os.path.exists(file_path):
                    available_videos[video_sign.name] = file_path
                else:
                    logger.warning(f"Video not found: {file_path}")
            
            return available_videos
        except Exception as e:
            logger.error(f"Error getting available videos: {e}")
            return {}
    
    def load_video(self, sign_name: str) -> bool:
        """
        Load a video file by sign name.
        
        Args:
            sign_name: Name of the sign (e.g., 'HELLO', 'THANK_YOU').
                      Should match VideoSign enum.
        
        Returns:
            True if video loaded successfully, False otherwise.
        """
        try:
            # Stop current playback
            self.stop()
            
            # Find the video file
            try:
                video_sign = VideoSign[sign_name.upper()]
                video_filename = video_sign.value
            except KeyError:
                logger.error(f"Unknown sign: {sign_name}")
                return False
            
            video_path = os.path.join(self.video_folder, video_filename)
            
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return False
            
            # Open video file
            self.video_capture = cv2.VideoCapture(video_path)
            
            if not self.video_capture.isOpened():
                logger.error(f"Cannot open video file: {video_path}")
                return False
            
            # Get video properties
            self.video_info = self._get_video_info(video_path)
            self.current_video_path = video_path
            self.current_frame_index = 0
            self.playback_state = PlaybackState.STOPPED
            
            logger.info(f"Video loaded: {self.video_info}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return False
    
    def _get_video_info(self, video_path: str) -> VideoInfo:
        """
        Extract metadata from video file.
        
        Args:
            video_path: Path to video file.
        
        Returns:
            VideoInfo object with video properties.
        """
        try:
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            # Get codec (approximate, as OpenCV doesn't always provide it)
            fourcc = int(self.video_capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            filename = os.path.basename(video_path)
            sign_name = filename.replace('.mp4', '').replace('_', ' ').title()
            
            return VideoInfo(
                filename=filename,
                sign_name=sign_name,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration_seconds,
                codec=codec,
                file_path=video_path
            )
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            # Return default VideoInfo
            return VideoInfo(
                filename=os.path.basename(video_path),
                sign_name="Unknown",
                width=0,
                height=0,
                fps=0,
                total_frames=0,
                duration_seconds=0,
                codec="Unknown",
                file_path=video_path
            )
    
    def play(self) -> bool:
        """
        Start playing the loaded video.
        
        Returns:
            True if playback started successfully, False otherwise.
        """
        try:
            if not self.video_capture or not self.video_capture.isOpened():
                logger.error("No video loaded")
                return False
            
            if self.playback_state == PlaybackState.PLAYING:
                logger.warning("Video already playing")
                return True
            
            self.is_playing = True
            self.playback_state = PlaybackState.PLAYING
            self.is_running = True
            
            # Start playback in separate thread
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
            
            logger.info("Video playback started")
            return True
        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            return False
    
    def _playback_loop(self):
        """
        Main playback loop running in separate thread.
        
        Continuously reads frames from video and calls frame callback.
        This runs in a background thread and should not be called directly.
        """
        try:
            if not self.video_capture:
                return
            
            while self.is_playing and self.is_running:
                try:
                    ret, frame = self.video_capture.read()
                    
                    if not ret:
                        # End of video reached
                        logger.info("End of video reached")
                        self.stop()
                        break
                    
                    # Store current frame
                    self.current_frame = frame
                    self.current_frame_index = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    # Call frame callback if provided
                    if self.frame_callback:
                        self.frame_callback(frame)
                    
                    # Control playback speed using frame rate
                    if self.video_info and self.video_info.fps > 0:
                        frame_delay = int(1000 / self.video_info.fps)
                        threading.Event().wait(frame_delay / 1000.0)
                
                except Exception as e:
                    logger.error(f"Error in playback loop: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Critical error in playback loop: {e}")
            self.is_playing = False
    
    def pause(self) -> bool:
        """
        Pause video playback.
        
        Returns:
            True if video was paused successfully, False otherwise.
        """
        try:
            if self.playback_state != PlaybackState.PLAYING:
                logger.warning("Video is not playing")
                return False
            
            self.is_playing = False
            self.playback_state = PlaybackState.PAUSED
            logger.info("Video paused")
            return True
        except Exception as e:
            logger.error(f"Error pausing video: {e}")
            return False
    
    def resume(self) -> bool:
        """
        Resume paused video playback.
        
        Returns:
            True if video was resumed successfully, False otherwise.
        """
        try:
            if self.playback_state != PlaybackState.PAUSED:
                logger.warning("Video is not paused")
                return False
            
            self.is_playing = True
            self.playback_state = PlaybackState.PLAYING
            
            # Restart playback thread if needed
            if not self.playback_thread or not self.playback_thread.is_alive():
                self.is_running = True
                self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
                self.playback_thread.start()
            
            logger.info("Video resumed")
            return True
        except Exception as e:
            logger.error(f"Error resuming video: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop video playback and reset to beginning.
        
        Returns:
            True if video was stopped successfully, False otherwise.
        """
        try:
            self.is_playing = False
            self.is_running = False
            self.playback_state = PlaybackState.STOPPED
            
            # Wait for playback thread to finish
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=1.0)
            
            # Reset video to beginning
            if self.video_capture:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_index = 0
            
            logger.info("Video stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping video: {e}")
            return False
    
    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame in the video.
        
        Args:
            frame_number: Frame number to seek to (0-indexed).
        
        Returns:
            True if seek was successful, False otherwise.
        """
        try:
            if not self.video_capture:
                logger.error("No video loaded")
                return False
            
            if self.video_info and frame_number >= self.video_info.total_frames:
                logger.warning(f"Frame {frame_number} exceeds total frames {self.video_info.total_frames}")
                return False
            
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_index = frame_number
            
            # Read the frame at seek position
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame
                if self.frame_callback:
                    self.frame_callback(frame)
            
            logger.debug(f"Seeked to frame {frame_number}")
            return True
        except Exception as e:
            logger.error(f"Error seeking to frame: {e}")
            return False
    
    def seek_to_time(self, time_seconds: float) -> bool:
        """
        Seek to a specific time in the video.
        
        Args:
            time_seconds: Time in seconds to seek to.
        
        Returns:
            True if seek was successful, False otherwise.
        """
        try:
            if not self.video_capture or not self.video_info:
                logger.error("No video loaded")
                return False
            
            # Calculate frame number from time
            frame_number = int(time_seconds * self.video_info.fps)
            return self.seek_to_frame(frame_number)
        except Exception as e:
            logger.error(f"Error seeking to time: {e}")
            return False
    
    def get_current_time(self) -> float:
        """
        Get current playback time in seconds.
        
        Returns:
            Current time in seconds, or 0 if no video loaded.
        """
        try:
            if not self.video_info or self.video_info.fps == 0:
                return 0.0
            
            return self.current_frame_index / self.video_info.fps
        except Exception as e:
            logger.error(f"Error getting current time: {e}")
            return 0.0
    
    def get_progress_percentage(self) -> float:
        """
        Get current playback progress as percentage.
        
        Returns:
            Progress percentage (0-100), or 0 if no video loaded.
        """
        try:
            if not self.video_info or self.video_info.total_frames == 0:
                return 0.0
            
            return (self.current_frame_index / self.video_info.total_frames) * 100
        except Exception as e:
            logger.error(f"Error getting progress: {e}")
            return 0.0
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """
        Set callback function to be called when a new frame is ready.
        
        Args:
            callback: Function to call with frame numpy array.
                     Signature: callback(frame: np.ndarray) -> None
        """
        self.frame_callback = callback
    
    def set_status_callback(self, callback: Callable[[str, dict], None]):
        """
        Set callback function to be called on status updates.
        
        Args:
            callback: Function to call with status updates.
                     Signature: callback(status: str, data: dict) -> None
        """
        self.status_callback = callback
    
    def cleanup(self):
        """
        Clean up video resources and stop playback.
        
        Should be called before application exit to properly release
        video file and thread resources.
        """
        try:
            self.stop()
            
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            
            logger.info("VideoPlayer cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class VideoPlayerPanel(ctk.CTkFrame):
    """
    CustomTkinter GUI panel for video player.
    
    Provides an integrated video player panel with playback controls
    and video display area. Can be embedded in any CTkFrame.
    """
    
    def __init__(self, parent, video_folder: Optional[str] = None, **kwargs):
        """
        Initialize video player panel.
        
        Args:
            parent: Parent CustomTkinter widget.
            video_folder: Path to folder containing videos.
            **kwargs: Additional arguments for CTkFrame.
        """
        super().__init__(parent, **kwargs)
        
        self.video_player = None
        self.video_label = None
        self.info_label = None
        self.progress_var = None
        
        try:
            self.video_player = VideoPlayer(video_folder)
            self._create_ui()
            logger.info("VideoPlayerPanel initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VideoPlayerPanel: {e}")
            raise
    
    def _create_ui(self):
        """Create UI elements for the video player panel."""
        try:
            # Video display area
            self.video_label = ctk.CTkLabel(self, text="No Video Loaded", fg_color="gray20")
            self.video_label.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Control panel
            control_frame = ctk.CTkFrame(self)
            control_frame.pack(fill="x", padx=5, pady=5)
            
            # Video selection dropdown
            video_list = list(self.video_player.get_available_videos().keys())
            self.video_combo = ctk.CTkComboBox(
                control_frame,
                values=video_list,
                command=self._on_video_selected,
                width=200
            )
            self.video_combo.pack(side="left", padx=5)
            
            # Play button
            self.play_button = ctk.CTkButton(
                control_frame,
                text="▶ Play",
                command=self._on_play_clicked,
                width=80
            )
            self.play_button.pack(side="left", padx=2)
            
            # Pause button
            self.pause_button = ctk.CTkButton(
                control_frame,
                text="⏸ Pause",
                command=self._on_pause_clicked,
                width=80
            )
            self.pause_button.pack(side="left", padx=2)
            
            # Stop button
            self.stop_button = ctk.CTkButton(
                control_frame,
                text="⏹ Stop",
                command=self._on_stop_clicked,
                width=80
            )
            self.stop_button.pack(side="left", padx=2)
            
            # Progress bar
            progress_frame = ctk.CTkFrame(self)
            progress_frame.pack(fill="x", padx=5, pady=5)
            
            ctk.CTkLabel(progress_frame, text="Progress:").pack(side="left", padx=5)
            
            self.progress_var = ctk.DoubleVar(value=0)
            self.progress_slider = ctk.CTkSlider(
                progress_frame,
                from_=0,
                to=100,
                variable=self.progress_var,
                command=self._on_progress_changed,
                width=300
            )
            self.progress_slider.pack(side="left", fill="x", expand=True, padx=5)
            
            self.time_label = ctk.CTkLabel(progress_frame, text="00:00 / 00:00", width=100)
            self.time_label.pack(side="left", padx=5)
            
            # Info label
            self.info_label = ctk.CTkLabel(self, text="Load a video to see details", fg_color="gray30")
            self.info_label.pack(fill="x", padx=5, pady=5)
            
            # Set frame callback to update display
            self.video_player.set_frame_callback(self._on_frame_received)
        
        except Exception as e:
            logger.error(f"Error creating UI: {e}")
            raise
    
    def _on_video_selected(self, selected_video: str):
        """Handle video selection from dropdown."""
        try:
            if self.video_player.load_video(selected_video):
                info = self.video_player.video_info
                self.info_label.configure(text=f"Loaded: {info}")
                self.progress_var.set(0)
                self._update_time_label()
            else:
                self.info_label.configure(text="Failed to load video")
        except Exception as e:
            logger.error(f"Error selecting video: {e}")
    
    def _on_play_clicked(self):
        """Handle play button click."""
        try:
            self.video_player.play()
        except Exception as e:
            logger.error(f"Error playing video: {e}")
    
    def _on_pause_clicked(self):
        """Handle pause button click."""
        try:
            self.video_player.pause()
        except Exception as e:
            logger.error(f"Error pausing video: {e}")
    
    def _on_stop_clicked(self):
        """Handle stop button click."""
        try:
            self.video_player.stop()
            self.progress_var.set(0)
            self._update_time_label()
        except Exception as e:
            logger.error(f"Error stopping video: {e}")
    
    def _on_progress_changed(self, value: float):
        """Handle progress slider movement."""
        try:
            if self.video_player.video_info:
                time_seconds = (value / 100) * self.video_player.video_info.duration_seconds
                self.video_player.seek_to_time(time_seconds)
        except Exception as e:
            logger.error(f"Error changing progress: {e}")
    
    def _on_frame_received(self, frame: np.ndarray):
        """
        Handle new frame from video player.
        
        Converts OpenCV frame to CTkImage and displays in label.
        """
        try:
            if frame is None or frame.size == 0:
                return
            
            # Resize frame to display size
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            display_h = 400
            display_w = int(display_h * aspect_ratio)
            
            frame_resized = cv2.resize(frame, (display_w, display_h))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to CTkImage
            ctk_image = ctk.CTkImage(light_image=pil_image, size=(display_w, display_h))
            
            # Update label
            self.video_label.configure(image=ctk_image, text="")
            self.video_label.image = ctk_image
            
            # Update progress
            if self.video_player.video_info:
                progress = self.video_player.get_progress_percentage()
                self.progress_var.set(progress)
                self._update_time_label()
        
        except Exception as e:
            logger.error(f"Error updating frame display: {e}")
    
    def _update_time_label(self):
        """Update the time display label."""
        try:
            current_time = self.video_player.get_current_time()
            total_time = self.video_player.video_info.duration_seconds if self.video_player.video_info else 0
            
            current_str = self._format_time(current_time)
            total_str = self._format_time(total_time)
            
            self.time_label.configure(text=f"{current_str} / {total_str}")
        except Exception as e:
            logger.error(f"Error updating time label: {e}")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to MM:SS format."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.video_player:
                self.video_player.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage and testing
if __name__ == "__main__":
    try:
        # Create root window
        root = ctk.CTk()
        root.title("Sign Language Video Player")
        root.geometry("800x700")
        
        # Create video player panel
        video_panel = VideoPlayerPanel(root)
        video_panel.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Handle cleanup on window close
        def on_closing():
            video_panel.cleanup()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start GUI
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        print(f"Error: {e}")
