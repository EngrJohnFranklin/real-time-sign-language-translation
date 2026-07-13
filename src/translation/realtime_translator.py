"""
Real-Time Sign Language Translator

Handles the display and TTS logic for real-time sign language translation.
Features:
- Displays only the current recognized sign (not accumulated)
- Ignores duplicate detections while same sign is held
- Automatically clears display after 3 seconds of no detection
- Triggers TTS only once per newly recognized sign
- Tracks confidence and hand side information
"""

import logging
import time
from typing import Optional, Tuple, Callable
from enum import Enum

logger = logging.getLogger(__name__)

# Mapping from gesture enum names to natural language meanings
GESTURE_TO_MEANING = {
    "Closed Fist": "Sorry",
    "Open Palm": "Good Morning",
    "Thumbs Up": "Yes",
    "Thumbs Down": "No",
    "Index Finger": "Help",
    "Peace Sign": "Thank You",
    "OK Sign": "Good",
    "I Love You": "I Love You",
    "Hello": "Hello",
    "Goodbye": "Goodbye",
    "Please": "Please",
    "Unknown": "",
}

# Timeout in seconds after which display is cleared if no new sign is detected
DISPLAY_TIMEOUT = 3.0

# Confidence threshold for TTS
SPEECH_CONFIDENCE_THRESHOLD = 0.70


class RealTimeTranslator:
    """
    Translates detected hand signs to natural language in real-time.
    
    Manages:
    - Current sign display (single sign at a time)
    - Duplicate detection (ignores while same sign is held)
    - Timeout-based clearing (clears after 3 seconds of inactivity)
    - TTS triggering (only once per new sign)
    - Hand side tracking
    - Confidence filtering
    """

    def __init__(self):
        """Initialize the real-time translator."""
        self.current_sign: Optional[str] = None  # Natural language meaning
        self.current_gesture_name: Optional[str] = None  # Gesture enum name (e.g., "Hello")
        self.current_hand_side: Optional[str] = None
        self.current_confidence: float = 0.0
        self.last_update_time: float = time.time()
        self.sign_spoken: bool = False  # Has TTS been triggered for current sign?

    def update(
        self, 
        gesture_name: str, 
        hand_side: str, 
        confidence: float = 0.8
    ) -> Tuple[Optional[str], bool]:
        """
        Process a newly detected gesture.
        
        Args:
            gesture_name: Gesture enum name (e.g., "Hello", "Thumbs Up")
            hand_side: "Left" or "Right"
            confidence: Confidence score (0.0-1.0)
            
        Returns:
            Tuple of (meaning_to_display, should_speak):
            - meaning_to_display: Natural language meaning if sign changed, None if duplicate
            - should_speak: True if TTS should be triggered (only once per sign)
        """
        current_time = time.time()
        self.last_update_time = current_time
        
        # Check if this is a new sign (different from current or first sign)
        is_new_sign = (
            self.current_gesture_name != gesture_name or
            gesture_name == "Unknown"
        )
        
        if is_new_sign:
            # New sign detected - update current sign
            self.current_gesture_name = gesture_name
            self.current_hand_side = hand_side
            self.current_confidence = confidence
            self.sign_spoken = False  # Reset TTS flag for new sign
            
            # Convert to natural language meaning
            self.current_sign = GESTURE_TO_MEANING.get(gesture_name, "")
            
            logger.info(
                f"New sign detected: {gesture_name} → {self.current_sign} "
                f"({hand_side}, confidence: {confidence:.2f})"
            )
            
            # Should speak only if confidence is high enough and not Unknown
            should_speak = (
                confidence >= SPEECH_CONFIDENCE_THRESHOLD and
                gesture_name != "Unknown" and
                not self.sign_spoken
            )
            
            if should_speak:
                self.sign_spoken = True
            
            return (self.current_sign, should_speak)
        
        # Same sign - no update needed
        return (None, False)

    def check_timeout(self) -> Optional[str]:
        """
        Check if display should be cleared due to inactivity timeout.
        
        Returns:
            Empty string if timeout reached (to clear display), None otherwise
        """
        if self.current_sign and self.current_gesture_name != "Unknown":
            elapsed = time.time() - self.last_update_time
            if elapsed >= DISPLAY_TIMEOUT:
                logger.info(
                    f"Sign display timeout ({elapsed:.1f}s) - clearing '{self.current_sign}'"
                )
                self.clear()
                return ""  # Return empty string to clear display
        
        return None

    def clear(self) -> None:
        """Clear the current sign and reset state."""
        self.current_sign = None
        self.current_gesture_name = None
        self.current_hand_side = None
        self.current_confidence = 0.0
        self.sign_spoken = False
        self.last_update_time = time.time()

    def get_current_display(self) -> str:
        """
        Get the current display text.
        
        Returns:
            Natural language meaning of current sign, or empty string if none
        """
        return self.current_sign or ""

    def get_status(self) -> str:
        """
        Get detailed status information.
        
        Returns:
            Status string with current sign, hand, and confidence
        """
        if not self.current_sign or self.current_gesture_name == "Unknown":
            return "No sign detected"
        
        return (
            f"{self.current_sign} "
            f"({self.current_hand_side} hand, {self.current_confidence:.0%})"
        )
