"""
Sign Language Detector using MediaPipe Hand Landmarks and Geometric Rules.

This module recognizes 10 specific sign language gestures using hand landmark
positions and geometric properties (angles, distances) without requiring ML models.
Supported signs: Hello, Thank You, Yes, No, Help, Good Morning, Sorry, Please, Goodbye, I Love You.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum
import logging

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class SignType(Enum):
    """Enumeration of supported sign types."""
    HELLO = "Hello"
    THANK_YOU = "Thank You"
    YES = "Yes"
    NO = "No"
    HELP = "Help"
    GOOD_MORNING = "Good Morning"
    SORRY = "Sorry"
    PLEASE = "Please"
    GOODBYE = "Goodbye"
    I_LOVE_YOU = "I Love You"
    UNKNOWN = "Unknown"


@dataclass
class SignResult:
    """Data class to hold sign recognition results."""
    sign_type: SignType
    confidence: float
    hand_side: str  # 'Left' or 'Right'
    landmarks: Optional[List[Tuple[float, float, float]]] = None
    
    def __str__(self) -> str:
        """String representation of sign result."""
        return f"{self.sign_type.value} ({self.confidence:.2f}) - {self.hand_side} hand"


class HandGeometryAnalyzer:
    """
    Analyzes hand geometry using MediaPipe landmarks.
    Calculates distances, angles, and geometric properties for sign recognition.
    """
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    # Finger PIP (Proximal Interphalangeal) joint indices
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18
    THUMB_IP = 3
    
    # Finger base (MCP) joint indices
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17
    THUMB_MCP = 2
    
    def __init__(self):
        """Initialize hand geometry analyzer."""
        pass
    
    @staticmethod
    def distance(point1: Tuple[float, float, float], 
                point2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two 3D points.
        
        Args:
            point1: First point (x, y, z)
            point2: Second point (x, y, z)
            
        Returns:
            Euclidean distance between points
        """
        try:
            return math.sqrt(
                (point1[0] - point2[0]) ** 2 +
                (point1[1] - point2[1]) ** 2 +
                (point1[2] - point2[2]) ** 2
            )
        except (TypeError, IndexError) as e:
            logger.error(f"Error calculating distance: {e}")
            return 0.0
    
    @staticmethod
    def angle_between_points(point1: Tuple[float, float, float],
                            vertex: Tuple[float, float, float],
                            point2: Tuple[float, float, float]) -> float:
        """
        Calculate angle (in degrees) formed by three points.
        Vertex is the angle vertex.
        
        Args:
            point1: First point
            vertex: Vertex point (angle is measured at this point)
            point2: Second point
            
        Returns:
            Angle in degrees (0-180)
        """
        try:
            # Calculate vectors from vertex to the two points
            vec1 = np.array([
                point1[0] - vertex[0],
                point1[1] - vertex[1],
                point1[2] - vertex[2]
            ])
            vec2 = np.array([
                point2[0] - vertex[0],
                point2[1] - vertex[1],
                point2[2] - vertex[2]
            ])
            
            # Avoid division by zero
            magnitude1 = np.linalg.norm(vec1)
            magnitude2 = np.linalg.norm(vec2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            # Calculate cosine of angle
            cos_angle = np.dot(vec1, vec2) / (magnitude1 * magnitude2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to valid range
            
            # Convert to degrees
            angle_rad = math.acos(cos_angle)
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return 0.0
    
    def is_finger_extended(self, landmarks: List[Tuple[float, float, float]],
                          tip_idx: int, pip_idx: int) -> bool:
        """
        Determine if a finger is extended (pointing) or curled.
        
        Args:
            landmarks: List of hand landmarks
            tip_idx: Index of finger tip
            pip_idx: Index of finger PIP joint
            
        Returns:
            True if finger is extended, False if curled
        """
        try:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            
            # If tip is significantly above PIP (y coordinate), finger is extended
            # In MediaPipe, y decreases as we go up
            return tip[1] < pip[1] - 0.05
        except (IndexError, TypeError) as e:
            logger.error(f"Error checking finger extension: {e}")
            return False
    
    def get_hand_position(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Get hand position and orientation information.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Dictionary with position information
        """
        try:
            wrist = landmarks[self.WRIST]
            middle_tip = landmarks[self.MIDDLE_TIP]
            
            return {
                'wrist_x': wrist[0],
                'wrist_y': wrist[1],
                'wrist_z': wrist[2],
                'hand_height': abs(middle_tip[1] - wrist[1]),
                'hand_direction': 'up' if middle_tip[1] < wrist[1] else 'down'
            }
        except (IndexError, TypeError) as e:
            logger.error(f"Error getting hand position: {e}")
            return {}
    
    def analyze_finger_configuration(self, landmarks: List[Tuple[float, float, float]]) -> Dict[str, bool]:
        """
        Analyze which fingers are extended or curled.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Dictionary with finger extension states
        """
        try:
            return {
                'thumb_extended': self.is_finger_extended(landmarks, self.THUMB_TIP, self.THUMB_IP),
                'index_extended': self.is_finger_extended(landmarks, self.INDEX_TIP, self.INDEX_PIP),
                'middle_extended': self.is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP),
                'ring_extended': self.is_finger_extended(landmarks, self.RING_TIP, self.RING_PIP),
                'pinky_extended': self.is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_PIP),
            }
        except Exception as e:
            logger.error(f"Error analyzing finger configuration: {e}")
            return {}


class SignRecognizer:
    """
    Main sign language recognizer using geometric rules and hand landmarks.
    Uses MediaPipe for hand detection and custom geometric rules for sign recognition.
    """
    
    def __init__(self):
        """Initialize the sign recognizer with MediaPipe hand detector."""
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.analyzer = HandGeometryAnalyzer()
            logger.info("SignRecognizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SignRecognizer: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[SignResult], Optional[SignResult]]:
        """
        Process a frame and recognize signs in both hands.
        
        Args:
            frame: Input frame from video capture (BGR format)
            
        Returns:
            Tuple of (left_hand_result, right_hand_result). Each can be None if hand not detected.
        """
        try:
            if frame is None or frame.size == 0:
                logger.warning("Empty frame received")
                return None, None
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks or not results.multi_handedness:
                return None, None
            
            left_result = None
            right_result = None
            
            # Process each detected hand
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Convert landmarks to list of tuples
                landmark_list = [
                    (lm.x, lm.y, lm.z) for lm in landmarks.landmark
                ]
                
                hand_side = handedness.classification[0].label
                recognized_sign = self._recognize_sign(landmark_list)
                
                if hand_side == 'Right':
                    right_result = SignResult(
                        sign_type=recognized_sign,
                        confidence=0.8,  # Geometric rules don't provide confidence scores
                        hand_side='Right',
                        landmarks=landmark_list
                    )
                else:
                    left_result = SignResult(
                        sign_type=recognized_sign,
                        confidence=0.8,
                        hand_side='Left',
                        landmarks=landmark_list
                    )
            
            return left_result, right_result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None, None
    
    def _recognize_sign(self, landmarks: List[Tuple[float, float, float]]) -> SignType:
        """
        Recognize a sign from hand landmarks using geometric rules.
        
        Args:
            landmarks: List of hand landmarks
            
        Returns:
            Recognized SignType
        """
        try:
            if not landmarks or len(landmarks) < 21:
                return SignType.UNKNOWN
            
            # Check for each sign using geometric rules
            if self._is_i_love_you(landmarks):
                return SignType.I_LOVE_YOU
            elif self._is_thumbs_up_or_hello(landmarks):
                return SignType.HELLO
            elif self._is_thank_you(landmarks):
                return SignType.THANK_YOU
            elif self._is_yes(landmarks):
                return SignType.YES
            elif self._is_no(landmarks):
                return SignType.NO
            elif self._is_help(landmarks):
                return SignType.HELP
            elif self._is_good_morning(landmarks):
                return SignType.GOOD_MORNING
            elif self._is_sorry(landmarks):
                return SignType.SORRY
            elif self._is_please(landmarks):
                return SignType.PLEASE
            elif self._is_goodbye(landmarks):
                return SignType.GOODBYE
            else:
                return SignType.UNKNOWN
                
        except Exception as e:
            logger.error(f"Error in sign recognition: {e}")
            return SignType.UNKNOWN
    
    def _is_i_love_you(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'I Love You' sign (thumb, index, and pinky extended; middle and ring folded).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if I Love You sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            
            return (fingers.get('thumb_extended', False) and
                    fingers.get('index_extended', False) and
                    not fingers.get('middle_extended', True) and
                    not fingers.get('ring_extended', True) and
                    fingers.get('pinky_extended', False))
        except Exception as e:
            logger.error(f"Error in _is_i_love_you: {e}")
            return False
    
    def _is_thumbs_up_or_hello(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'Hello' sign (palm open, fingers extended, hand waving position).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if Hello sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            position = self.analyzer.get_hand_position(landmarks)
            
            # Hello: Most fingers extended, hand at upper-middle area
            most_fingers_extended = sum([
                fingers.get('index_extended', False),
                fingers.get('middle_extended', False),
                fingers.get('ring_extended', False),
                fingers.get('pinky_extended', False)
            ]) >= 3
            
            return most_fingers_extended and position.get('hand_direction') == 'up'
        except Exception as e:
            logger.error(f"Error in _is_thumbs_up_or_hello: {e}")
            return False
    
    def _is_thank_you(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'Thank You' sign (open hand, palm facing up, moved from mouth downward).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if Thank You sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            position = self.analyzer.get_hand_position(landmarks)
            
            # Thank you: All fingers extended, palm up, hand at face level
            all_fingers_extended = all([
                fingers.get('thumb_extended', False),
                fingers.get('index_extended', False),
                fingers.get('middle_extended', False),
                fingers.get('ring_extended', False),
                fingers.get('pinky_extended', False)
            ])
            
            # Check if palm is facing camera (open hand)
            wrist = landmarks[self.analyzer.WRIST]
            fingers_above_wrist = sum([
                landmarks[self.analyzer.INDEX_TIP][1] < wrist[1],
                landmarks[self.analyzer.MIDDLE_TIP][1] < wrist[1],
                landmarks[self.analyzer.RING_TIP][1] < wrist[1],
                landmarks[self.analyzer.PINKY_TIP][1] < wrist[1]
            ]) >= 3
            
            return all_fingers_extended and fingers_above_wrist
        except Exception as e:
            logger.error(f"Error in _is_thank_you: {e}")
            return False
    
    def _is_yes(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'Yes' sign (fist with thumb up, moving up-down or nodding motion).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if Yes sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            
            # Yes: Thumb extended, other fingers curled (closed fist with thumb up)
            thumb_up = fingers.get('thumb_extended', False)
            fingers_curled = not (
                fingers.get('index_extended', False) or
                fingers.get('middle_extended', False) or
                fingers.get('ring_extended', False) or
                fingers.get('pinky_extended', False)
            )
            
            return thumb_up and fingers_curled
        except Exception as e:
            logger.error(f"Error in _is_yes: {e}")
            return False
    
    def _is_no(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'No' sign (index and middle fingers forming V-shape, moving side to side).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if No sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            
            # No: Index and middle extended, other fingers curled, in V-shape
            index_extended = fingers.get('index_extended', False)
            middle_extended = fingers.get('middle_extended', False)
            thumb_curled = not fingers.get('thumb_extended', False)
            
            # Calculate distance between index and middle tips (should be relatively far)
            index_tip = landmarks[self.analyzer.INDEX_TIP]
            middle_tip = landmarks[self.analyzer.MIDDLE_TIP]
            distance = self.analyzer.distance(index_tip, middle_tip)
            
            return (index_extended and middle_extended and thumb_curled and distance > 0.05)
        except Exception as e:
            logger.error(f"Error in _is_no: {e}")
            return False
    
    def _is_help(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'Help' sign (open hand with all fingers extended, palm facing up).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if Help sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            
            # Help: All fingers extended, open palm
            all_extended = all([
                fingers.get('thumb_extended', False),
                fingers.get('index_extended', False),
                fingers.get('middle_extended', False),
                fingers.get('ring_extended', False),
                fingers.get('pinky_extended', False)
            ])
            
            return all_extended
        except Exception as e:
            logger.error(f"Error in _is_help: {e}")
            return False
    
    def _is_good_morning(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'Good Morning' sign (hand wave with open palm, moving side to side).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if Good Morning sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            position = self.analyzer.get_hand_position(landmarks)
            
            # Good Morning: Similar to Hello (waving motion)
            most_fingers_extended = sum([
                fingers.get('index_extended', False),
                fingers.get('middle_extended', False),
                fingers.get('ring_extended', False),
                fingers.get('pinky_extended', False)
            ]) >= 3
            
            # Hand should be at upper position (waving height)
            hand_high = position.get('wrist_y', 1) < 0.4
            
            return most_fingers_extended and hand_high
        except Exception as e:
            logger.error(f"Error in _is_good_morning: {e}")
            return False
    
    def _is_sorry(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'Sorry' sign (closed fist touching chest area or hand over heart motion).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if Sorry sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            
            # Sorry: Closed fist (all fingers curled)
            all_curled = not any([
                fingers.get('thumb_extended', False),
                fingers.get('index_extended', False),
                fingers.get('middle_extended', False),
                fingers.get('ring_extended', False),
                fingers.get('pinky_extended', False)
            ])
            
            # Hand should be at center/chest level
            wrist = landmarks[self.analyzer.WRIST]
            at_chest = 0.3 < wrist[0] < 0.7 and 0.3 < wrist[1] < 0.7
            
            return all_curled and at_chest
        except Exception as e:
            logger.error(f"Error in _is_sorry: {e}")
            return False
    
    def _is_please(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'Please' sign (open hand, palm facing up, hand moving upward).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if Please sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            position = self.analyzer.get_hand_position(landmarks)
            
            # Please: All fingers extended (open palm), hand moving upward
            all_extended = all([
                fingers.get('thumb_extended', False),
                fingers.get('index_extended', False),
                fingers.get('middle_extended', False),
                fingers.get('ring_extended', False),
                fingers.get('pinky_extended', False)
            ])
            
            hand_moving_up = position.get('hand_direction') == 'up'
            
            return all_extended and hand_moving_up
        except Exception as e:
            logger.error(f"Error in _is_please: {e}")
            return False
    
    def _is_goodbye(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """
        Recognize 'Goodbye' sign (waving motion, fingers extended, hand at shoulder level).
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            True if Goodbye sign is detected
        """
        try:
            fingers = self.analyzer.analyze_finger_configuration(landmarks)
            position = self.analyzer.get_hand_position(landmarks)
            
            # Goodbye: Most fingers extended (waving hand)
            most_fingers_extended = sum([
                fingers.get('index_extended', False),
                fingers.get('middle_extended', False),
                fingers.get('ring_extended', False),
                fingers.get('pinky_extended', False)
            ]) >= 3
            
            # Hand at shoulder/upper arm level
            hand_at_shoulder = 0.2 < position.get('wrist_y', 1) < 0.5
            
            return most_fingers_extended and hand_at_shoulder
        except Exception as e:
            logger.error(f"Error in _is_goodbye: {e}")
            return False
    
    def close(self):
        """Clean up resources used by the recognizer."""
        try:
            if self.hands:
                self.hands.close()
            logger.info("SignRecognizer closed successfully")
        except Exception as e:
            logger.error(f"Error closing SignRecognizer: {e}")


# For backward compatibility with potential imports
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not imported at module level")


if __name__ == "__main__":
    """
    Example usage of the sign detector.
    Requires OpenCV to be installed for video capture.
    """
    import cv2
    
    try:
        # Initialize recognizer
        recognizer = SignRecognizer()
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            exit(1)
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Flip frame for selfie-view
            frame = cv2.flip(frame, 1)
            
            # Recognize signs
            left_result, right_result = recognizer.process_frame(frame)
            
            # Display results
            if left_result and left_result.sign_type != SignType.UNKNOWN:
                cv2.putText(frame, f"L: {left_result}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if right_result and right_result.sign_type != SignType.UNKNOWN:
                cv2.putText(frame, f"R: {right_result}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Sign Detector", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        exit(1)
