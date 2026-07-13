#!/usr/bin/env python3
"""
Gesture Verification Tool - Display expected hand landmarks for each gesture.

This tool helps verify that you're performing each gesture correctly by showing
where the hand landmarks should be positioned for optimal classification.

Usage:
    python scripts/verify_gesture_landmarks.py

Controls:
    N - Next gesture
    P - Previous gesture
    SPACE - Show reference information
    ESC - Exit
"""

import sys
import pathlib
import cv2
import numpy as np
from typing import Tuple, List

# Resolve project root
_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from models.sign_detector import SignType


# Expected landmark patterns for each gesture (normalized, wrist at origin)
# Relative positions of key landmarks for each gesture
GESTURE_LANDMARKS = {
    "Closed Fist": {
        "description": "All fingers curled inward - most compact shape",
        "key_features": [
            "All fingertips close to wrist (tight cluster)",
            "No fingers extended",
            "Thumb tucked or resting on hand",
        ],
        "tip_positions": {
            "thumb": "Near wrist (low spread)",
            "index": "Near wrist (low spread)",
            "middle": "Near wrist (low spread)",
            "ring": "Near wrist (low spread)",
            "pinky": "Near wrist (low spread)",
        },
        "difficulty": 1,
        "hint": "Make a tight fist",
    },
    
    "Open Palm": {
        "description": "All fingers fully extended and spread",
        "key_features": [
            "All 5 fingertips far from wrist",
            "Maximum spread between fingers",
            "Palm flat and facing camera",
        ],
        "tip_positions": {
            "thumb": "Far right (extended)",
            "index": "Far forward (extended)",
            "middle": "Far forward (extended)",
            "ring": "Far forward (extended)",
            "pinky": "Far left (extended)",
        },
        "difficulty": 1,
        "hint": "Spread your hand wide like a stop sign",
    },
    
    "Thumbs Up": {
        "description": "Thumb pointing upward, other fingers closed",
        "key_features": [
            "Thumb tip highest among all fingers",
            "Other 4 fingers clustered at base",
            "Creates clear vertical line from thumb",
        ],
        "tip_positions": {
            "thumb": "HIGH (extended upward)",
            "index": "LOW (clustered)",
            "middle": "LOW (clustered)",
            "ring": "LOW (clustered)",
            "pinky": "LOW (clustered)",
        },
        "difficulty": 1,
        "hint": "Point your thumb straight up",
    },
    
    "Thumbs Down": {
        "description": "Thumb pointing downward, other fingers closed",
        "key_features": [
            "Thumb tip lowest among all fingers",
            "Other 4 fingers clustered at top",
            "Mirror image of Thumbs Up",
        ],
        "tip_positions": {
            "thumb": "LOW (extended downward)",
            "index": "HIGH (clustered)",
            "middle": "HIGH (clustered)",
            "ring": "HIGH (clustered)",
            "pinky": "HIGH (clustered)",
        },
        "difficulty": 1,
        "hint": "Point your thumb straight down",
    },
    
    "Index Finger": {
        "description": "Index finger up, all others closed",
        "key_features": [
            "Index finger tip highest",
            "Thumb NOT extended (distinguish from Thumbs Up)",
            "Other fingers clustered with thumb",
        ],
        "tip_positions": {
            "thumb": "LOW (clustered, NOT extended)",
            "index": "HIGH (extended upward)",
            "middle": "LOW (clustered)",
            "ring": "LOW (clustered)",
            "pinky": "LOW (clustered)",
        },
        "difficulty": 2,
        "hint": "Point your index finger up (keep thumb relaxed)",
    },
    
    "Peace Sign": {
        "description": "Index and middle fingers extended with gap",
        "key_features": [
            "Index and middle fingers extended upward",
            "Clear gap/separation between index and middle",
            "Ring and pinky clustered together",
            "Thumb may be slightly extended",
        ],
        "tip_positions": {
            "thumb": "MEDIUM (partially extended or relaxed)",
            "index": "HIGH (extended)",
            "middle": "HIGH (extended)",
            "ring": "LOW (clustered)",
            "pinky": "LOW (clustered)",
        },
        "difficulty": 2,
        "hint": "Make a 'V' with your index and middle fingers",
    },
    
    "OK Sign": {
        "description": "Thumb and index forming circle, other 3 extended",
        "key_features": [
            "Thumb and index tips very close (circle)",
            "Middle, ring, pinky extended upward",
            "Clear separation of the circle from other fingers",
        ],
        "tip_positions": {
            "thumb": "CLOSE to index (forming circle)",
            "index": "CLOSE to thumb (forming circle)",
            "middle": "HIGH (extended)",
            "ring": "HIGH (extended)",
            "pinky": "HIGH (extended)",
        },
        "difficulty": 3,
        "hint": "Make a circle with thumb and index, extend other 3 fingers",
    },
    
    "I Love You": {
        "description": "Thumb, index, pinky extended; middle and ring folded",
        "key_features": [
            "Thumb, index, and pinky extended",
            "Middle and ring fingers folded/closed",
            "Creates distinctive three-point pattern",
        ],
        "tip_positions": {
            "thumb": "Extended (side position)",
            "index": "HIGH (extended)",
            "middle": "LOW (folded - key feature)",
            "ring": "LOW (folded - key feature)",
            "pinky": "Extended (side position)",
        },
        "difficulty": 2,
        "hint": "Index up, middle and ring down, thumb and pinky out",
    },
    
    "Hello": {
        "description": "Thumb and pinky extended, middle three closed",
        "key_features": [
            "Thumb and pinky extended (non-adjacent)",
            "Index, middle, ring fingers closed together",
            "Creates 'hang loose' shape",
        ],
        "tip_positions": {
            "thumb": "Extended (one side)",
            "index": "LOW (clustered)",
            "middle": "LOW (clustered)",
            "ring": "LOW (clustered)",
            "pinky": "Extended (other side)",
        },
        "difficulty": 2,
        "hint": "Extend your thumb and pinky, close the middle three",
    },
    
    "Goodbye": {
        "description": "All 5 fingers extended with gap between middle and ring",
        "key_features": [
            "All five fingers extended",
            "Distinctive gap between middle and ring finger",
            "Gap is the defining feature (critical!)",
        ],
        "tip_positions": {
            "thumb": "Extended",
            "index": "Extended",
            "middle": "Extended (gap on right side)",
            "ring": "Extended (gap on left side)",
            "pinky": "Extended",
        },
        "difficulty": 3,
        "hint": "Extend all fingers with a gap between middle and ring",
    },
}


def draw_gesture_info(frame: np.ndarray, gesture_name: str, gesture_idx: int, total_gestures: int) -> np.ndarray:
    """Draw gesture information on frame."""
    h, w = frame.shape[:2]
    
    # Get gesture data
    gesture_data = GESTURE_LANDMARKS.get(gesture_name, {})
    
    # Background box for text
    cv2.rectangle(frame, (10, 10), (w-10, 250), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (w-10, 250), (0, 255, 0), 2)
    
    # Title
    cv2.putText(frame, f"Gesture {gesture_idx+1}/{total_gestures}: {gesture_name}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # Difficulty indicator
    difficulty = gesture_data.get("difficulty", 1)
    stars = "⭐" * difficulty
    cv2.putText(frame, f"Difficulty: {stars}", 
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
    
    # Description
    description = gesture_data.get("description", "")
    cv2.putText(frame, description[:60], 
                (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Key features
    y = 135
    cv2.putText(frame, "Key Features:", 
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 1)
    y += 25
    
    for feature in gesture_data.get("key_features", [])[:2]:
        cv2.putText(frame, f"• {feature[:50]}", 
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1)
        y += 22
    
    # Controls
    cv2.rectangle(frame, (10, h-40), (w-10, h-5), (50, 50, 50), -1)
    cv2.putText(frame, "N=Next  P=Prev  SPACE=Details  ESC=Exit", 
                (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    return frame


def draw_gesture_details(frame: np.ndarray, gesture_name: str) -> np.ndarray:
    """Draw detailed gesture information on frame."""
    h, w = frame.shape[:2]
    
    gesture_data = GESTURE_LANDMARKS.get(gesture_name, {})
    
    # Background
    cv2.rectangle(frame, (10, 10), (w-10, h-50), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (w-10, h-50), (0, 200, 255), 2)
    
    # Title
    cv2.putText(frame, gesture_name, 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    
    y = 70
    
    # Description
    cv2.putText(frame, gesture_data.get("description", ""), 
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
    y += 35
    
    # Hint
    hint = gesture_data.get("hint", "")
    cv2.putText(frame, f"💡 {hint}", 
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 100), 1)
    y += 35
    
    # Finger positions
    cv2.putText(frame, "Finger Positions:", 
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 1)
    y += 28
    
    tip_positions = gesture_data.get("tip_positions", {})
    for finger, position in tip_positions.items():
        text = f"• {finger.upper():10} → {position}"
        cv2.putText(frame, text, 
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 200, 255), 1)
        y += 25
        if y > h - 60:
            break
    
    return frame


def main():
    """Main loop for gesture verification."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    gestures = list(GESTURE_LANDMARKS.keys())
    gesture_idx = 0
    show_details = False
    
    print("\n=== Gesture Verification Tool ===")
    print(f"Gestures: {len(gestures)}")
    print(f"Controls: N=Next, P=Prev, SPACE=Details, ESC=Exit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        current_gesture = gestures[gesture_idx]
        
        if show_details:
            frame = draw_gesture_details(frame, current_gesture)
        else:
            frame = draw_gesture_info(frame, current_gesture, gesture_idx, len(gestures))
        
        cv2.imshow("Gesture Verification", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('n') or key == ord('N'):
            gesture_idx = (gesture_idx + 1) % len(gestures)
            show_details = False
            print(f"Next: {gestures[gesture_idx]}")
        elif key == ord('p') or key == ord('P'):
            gesture_idx = (gesture_idx - 1) % len(gestures)
            show_details = False
            print(f"Previous: {gestures[gesture_idx]}")
        elif key == ord(' '):
            show_details = not show_details
            print(f"Toggle details: {show_details}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nGesture verification complete!")


if __name__ == "__main__":
    main()
