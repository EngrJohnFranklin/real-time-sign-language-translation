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
from utils.camera_selector import select_camera_index


# Build FSL alphabet gesture reference from SignType
# Import the PROVISIONAL handshape descriptions (see docs/FSL_REFERENCE.md)
try:
    from models.sign_detector import FSL_ALPHABET_HANDSHAPES
except ImportError:
    FSL_ALPHABET_HANDSHAPES = {}


def build_gesture_landmarks():
    """Build the gesture reference from SignType enum and FSL handshapes."""
    gestures = {}
    for sign in SignType:
        if sign == SignType.UNKNOWN:
            continue
        
        label = sign.value
        description = FSL_ALPHABET_HANDSHAPES.get(label, "No description available")
        
        gestures[label] = {
            "description": description,
            "key_features": [
                "Refer to docs/FSL_REFERENCE.md for verified details",
                "Handshape definitions are PROVISIONAL and UNVERIFIED",
            ],
            "tip_positions": {
                "note": "Specific finger positions depend on the handshape",
            },
            "difficulty": 1,
            "hint": f"Form the FSL letter '{label}' handshape",
        }
    
    return gestures


GESTURE_LANDMARKS = build_gesture_landmarks()


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
    camera_index = select_camera_index()
    if camera_index is None:
        print("Error: No camera devices detected")
        return
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
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
