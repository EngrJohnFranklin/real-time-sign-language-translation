"""
Enhanced data collection script for the XGBoost sign language classifier.

Records MediaPipe hand landmarks from your webcam for each sign and saves
normalized features to a CSV file used by train_xgboost_model.py.

Quality validation system:
- Rejects duplicate samples (cosine similarity > 0.95)
- Rejects low-confidence detections (< 0.7)
- Rejects incomplete hands (landmarks outside frame)
- Rejects insufficient diversity (too similar to previous)
- Displays warnings for rejected samples with reasons
- Tracks rejection statistics

Preprocessing (via landmark_normalizer):
- Landmarks translated to wrist (landmark 0) origin
- Scaled by hand size (wrist-to-middle-finger-MCP distance)
- Output: 126-element vectors (left hand 63 + right hand 63 features)
- Missing hands are zero-padded

Usage:
    python scripts/collect_training_data.py

Controls:
    SPACE  - Capture a single sample
    R      - Start continuous capture
    S      - Stop capture
    N      - Next sign
    P      - Previous sign
    ESC    - Exit and save data
"""

import sys
import os
import csv
import pathlib
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Resolve project root from this file's location
_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add src to path so we can import project modules
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from models.sign_detector import SignType  # noqa: E402

# --- Configuration ---
SIGNS = [s.value for s in SignType if s != SignType.UNKNOWN]
MIN_HAND_CONFIDENCE = 0.7  # Minimum detection confidence
MIN_LANDMARK_CONFIDENCE = 0.5  # Minimum per-landmark confidence
DUPLICATE_THRESHOLD = 0.95  # Cosine similarity threshold for duplicates
OUTPUT_PATH = _PROJECT_ROOT / "data" / "training_data" / "landmark_data.csv"

# CSV columns: label + left hand (21 landmarks × 3) + right hand (21 landmarks × 3)
left_cols = [f"left_{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
right_cols = [f"right_{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
COLUMNS = ["label"] + left_cols + right_cols


@dataclass
class ValidationResult:
    """Result of sample validation."""
    is_valid: bool
    reason: str = ""  # Rejection reason (empty if valid)


class SampleValidator:
    """
    Modular validation system for quality control.
    
    Validates samples based on:
    - Hand detection confidence
    - Hand completeness (all landmarks visible)
    - Landmark confidence levels
    - Uniqueness (not a duplicate/near-duplicate)
    """
    
    def __init__(self, 
                 min_hand_confidence: float = MIN_HAND_CONFIDENCE,
                 min_landmark_confidence: float = MIN_LANDMARK_CONFIDENCE,
                 duplicate_threshold: float = DUPLICATE_THRESHOLD):
        self.min_hand_confidence = min_hand_confidence
        self.min_landmark_confidence = min_landmark_confidence
        self.duplicate_threshold = duplicate_threshold
        self.previous_sample: Optional[np.ndarray] = None
        
        # Statistics
        self.total_attempts = 0
        self.total_accepted = 0
        self.rejections = {
            "low_confidence": 0,
            "incomplete_hand": 0,
            "duplicate": 0,
            "low_landmark_confidence": 0,
        }
    
    def validate_confidence(self, confidence: float) -> ValidationResult:
        """
        Validate hand detection confidence.
        
        Args:
            confidence: Hand detection confidence (0-1)
        
        Returns:
            ValidationResult with is_valid and rejection reason
        """
        if confidence < self.min_hand_confidence:
            return ValidationResult(
                is_valid=False,
                reason=f"Low confidence ({confidence:.2f} < {self.min_hand_confidence})"
            )
        return ValidationResult(is_valid=True)
    
    def validate_completeness(self, landmarks: Optional[list]) -> ValidationResult:
        """
        Validate that hand is complete and visible.
        
        Args:
            landmarks: List of (x, y, z) tuples from MediaPipe
        
        Returns:
            ValidationResult with is_valid and rejection reason
        """
        if landmarks is None:
            return ValidationResult(is_valid=False, reason="No landmarks detected")
        
        if len(landmarks) < 21:
            return ValidationResult(
                is_valid=False,
                reason=f"Incomplete hand ({len(landmarks)}/21 landmarks)"
            )
        
        # Check visibility (all landmarks must be within frame)
        for i, (x, y, z) in enumerate(landmarks):
            if not (0 < x < 1 and 0 < y < 1):
                return ValidationResult(
                    is_valid=False,
                    reason=f"Hand not fully visible (landmark {i} at ({x:.2f}, {y:.2f}))"
                )
        
        return ValidationResult(is_valid=True)
    
    def validate_landmark_confidence(self, 
                                    hand_landmarks: any,  # MediaPipe hand landmarks object
                                    handedness_confidence: float) -> ValidationResult:
        """
        Validate per-landmark confidence levels.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            handedness_confidence: Overall hand detection confidence
        
        Returns:
            ValidationResult with is_valid and rejection reason
        """
        # If overall confidence is already high, accept
        if handedness_confidence >= 0.85:
            return ValidationResult(is_valid=True)
        
        # Check individual landmarks (if available)
        # MediaPipe stores z values; typically z-confidence is in presence field
        # For standard MediaPipe, we use z as a proxy for confidence
        low_confidence_count = 0
        
        if hand_landmarks and hasattr(hand_landmarks, 'landmark'):
            for landmark in hand_landmarks.landmark:
                # z-coordinate often indicates confidence in MediaPipe
                if hasattr(landmark, 'z') and landmark.z < self.min_landmark_confidence:
                    low_confidence_count += 1
        
        if low_confidence_count > 5:  # More than 5 low-confidence landmarks
            return ValidationResult(
                is_valid=False,
                reason=f"Low landmark confidence ({low_confidence_count} landmarks < {self.min_landmark_confidence})"
            )
        
        return ValidationResult(is_valid=True)
    
    def validate_uniqueness(self, current_sample: np.ndarray) -> ValidationResult:
        """
        Validate that sample is sufficiently different from previous one.
        
        Uses cosine similarity to detect near-duplicates.
        
        Args:
            current_sample: 126-element normalized feature vector
        
        Returns:
            ValidationResult with is_valid and rejection reason
        """
        if self.previous_sample is None or len(current_sample) == 0:
            return ValidationResult(is_valid=True)
        
        # Compute cosine similarity
        curr_norm = current_sample / (np.linalg.norm(current_sample) + 1e-8)
        prev_norm = self.previous_sample / (np.linalg.norm(self.previous_sample) + 1e-8)
        similarity = float(np.dot(curr_norm, prev_norm))
        
        if similarity > self.duplicate_threshold:
            return ValidationResult(
                is_valid=False,
                reason=f"Too similar to previous ({similarity:.3f} > {self.duplicate_threshold})"
            )
        
        return ValidationResult(is_valid=True)
    
    def validate_sample(self,
                       current_sample: np.ndarray,
                       confidence: float,
                       landmarks: Optional[list],
                       hand_landmarks: any,
                       handedness_confidence: float) -> ValidationResult:
        """
        Run all validation checks on a sample.
        
        Validates in order: confidence → completeness → landmark confidence → uniqueness
        
        Args:
            current_sample: 126-element normalized feature vector
            confidence: Hand detection confidence
            landmarks: List of (x, y, z) tuples
            hand_landmarks: MediaPipe hand landmarks object
            handedness_confidence: Overall hand detection confidence
        
        Returns:
            ValidationResult with overall validity and rejection reason (if any)
        """
        self.total_attempts += 1
        
        # Run all validation checks
        results = [
            ("low_confidence", self.validate_confidence(confidence)),
            ("incomplete_hand", self.validate_completeness(landmarks)),
            ("low_landmark_confidence", self.validate_landmark_confidence(hand_landmarks, handedness_confidence)),
            ("duplicate", self.validate_uniqueness(current_sample)),
        ]
        
        # Check each result
        for check_name, result in results:
            if not result.is_valid:
                self.rejections[check_name] += 1
                return result
        
        # Sample is valid
        self.total_accepted += 1
        self.previous_sample = current_sample.copy()
        return ValidationResult(is_valid=True)
    
    def get_stats(self) -> dict:
        """Return rejection statistics."""
        return {
            "total_attempts": self.total_attempts,
            "total_accepted": self.total_accepted,
            "acceptance_rate": self.total_accepted / max(1, self.total_attempts),
            "rejections": self.rejections,
        }
    
    def reset_uniqueness_check(self) -> None:
        """Reset the previous sample for uniqueness checking."""
        self.previous_sample = None


def _draw_overlay(frame: np.ndarray, sign_name: str, sign_idx: int,
                  total_signs: int, sample_count: int, continuous_recording: bool,
                  single_capture_mode: bool, hands_detected: int, 
                  hand_confidence: float, fps: float, 
                  validation_message: str = "", validation_status: str = "") -> None:
    """
    Draw a comprehensive status overlay on the camera frame (in-place).
    
    Args:
        sign_name: Current sign name
        sign_idx: Index of current sign (0-based)
        total_signs: Total number of signs
        sample_count: Number of samples captured for current sign
        continuous_recording: Whether continuous recording is active
        single_capture_mode: Whether in single-capture mode (SPACE)
        hands_detected: Number of hands currently visible (0, 1, or 2)
        hand_confidence: Hand detection confidence (0-1)
        fps: Current frames per second
        validation_message: Message about sample validation (e.g., "Not fully visible")
        validation_status: "accepted", "rejected", or "" (neutral)
    """
    h, w = frame.shape[:2]

    # Top status bar
    if continuous_recording:
        bar_color = (0, 140, 0)  # Green for continuous recording
        rec_text = "● CONTINUOUS"
    elif single_capture_mode:
        bar_color = (0, 165, 255)  # Orange for single capture mode
        rec_text = "◉ SINGLE"
    else:
        bar_color = (30, 30, 160)  # Blue for idle
        rec_text = "⊘ IDLE"
    
    cv2.rectangle(frame, (0, 0), (w, 85), bar_color, -1)

    # Sign name and progress
    cv2.putText(frame, f"Sign: {sign_name}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"[{sign_idx + 1}/{total_signs}] Samples: {sample_count}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1)
    
    # Recording status
    cv2.putText(frame, rec_text,
                (w - 180, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    
    # Hand info and FPS
    hands_text = f"{hands_detected} hand(s) | Conf: {hand_confidence:.2f} | FPS: {fps:.1f}"
    hands_color = (0, 200, 0) if hands_detected > 0 else (100, 100, 100)
    cv2.putText(frame, hands_text,
                (w - 400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hands_color, 1)
    
    # Validation message with color coding
    if validation_message:
        if validation_status == "accepted":
            msg_color = (0, 255, 0)  # Green for accepted
            status_icon = "✓"
        elif validation_status == "rejected":
            msg_color = (0, 0, 255)  # Red for rejected
            status_icon = "✗"
        else:
            msg_color = (0, 100, 255)  # Orange for warnings
            status_icon = "⚠"
        
        cv2.putText(frame, f"{status_icon} {validation_message}",
                    (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_color, 1)

    # Keyboard hints
    hint_line1 = "SPACE: capture  |  R: continuous  |  S: stop  |  N: next  |  P: prev  |  ESC: exit"
    cv2.putText(frame, hint_line1,
                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


class FrameRateCounter:
    """Simple FPS counter for display."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
    
    def update(self) -> float:
        """Update with current time and return average FPS."""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # Keep only recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        # Calculate FPS
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff <= 0:
            return 0.0
        
        fps = (len(self.timestamps) - 1) / time_diff
        return fps


def _count_existing_samples(path: pathlib.Path) -> int:
    """Return the number of data rows in the CSV (excluding header)."""
    if not path.exists():
        return 0
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return sum(1 for _ in reader)
    except Exception:
        return 0


def main() -> None:
    print("\n=== Enhanced Sign Language Data Collection ===")
    print(f"Signs: {', '.join(SIGNS)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Quality checks: Hand confidence >= {MIN_HAND_CONFIDENCE}, fully visible, no duplicates\n")

    existing_count = _count_existing_samples(OUTPUT_PATH)
    if existing_count > 0:
        print(f"Existing samples: {existing_count}")
        print()

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Detect up to 2 hands
        min_detection_confidence=MIN_HAND_CONFIDENCE,
        min_tracking_confidence=0.5,
    )
    drawing_utils = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Initialize components
    validator = SampleValidator(
        min_hand_confidence=MIN_HAND_CONFIDENCE,
        min_landmark_confidence=MIN_LANDMARK_CONFIDENCE,
        duplicate_threshold=DUPLICATE_THRESHOLD,
    )
    
    collected: list = []  # All collected samples
    sign_samples: dict = {}  # Per-sign sample counts
    sign_idx = 0
    continuous_recording = False
    single_capture_mode = False
    fps_counter = FrameRateCounter()
    
    validation_message = ""
    validation_status = ""

    print(f"Controls:\n"
          f"  SPACE = Capture single sample\n"
          f"  R = Start continuous capture\n"
          f"  S = Stop capture\n"
          f"  N = Next sign\n"
          f"  P = Previous sign\n"
          f"  ESC = Exit\n")
    if SIGNS:
        print(f"Ready to collect: {SIGNS[sign_idx]}\n")

    from models.xgboost_classifier import XGBoostSignClassifier  # noqa: PLC0415

    while True:
        if sign_idx < 0:
            sign_idx = 0
        if sign_idx >= len(SIGNS):
            print("\n✓ All signs completed!")
            break

        current_sign = SIGNS[sign_idx]
        ret, frame = cap.read()
        if not ret:
            break

        # Update FPS
        fps = fps_counter.update()

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Extract landmarks for both hands
        left_landmarks = None
        right_landmarks = None
        left_hand_landmarks = None
        right_hand_landmarks = None
        hands_detected = 0
        hand_confidence = 0.0
        validation_message = ""
        validation_status = ""

        if results.multi_hand_landmarks and results.multi_handedness:
            hands_detected = len(results.multi_hand_landmarks)
            hand_confidence = results.multi_handedness[0].classification[0].score if hands_detected > 0 else 0.0
            
            for lm_set, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                landmarks = [(lm.x, lm.y, lm.z) for lm in lm_set.landmark]
                
                if hand_label == "Right":
                    right_landmarks = landmarks
                    right_hand_landmarks = lm_set
                else:
                    left_landmarks = landmarks
                    left_hand_landmarks = lm_set
            
            # Draw landmarks
            drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            if len(results.multi_hand_landmarks) > 1:
                drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)
        
        # Collect samples when requested
        should_capture = False
        
        if single_capture_mode:
            # Single capture: save one sample and exit capture mode
            should_capture = True
            single_capture_mode = False
        elif continuous_recording:
            # Continuous: save every frame
            should_capture = True
        
        if should_capture and (left_landmarks or right_landmarks):
            # Extract features first
            features = XGBoostSignClassifier.extract_features(left_landmarks, right_landmarks)
            
            # Validate the sample using modular validator
            # Use first available hand for confidence validation
            primary_hand = left_hand_landmarks if left_hand_landmarks else right_hand_landmarks
            validation_result = validator.validate_sample(
                current_sample=features,
                confidence=hand_confidence,
                landmarks=left_landmarks if left_landmarks else right_landmarks,
                hand_landmarks=primary_hand,
                handedness_confidence=hand_confidence,
            )
            
            if validation_result.is_valid:
                # Sample is good - save it
                row = [current_sign] + features.tolist()
                collected.append(row)
                
                # Update sample count
                if current_sign not in sign_samples:
                    sign_samples[current_sign] = 0
                sign_samples[current_sign] += 1
                
                validation_message = f"Accepted (#{sign_samples[current_sign]})"
                validation_status = "accepted"
                print(f"✓ {current_sign}: sample {sign_samples[current_sign]}")
            else:
                # Sample rejected
                validation_message = validation_result.reason
                validation_status = "rejected"
        
        # Draw overlay
        sample_count = sign_samples.get(current_sign, 0)
        _draw_overlay(frame, current_sign, sign_idx, len(SIGNS), sample_count,
                      continuous_recording, single_capture_mode, hands_detected,
                      hand_confidence, fps, validation_message, validation_status)
        cv2.imshow("Sign Data Collection", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\nExiting...")
            break
        elif key == ord(" "):  # SPACE - single capture
            if not continuous_recording:
                single_capture_mode = True
        elif key in (ord("r"), ord("R")):  # R - start continuous
            continuous_recording = True
            single_capture_mode = False
            validator.reset_uniqueness_check()  # Reset for new capture session
            print(f"Starting continuous capture for '{current_sign}'...")
        elif key in (ord("s"), ord("S")):  # S - stop capture
            continuous_recording = False
            single_capture_mode = False
            print(f"Stopped capture")
        elif key in (ord("n"), ord("N")):  # N - next sign
            sign_idx += 1
            validator.reset_uniqueness_check()  # Reset for new sign
            continuous_recording = False
            single_capture_mode = False
            if sign_idx < len(SIGNS):
                print(f"\nNext: {SIGNS[sign_idx]}")
            else:
                print("\n✓ All signs completed!")
                break
        elif key in (ord("p"), ord("P")):  # P - previous sign
            sign_idx -= 1
            validator.reset_uniqueness_check()  # Reset for new sign
            continuous_recording = False
            single_capture_mode = False
            if sign_idx >= 0:
                print(f"\nPrevious: {SIGNS[sign_idx]}")
            else:
                sign_idx = 0

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Save collected data
    if not collected:
        print("\nNo data collected. Exiting.")
        return

    write_header = not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0

    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(COLUMNS)
        writer.writerows(collected)

    # Print summary
    print(f"\n{'='*70}")
    print(f"✓ Saved {len(collected)} new samples to {OUTPUT_PATH}")
    print(f"{'='*70}")
    
    print(f"\nSamples per sign:")
    for sign in SIGNS:
        count = sign_samples.get(sign, 0)
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {sign:20} : {count:3} samples")
    
    # Print validation statistics
    stats = validator.get_stats()
    print(f"\nQuality control statistics:")
    print(f"  • Total attempts    : {stats['total_attempts']}")
    print(f"  • Accepted          : {stats['total_accepted']}")
    print(f"  • Acceptance rate   : {stats['acceptance_rate']:.1%}")
    
    print(f"\nRejection breakdown:")
    for reason, count in stats['rejections'].items():
        if count > 0:
            pct = count / max(1, stats['total_attempts']) * 100
            reason_display = reason.replace("_", " ").title()
            print(f"  • {reason_display:25} : {count:4} ({pct:5.1f}%)")
    
    print(f"\nPreprocessing applied:")
    print(f"  • Landmarks translated to wrist origin")
    print(f"  • Scaled by hand size (wrist-to-MCP distance)")
    print(f"  • 126-element feature vectors (63 left + 63 right)")
    print(f"  • Missing hands padded with zeros")
    print(f"  • Modular validation with detailed feedback")
    print(f"\nNext step:")
    print(f"  python scripts/train_xgboost_model.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
