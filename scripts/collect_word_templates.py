"""
Record temporal word template sequences for DTW-based word recognition.

Similar to collect_training_data.py but captures temporal sequences instead of 
single-frame samples. Each recording is a rolling sequence of MediaPipe normalized
landmarks saved as (T, 126) NumPy arrays.

Supports 10 word classes:
- Static: love, i_love_you, thank_you, quiet, no, hello, sorry, yes, me, good
- Static: love, i_love_you, stop

Quality validation:
- Rejects incomplete hands
- Rejects low-confidence detections
- Rejects sequences that are too short
- Validates all frames match the (T, 126) shape

Usage:
    python scripts/collect_word_templates.py [--dry-run] [--word WORD]

Controls:
    SPACE      - Start/stop recording a sequence
    ENTER      - Save current recording and move to next
    N          - Skip to next word
    P          - Go to previous word
    ESC        - Exit (saves metadata)
    
Example:
    python scripts/collect_word_templates.py --dry-run
    python scripts/collect_word_templates.py --word hello
"""

import sys
import os
import json
import pathlib
import argparse
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# Resolve project root
_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add src to path
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from translation.word_template_loader import (  # noqa: E402
    MOTION_BASED_WORDS,
    STATIC_WORDS,
    WORD_CLASSES,
)
from utils.landmark_normalizer import normalize_dual_hand_features  # noqa: E402
from utils.camera_selector import select_camera_index  # noqa: E402

# --- Configuration ---
WORDS = WORD_CLASSES

# Recording parameters
MIN_HAND_CONFIDENCE = 0.7
MIN_SEQUENCE_LENGTH = 10  # frames
MAX_SEQUENCE_LENGTH = 300  # ~10 seconds at 30 fps
OUTPUT_ROOT = _PROJECT_ROOT / "data" / "word_templates"


@dataclass
class RecordingMetadata:
    """Metadata for a recorded word template."""
    word: str
    sample_index: int
    num_frames: int
    motion_based: bool
    timestamp: str
    camera_index: int
    notes: str = ""


class WordTemplateRecorder:
    """Records temporal word template sequences."""

    def __init__(self, camera_index: int = 0, dry_run: bool = False):
        self.camera_index = camera_index
        self.dry_run = dry_run
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_HAND_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(camera_index)
        self.current_word_idx = 0
        self.recording = False
        self.current_sequence: List[np.ndarray] = []

    def _process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], object]:
        """Extract normalized landmarks and MediaPipe results from a frame."""
        try:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None, results
            
            left_lm = None
            right_lm = None
            
            for landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                lm_list = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                if handedness.classification[0].label == "Right":
                    right_lm = lm_list
                else:
                    left_lm = lm_list
            
            return normalize_dual_hand_features(left_lm, right_lm), results
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None, None

    def _save_sequence(self, word: str) -> bool:
        """Save current sequence to disk."""
        if len(self.current_sequence) < MIN_SEQUENCE_LENGTH:
            print(
                f"Sequence too short ({len(self.current_sequence)} < {MIN_SEQUENCE_LENGTH}). "
                f"Discarding."
            )
            self.current_sequence = []
            return False
        
        if self.dry_run:
            print(f"[DRY-RUN] Would save {len(self.current_sequence)} frames for '{word}'")
            self.current_sequence = []
            return True
        
        word_dir = OUTPUT_ROOT / word
        word_dir.mkdir(parents=True, exist_ok=True)
        
        # Find next sample index
        existing_indices = [
            int(sample_path.stem.removeprefix("sample_"))
            for sample_path in word_dir.glob("sample_*.npy")
            if sample_path.stem.removeprefix("sample_").isdigit()
        ]
        next_idx = max(existing_indices, default=0) + 1
        
        # Save sequence
        sequence_array = np.array(self.current_sequence, dtype=np.float32)
        sample_path = word_dir / f"sample_{next_idx:02d}.npy"
        np.save(str(sample_path), sequence_array)
        
        # Save metadata
        metadata = RecordingMetadata(
            word=word,
            sample_index=next_idx,
            num_frames=len(self.current_sequence),
            motion_based=word in MOTION_BASED_WORDS,
            timestamp=datetime.now().isoformat(),
            camera_index=self.camera_index
        )
        
        metadata_path = word_dir / "metadata.json"
        with open(metadata_path, "a") as f:
            f.write(json.dumps(asdict(metadata)) + "\n")
        
        print(f"Saved {len(self.current_sequence)} frames to {sample_path}")
        self.current_sequence = []
        return True

    def run(self, target_word: Optional[str] = None):
        """Run the recording loop."""
        if target_word:
            if target_word not in WORDS:
                print(f"Unknown word: {target_word}")
                return
            self.current_word_idx = WORDS.index(target_word)
        
        print(f"\n{'='*60}")
        print("Word Template Recorder")
        print(f"{'='*60}")
        print(f"Dry-run: {self.dry_run}")
        print(f"\nControls:")
        print("  SPACE - Start/stop recording")
        print("  ENTER - Save and next")
        print("  N - Next word")
        print("  P - Previous word")
        print("  ESC - Exit")
        print(f"\nWords ({len(WORDS)} total):")
        for i, w in enumerate(WORDS):
            marker = " > " if i == self.current_word_idx else "   "
            motion = "[MOTION]" if w in MOTION_BASED_WORDS else "[STATIC]"
            print(f"{marker}{i+1:2d}. {w:15s} {motion}")
        print(f"{'='*60}\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            landmarks, results = self._process_frame(frame)
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Save features only while recording; preview skeletons are always drawn.
            if self.recording:
                if landmarks is not None:
                    self.current_sequence.append(landmarks)
            
            # Draw UI
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                f"Word: {WORDS[self.current_word_idx].upper()}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2
            )
            
            status = "RECORDING" if self.recording else "READY"
            color = (0, 0, 255) if self.recording else (0, 255, 0)
            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )
            
            cv2.putText(
                frame,
                f"Frames: {len(self.current_sequence)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1
            )
            
            cv2.imshow("Word Template Recorder", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                if self.recording:
                    self.recording = False
                else:
                    break
            elif key == ord(" "):  # SPACE
                # If the user starts a fresh recording after stopping a previous take,
                # clear any leftover unsaved frames from the prior attempt.
                if not self.recording and self.current_sequence:
                    print(
                        "Discarding unsaved frames from previous take before starting a new recording."
                    )
                    self.current_sequence = []

                self.recording = not self.recording
                if self.recording:
                    print(f"Started recording '{WORDS[self.current_word_idx]}'...")
                else:
                    print(f"Stopped recording. Frames: {len(self.current_sequence)}")
            elif key == 13:  # ENTER
                self.recording = False
                if self.current_sequence:
                    word = WORDS[self.current_word_idx]
                    self._save_sequence(word)
                    self.current_word_idx = (self.current_word_idx + 1) % len(WORDS)
            elif key == ord("n"):  # N
                self.recording = False
                self.current_sequence = []
                self.current_word_idx = (self.current_word_idx + 1) % len(WORDS)
                print(f"Switched to '{WORDS[self.current_word_idx]}'. Frames reset to 0.")
            elif key == ord("p"):  # P
                self.recording = False
                self.current_sequence = []
                self.current_word_idx = (self.current_word_idx - 1) % len(WORDS)
                print(f"Switched to '{WORDS[self.current_word_idx]}'. Frames reset to 0.")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nRecording session complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Record temporal word template sequences for DTW-based word recognition."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not save files; only simulate recording."
    )
    parser.add_argument(
        "--word",
        type=str,
        help="Start with a specific word instead of 'hello'."
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera index to use instead of interactive selection."
    )
    
    args = parser.parse_args()

    camera_index = args.camera
    if camera_index is None:
        camera_index = select_camera_index()
        if camera_index is None:
            return
    else:
        print(f"Using Camera {camera_index} from --camera.")

    print(f"Initializing Camera {camera_index}...")
    recorder = WordTemplateRecorder(
        camera_index=camera_index,
        dry_run=args.dry_run
    )
    recorder.run(target_word=args.word)


if __name__ == "__main__":
    main()
