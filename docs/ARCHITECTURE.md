# Real-Time Sign Language Translation System Architecture

## 1. System Summary

This system is a desktop, offline, bidirectional translation application with two real-time paths:

1. Sign to Text and Speech
2. Speech to Text and Sign Animation

The application uses a responsive accessibility-focused UI in CustomTkinter and runs processing tasks in background threads to keep the interface responsive.

## 2. Current UI Architecture

The implemented interface in src/ui/gui.py follows this responsive grid layout:

+------------------------------------------------------+
| Live Camera            | Sign Language Avatar        |
| (expandable)           | (expandable)                |
+------------------------+-----------------------------+
| Current Translation    | Speech Status               |
| latest recognized text | Ready / Listening /         |
|                        | Processing / Speaking       |
+------------------------+-----------------------------+
| Start Camera           | Speak                       |
+------------------------------------------------------+

### UI behavior

- Top row has higher weight for larger visual panels.
- Middle row displays latest translation and current speech pipeline state.
- Bottom row has two equal-size action buttons.
- Rounded corners and consistent spacing are applied across panels.

## 3. Runtime Component Architecture

### 3.1 Entry and Orchestration

- src/main.py:
  - Starts the application.
  - Validates and initializes the runtime environment.
  - Launches MainWindow.

- src/ui/gui.py (MainWindow):
  - Central orchestrator for all modules.
  - Wires camera, sign recognition, speech handler, and video player.
  - Updates UI state from callbacks.

### 3.2 Sign Recognition Path

- src/ui/gui.py (CameraPanel):
  - Captures webcam frames in a background thread.
  - Calls SignRecognizer on each frame.
  - Applies temporal smoothing before emitting stable sign callbacks.

- src/models/sign_detector.py (SignRecognizer):
  - Uses MediaPipe hand landmarks.
  - Applies geometric rules to classify supported signs.
  - Returns per-hand sign result and confidence.

- src/translation/sign_to_text.py (SignToTextConverter):
  - Deduplicates repeated sign predictions.
  - Produces normalized text tokens.

- MainWindow UI update:
  - Updates Current Translation with latest recognized sign text.
  - Triggers async text-to-speech for confirmed sign text.

### 3.3 Speech Recognition Path

- src/translation/speech_handler.py (SpeechHandler + VoskSpeechRecognizer):
  - Listens to microphone audio offline via Vosk.
  - Emits partial and final recognition results.

- MainWindow speech flow:
  - Listening state: Speech Status is set to Listening.
  - Final result: Current Translation is updated.
  - Processing state: Speech Status is set to Processing.
  - Recognized text is sent to video playback queue for sign animation.
  - Status returns to Ready after processing transition.

### 3.4 Sign Animation Path

- src/ui/video_player.py (VideoPlayerPanel):
  - Maps text to video clips.
  - Uses whole-word clip matching first.
  - Falls back to letter-level clips when needed.
  - Plays queued clips sequentially.

- MainWindow UI display:
  - Receives frame callbacks.
  - Renders animation frames in Sign Language Avatar panel.

## 4. Data and Control Flow

### 4.1 Sign to Speech pipeline

1. Camera frame captured
2. MediaPipe landmark extraction
3. Sign classification
4. Temporal stability filtering
5. Deduplicated text output
6. Current Translation UI update
7. Async speech synthesis

### 4.2 Speech to Sign pipeline

1. Microphone capture
2. Vosk partial and final recognition
3. Final text normalization
4. Current Translation UI update
5. Speech Status set to Processing
6. Text converted to sign video sequence
7. Avatar playback in UI

## 5. Threading Model

Main threads used by the app:

- UI thread: CustomTkinter event loop and all widget updates
- Camera thread: continuous frame capture and sign inference
- Speech recognition thread: continuous Vosk listening
- Video queue worker thread: sequential animation playback
- TTS thread: async speech synthesis

Thread-safety approach:

- Background callbacks are marshaled to UI thread using after(...).
- Queue-based sequencing is used for video playback.
- Camera loop uses controlled start and stop lifecycle management.

## 6. Current Functional Scope

Implemented capabilities:

- Offline sign recognition from webcam
- Offline speech recognition from microphone
- Current Translation panel with latest result
- Speech Status panel with dynamic state label
- Sign avatar playback from recognized speech
- Auto speech output for recognized sign text

UI-visible speech status values supported in the UI layer:

- Ready
- Listening
- Processing
- Speaking

## 7. Deployment and Environment Notes

- Target platform in current usage: Windows desktop
- Runtime: Python 3.8+
- Hardware dependencies: webcam and microphone
- Optional content: local sign animation video files under videos

## 8. Architectural Constraints

- Backend recognition logic should remain independent from UI layout concerns.
- UI can be redesigned without altering core model and translation modules.
- Offline-first behavior is maintained for both speech recognition and sign recognition.
- New features should preserve responsiveness by avoiding long work on the UI thread.
