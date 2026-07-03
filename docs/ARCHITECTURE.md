# Real-Time Sign Language Translation System - Architecture

## System Overview

This is a real-time bidirectional sign language translation system that converts between sign language (via camera) and spoken language (via microphone/speakers). The system operates entirely offline with no internet dependency.

## Core Workflows

### Workflow 1: Camera → Signs → Text → Speech
1. **Camera Input**: Live webcam feed (640x480 @ 30 FPS)
2. **Sign Detection**: MediaPipe hand landmark detection + geometric gesture recognition
3. **Temporal Smoothing**: 4-frame stability window with confidence threshold (0.75) to reduce flickering
4. **Text Generation**: Confirmed signs deduplicated and converted to readable text
5. **Speech Output**: Text automatically converted to speech via pyttsx3

### Workflow 2: Speech → Text → Signs → Animation
1. **Microphone Input**: Real-time audio capture (16kHz, Vosk recognition)
2. **Speech Recognition**: Offline Vosk engine supports English, Filipino, Spanish, French, German
3. **Text Display**: Final recognized text appended to speech textbox
4. **Sign Animation**: Text-to-sign conversion with whole-word then alphabet fallback
5. **Video Playback**: Sequential queue-based playback of sign language demonstration videos

## Phase 12: Code Quality Improvements & Bug Fixes

### Critical Issues Resolved

#### 1. Missing cv2 Import in SignRecognizer
**Issue**: The `cv2` module was only imported in the `__main__` block, causing ImportError when sign_detector.py was imported as a library.
**Fix**: Moved `import cv2` to module-level imports (line 7).
**Impact**: SignRecognizer now loads without errors.

#### 2. Thread Race Condition in CameraPanel
**Issue**: Check-then-act (TOCTOU) race on `self.camera_thread` between test and join(), could cause AttributeError in concurrent scenarios.
**Fix**: 
- Added `self.thread_lock = threading.Lock()` to CameraPanel.__init__
- Wrapped all camera_thread access with lock in start_camera() and stop_camera()
**Impact**: Thread-safe camera startup/shutdown; eliminates race condition.

#### 3. Event Object Allocation Every Frame
**Issue**: `threading.Event().wait(delay)` called ~30 times/second creates unnecessary objects.
**Fix**: 
- Added `import time` 
- Replaced `threading.Event().wait(frame_delay)` with `time.sleep(frame_delay)` in two locations (playback loop, queue wait)
**Impact**: Eliminated ~30 object allocations/second; reduced GC pressure.

#### 4. Unused Imports Cleanup
**Issue**: `from pathlib import Path` imported but never used; `re` import kept (used in regex word splitting).
**Fix**: Removed unused `from pathlib import Path` from video_player.py.
**Impact**: Cleaner code; reduced import overhead.

#### 5. Missing Frame Validation
**Issue**: Potential None frame passed to _update_frame_display() could cause crashes.
**Fix**: 
- Added None check before _update_frame_display() call (line 235)
- Added frame size validation inside _update_frame_display() (lines 374-376)
- Early return if frame is invalid
**Impact**: Defensive programming prevents display crashes.

### Performance Optimizations
- Thread lock prevents busy-waiting on camera state
- time.sleep() more efficient than Event().wait() for frame delays
- Cleaner import list reduces module load time

### Testing Status
✅ All syntax checks pass  
✅ SignRecognizer imports without errors  
✅ VideoPlayer imports without errors  
✅ CameraPanel thread safety verified  
✅ Both workflows (Camera→Speech, Microphone→Animation) remain functional  

## Technical Architecture

### Module Structure

```
src/
├── main.py                          # Entry point, environment setup, dependency validation
├── models/
│   ├── sign_detector.py            # MediaPipe-based hand detection + geometric rules (10 signs)
│   ├── gesture_recognizer.py       # ML model management (stub for future expansion)
│   ├── model_loader.py             # Model downloading and caching
│   └── inference.py                # Inference pipeline wrapper
├── camera/
│   ├── camera_handler.py           # Webcam capture management (stub - use OpenCV directly)
│   └── frame_processor.py          # Frame resizing and preprocessing (stub)
├── translation/
│   ├── sign_to_text.py             # SignToTextConverter: confirmed predictions → deduplicated text
│   ├── language_mapper.py          # Sign-to-language mapping (optional)
│   ├── text_to_speech.py           # Legacy TTS wrapper (integrated into speech_handler.py)
│   └── speech_handler.py           # Unified Vosk (speech-to-text) + pyttsx3 (text-to-speech)
├── ui/
│   ├── gui.py                      # MainWindow + CameraPanel - orchestrates all workflows
│   ├── main_window.py              # Window layout definition (legacy, merged into gui.py)
│   ├── video_player.py             # VideoPlayer + VideoPlayerPanel - sign animation playback
│   ├── widgets.py                  # Custom UI widgets (optional)
│   └── styles.py                   # Theme configuration (optional)
├── database/
│   ├── db_manager.py               # Database CRUD operations (optional persistence)
│   └── models.py                   # ORM schemas (optional)
└── utils/
    ├── config.py                   # Configuration file loader
    ├── logger.py                   # Logging setup
    ├── constants.py                # Application constants
    └── validators.py               # Input validation

tests/
├── conftest.py                     # Pytest fixtures and setup
├── test_*.py                       # Unit tests for each module
└── integration/
    └── test_end_to_end.py         # Full pipeline integration tests
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Real-Time Workflows                       │
└─────────────────────────────────────────────────────────────┘

WORKFLOW 1 (Camera → Speech):
  Webcam (640x480)
    ↓
  MediaPipe Hand Detection
    ↓
  Landmark Analysis (Geometric Rules)
    ↓
  Temporal Smoothing (4-frame window)
    ↓
  Confidence Threshold Filter (0.75)
    ↓
  Recognized Sign Callback
    ↓
  SignToTextConverter (Deduplication)
    ↓
  Text Textbox Display
    ↓
  pyttsx3 TTS (Async Thread)
    ↓
  Speaker Output

WORKFLOW 2 (Microphone → Animation):
  Microphone (16kHz, Mono)
    ↓
  Vosk Speech Recognition (Offline)
    ↓
  Parse JSON Result (Partial + Final)
    ↓
  Speech Text Callback
    ↓
  Speech Textbox Display
    ↓
  Text-to-Sign Pipeline
    ├─ Split text into words/characters
    ├─ Match to video files (whole-word priority)
    └─ Alphabet fallback for unmapped words
    ↓
  VideoPlayerPanel Queue
    ├─ Persistent Queue (FIFO)
    ├─ Background Queue Worker Thread
    └─ Sequential Playback (one video at a time)
    ↓
  Video Display Panel
```

### Key Components

#### 1. CameraPanel (gui.py)
- **Purpose**: Live camera feed with hand landmark overlay
- **Threading**: Background thread (_camera_loop) reads frames continuously
- **Detection**: SignRecognizer.process_frame() returns left/right hand results
- **Smoothing**: 
  - Left/right prediction history (deque, maxlen=4 for STABILITY_FRAMES)
  - Hold counter (LABEL_HOLD_FRAMES=6) for temporal hysteresis
  - Only trigger callbacks on new transitions (is_newly_accepted)
- **Visualization**: Hand landmarks drawn as circles + skeleton connections

#### 2. SignRecognizer (models/sign_detector.py)
- **Recognition Method**: MediaPipe landmarks + geometric hand analysis
- **Supported Signs** (10 total):
  1. Hello - wave motion detection
  2. Thank You - vertical hand motion
  3. Yes/No - nodding/shaking
  4. Help - hands up
  5. Good Morning - salute-like gesture
  6. Sorry - chest contact
  7. Please - sweeping motion
  8. Goodbye - hand wave down
  9. I Love You - three-finger gesture
  10. Unknown - no match or below confidence
- **Output**: SignResult (sign_type, confidence, hand_side, landmarks)

#### 3. SpeechHandler (translation/speech_handler.py)
- **Two Engines**:
  - **VoskSpeechRecognizer**: Offline speech-to-text
    - Supports: English, Filipino (with fallback), Spanish, French, German
    - Audio: PyAudio capture (16kHz, 4096 chunk size)
    - Parsing: Handles Vosk JSON format (partial vs final)
    - Silence filtering: Ignores texts < 2 chars (MIN_PARTIAL_CHARS, MIN_FINAL_CHARS)
    - Duplicate suppression: Tracks _last_partial_text, _last_final_text
  - **TextToSpeechEngine**: Offline text-to-speech
    - Library: pyttsx3 (works offline on all platforms)
    - Rate control: 50-300 WPM (default 150)
    - Volume control: 0.0-1.0 (default 0.8)
    - Async: speak_async() spawns daemon thread
- **Threading**: Recognition runs in daemon thread, callbacks marshaled to main UI thread via .after()

#### 4. SignToTextConverter (translation/sign_to_text.py)
- **Function**: Aggregate confirmed sign predictions with deduplication
- **State**: Maintains words list + last_word
- **Logic**: Ignores consecutive duplicates (case-insensitive)
- **Usage**: Receives sign predictions from camera → appends unique words → formats as space-separated text

#### 5. VideoPlayerPanel (ui/video_player.py)
- **Infrastructure**: OpenCV video capture + CustomTkinter image display
- **Playback Queue**:
  - Persistent queue.Queue for FIFO clip sequences
  - Background queue_worker_loop thread
  - Sequential playback (one video at a time)
  - Stop signaling via threading.Event
- **Video Index**: Regex-normalized filenames map to paths (handles underscores, hyphens, spaces)
- **Text-to-Sign Pipeline**:
  - Split input text via regex word boundary
  - Per-word: try whole-word video (e.g., "hello.mp4")
  - Fallback: per-character alphabet videos (e.g., "letter_a.mp4")
  - Enqueue result sequence

#### 6. MainWindow (ui/gui.py)
- **Orchestrator**: Coordinates all modules and workflows
- **Callbacks**:
  - _on_sign_recognized: from CameraPanel → SignToTextConverter → auto-speak
  - _update_speech_text: from SpeechHandler → _play_speech_text_as_signs
  - _play_text_sign_clicked: manual trigger for text-to-sign animation
- **State Management**: Tracks camera running, listening state, speech lines, auto-spoken text

### Threading Model

| Thread | Purpose | Type | Lifecycle |
|--------|---------|------|-----------|
| Main UI Thread | CustomTkinter event loop | Foreground | App lifetime |
| Camera Loop | Frame capture + detection | Daemon | While camera_running=True |
| Speech Recognition | Vosk listening + parsing | Daemon | While is_listening=True |
| Video Queue Worker | Sequential clip playback | Daemon | While queue non-empty or manual stop |
| TTS Audio | pyttsx3 synthesis | Daemon | Per-speak call |

**Synchronization**:
- UI updates from background threads: .after(0, lambda: ui.update())
- Speech result event: threading.Event (efficient compared to polling)
- Queue consumption: queue.Queue (thread-safe FIFO)
- Lock on video worker state: threading.Lock for is_running flag

## Performance Optimizations

1. **Temporal Smoothing**: Reduces jitter (4-frame stability + 6-frame hold)
2. **Confidence Threshold**: 0.75 minimum filters low-confidence false positives
3. **Threading Event**: Replaces polling in speech_handler (no CPU spinning)
4. **Async TTS**: Non-blocking speech output (daemon threads)
5. **Frame Resizing**: Lazy resize only for display (not reprocessed)
6. **Video Index Caching**: Pre-built mapping at startup

## External Dependencies

- **MediaPipe 0.10.0**: Hand pose estimation (21 landmarks per hand)
- **OpenCV 4.8.1.78**: Camera capture + video playback
- **CustomTkinter 5.2.0**: Modern GUI framework
- **Vosk 0.3.32**: Offline speech recognition (Kaldi-based)
- **pyttsx3 2.90**: Cross-platform offline TTS
- **Pillow 10.0.0**: Image format conversion
- **NumPy 1.24.3**: Array operations (MediaPipe landmarks)
- **PyYAML 6.0.1**: Configuration parsing
- **python-dotenv 1.0.0**: Environment variables

## Limitations & Future Improvements

**Current Limitations**:
- Limited to 10 predefined sign gestures (no continuous sign language)
- Video files must be manually provided (not auto-generated)
- No machine learning model training (geometric rules only)
- Single-user, single-camera system

**Future Enhancements**:
1. Expand sign vocabulary via ML model
2. Auto-generate sign language animations (avatar-based)
3. Multi-camera support
4. Database integration for translation history
5. Web interface (Flask/FastAPI + WebSockets)
6. Mobile app port (Android/iOS)
7. Real-time video streaming (RTMP/HLS)

## Installation & Setup

See `docs/SETUP.md` for environment configuration and dependency installation.

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

Run integration tests:
```bash
pytest tests/integration/
```

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues and solutions.
