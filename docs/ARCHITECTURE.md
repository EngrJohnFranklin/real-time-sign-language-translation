# Real-Time Sign Language Translation System — Architecture

## 1. System Summary

A desktop, offline, bidirectional translation application with two real-time workflows:

1. **Sign-to-Speech**: Webcam → MediaPipe → Sign Recognition → Text → TTS
2. **Speech-to-Sign**: Microphone → Vosk STT → Text → Video Animation

The application uses CustomTkinter with a dark modern theme and processes all
camera, audio, and video work in background threads to keep the UI responsive.

---

## 2. Clean Architecture Layer Map

```
┌─────────────────────────────────────────────────────────┐
│  Entry Point:  src/main.py                              │
├─────────────────────────────────────────────────────────┤
│  View Layer                                             │
│    src/ui/main_window.py   ← MainWindow (layout + wiring)│
│    src/ui/panels/camera_panel.py  ← camera display view │
│    src/ui/gui.py           ← backward-compat shim       │
│    src/ui/video_player.py  ← VideoPlayer / Panel (unchanged)│
├─────────────────────────────────────────────────────────┤
│  Controller Layer                                       │
│    src/controllers/camera_controller.py                 │
│    src/controllers/speech_controller.py                 │
│    src/controllers/sign_controller.py                   │
├─────────────────────────────────────────────────────────┤
│  Application Service Layer                              │
│    src/services/camera_service.py  (OpenCV capture)     │
│    src/services/recognition_service.py  (temporal smooth)│
│    src/services/speech_service.py  (STT + TTS facade)   │
│    src/services/video_service.py  (playback facade)     │
├─────────────────────────────────────────────────────────┤
│  State Management                                       │
│    src/app/app_state.py  (AppState — thread-safe)       │
├─────────────────────────────────────────────────────────┤
│  Domain Layer                                           │
│    src/models/sign_detector.py   (SignRecognizer)       │
│    src/translation/speech_handler.py  (Vosk + pyttsx3)  │
│    src/translation/sign_to_text.py   (de-dup converter) │
│    src/models/xgboost_classifier.py                     │
│    src/utils/landmark_normalizer.py  (XGBoost features) │
└─────────────────────────────────────────────────────────┘
```

---

## 3. UI Layout

```
┌────────────────────────┬─────────────────────────────┐
│  📷 Live Camera        │  🎬 Sign Language Avatar    │
│  (CameraPanel view)    │  (video_display_label)      │
│  Row 0 — weight 3      │  Row 0 — weight 3           │
├────────────────────────┼─────────────────────────────┤
│  📝 Current Translation│  📊 Speech Status           │
│  Row 1 — weight 1      │  Row 1 — weight 1           │
├────────────────────────┴─────────────────────────────┤
│  ▶ Start Camera          🎤 Speak                    │
│  Row 2 — fixed height                                │
├──────────────────────────────────────────────────────┤
│  Status bar (full width)                             │
└──────────────────────────────────────────────────────┘
```

---

## 4. Component Responsibilities

### 4.1 Entry Point — `src/main.py`

- Environment validation (Python version, dependencies, directories).
- Configures logging to file + console.
- Imports and launches `MainWindow` from `ui.gui` (or `ui.main_window`).
- Never imports UI widgets directly; stays dependency-free of GUI frameworks.

### 4.2 View Layer

#### `src/ui/main_window.py` — `MainWindow`
- Owns only layout construction and widget wiring.
- Creates all domain services and application services in `_init_services()`.
- Creates controllers in `_init_controllers()`, passing itself as the view.
- Implements the *view protocol* that controllers call back into:
  `update_translation()`, `update_speech_status()`, `update_status()`,
  `set_camera_button_text()`, `set_speak_button_text()`, `update_frame()`.
- Button click handlers delegate immediately to controllers with no logic.

#### `src/ui/panels/camera_panel.py` — `CameraPanel`
- Pure view widget: renders the camera feed + hand-landmark overlays.
- `update_frame(frame, left_raw, right_raw, left_stable, right_stable)`:
  draws landmark dots/lines (from raw results) and confidence text labels
  (from stable results), then displays the composite frame.
- No camera hardware access, no threading, no temporal logic.

#### `src/ui/gui.py` — Backward-compatibility shim
- One-line re-exports of `MainWindow` and `CameraPanel`.
- Keeps existing `from ui.gui import MainWindow` imports working.

### 4.3 Controller Layer

#### `src/controllers/camera_controller.py` — `CameraController`
- Handles Start/Stop camera button clicks.
- Receives `(frame, left_result, right_result)` from `CameraService`
  on the background thread; applies temporal smoothing via `RecognitionService`;
  schedules view updates via `view.after(0, …)`.
- Fires `sign_recognized_callback` (→ `SignController`) for newly-stable signs.

#### `src/controllers/speech_controller.py` — `SpeechController`
- Handles Speak/Stop button clicks.
- Receives `SpeechResult` from `SpeechService` on the recognition thread;
  schedules UI updates on the main thread.
- Routes final results to `VideoService.play_text()` for animations.

#### `src/controllers/sign_controller.py` — `SignController`
- Receives confirmed stable signs from `CameraController`.
- Passes them through `SignToTextConverter` (de-duplication + mapping).
- Updates translation display and triggers TTS for high-confidence signs
  (threshold: 0.82).

### 4.4 Application Service Layer

| Service | Wraps | Responsibility |
|---------|-------|----------------|
| `CameraService` | `cv2.VideoCapture` + `SignRecognizer` | Camera capture loop in background thread |
| `RecognitionService` | (self-contained) | Temporal smoothing: 3-of-4 majority vote, 6-frame hold-down |
| `SpeechService` | `SpeechHandler` | STT start/stop + async TTS facade |
| `VideoService` | `VideoPlayerPanel` | Text-to-sign clip queuing facade |

### 4.5 State Management — `src/app/app_state.py`

`AppState` is a thread-safe dataclass (internal `RLock`) holding:
- `current_translation` — latest recognized text
- `speech_status` — `"Ready"` | `"Listening"` | `"Processing"` | `"Speaking"`
- `camera_running` — camera loop active flag
- `listening` — STT active flag

All controllers read/write state through typed setters (`set_translation()`,
`set_camera_running()`, …) rather than mutating raw attributes.

### 4.6 Domain Layer

| Module | Technology | Role |
|--------|-----------|------|
| `models/sign_detector.py` | MediaPipe + XGBoost | Hand detection + sign classification |
| `translation/speech_handler.py` | Vosk + pyttsx3 | Offline STT + TTS |
| `translation/sign_to_text.py` | (pure Python) | Sign-label de-dup + text accumulation |
| `ui/video_player.py` | OpenCV + CTk | Video file playback + queue management |
| `utils/landmark_normalizer.py` | (pure Python) | Hand landmark normalization for XGBoost |

---

## 5. Threading Model

```
Main Thread (Tkinter event loop)
├── after(0, …) callbacks from camera thread
├── after(0, …) callbacks from speech thread
└── after(0, …) callbacks from video playback thread

Camera Thread (daemon)  [CameraService._capture_loop]
├── cv2.VideoCapture.read()
├── SignRecognizer.process_frame()
└── → CameraController._on_frame_received()

Speech Thread (daemon)  [VoskSpeechRecognizer._recognition_loop]
├── pyaudio stream.read()
├── vosk KaldiRecognizer.AcceptWaveform()
└── → SpeechController._handle_speech_result() [via after(0)]

Video Thread (daemon)  [VideoPlayer._playback_loop]
├── cv2.VideoCapture.read()
└── → MainWindow._display_video_frame() [via after(0)]

TTS Thread (daemon)  [TextToSpeechEngine.speak_async → speak()]
└── pyttsx3.engine.runAndWait()
```

All cross-thread UI updates use `widget.after(0, callback)` — the only
safe way to update Tkinter/CTk widgets from a background thread.

---

## 6. Sign-to-Speech Workflow

```
Webcam frame
  └─► CameraService._capture_loop()
       ├─► SignRecognizer.process_frame()   [MediaPipe + geometric/XGBoost]
       └─► CameraController._on_frame_received(frame, left_raw, right_raw)
              ├─► RecognitionService.update()   [3-of-4 majority vote]
              ├─► view.after(0, camera_panel.update_frame)   [draw overlays]
              └─► [if new stable sign] view.after(0, sign_controller.on_sign_recognized)
                       ├─► SignToTextConverter.add_confirmed_prediction()
                       ├─► MainWindow.update_translation()
                       └─► [if confidence ≥ 0.82] SpeechService.speak_async()
```

## 7. Speech-to-Sign Workflow

```
Microphone audio chunks
  └─► VoskSpeechRecognizer._recognition_loop()
       └─► SpeechController._handle_speech_result()   [via after(0)]
              ├─► [interim] update_speech_status("Listening")
              └─► [final]   MainWindow.update_translation(text)
                            VideoService.play_text(text)
                              └─► VideoPlayerPanel._queue_worker_loop()
                                    └─► VideoPlayer._playback_loop()
                                          └─► MainWindow._display_video_frame()
```

---

## 8. SOLID Principles Applied

| Principle | Application |
|-----------|------------|
| **S**ingle Responsibility | Each class has one job: `CameraService` captures, `RecognitionService` smooths, `CameraController` wires them, `CameraPanel` displays. |
| **O**pen/Closed | New recognition algorithms plug in via `SignRecognizer` without touching the view or controllers. |
| **L**iskov Substitution | `SpeechService` and `VideoService` are facades; their underlying engines are swappable. |
| **I**nterface Segregation | Controllers depend on a minimal *view protocol* (5–6 methods), not the full `MainWindow` class. |
| **D**ependency Inversion | Controllers receive services via constructor injection; no `import` of hardware drivers at the controller level. |

---

## 9. Regression Verification (2026-07-04)

All components verified against the live system after refactoring.

| Check | Result |
|-------|--------|
| All 24 module imports | PASS |
| MediaPipe SignRecognizer (blank frame) | PASS |
| XGBoost / geometric-rules fallback | PASS |
| RecognitionService 3-of-4 stability logic | PASS |
| RecognitionService hold-down + reset | PASS |
| Confidence gate (0.75) and UNKNOWN filter | PASS |
| SignToTextConverter de-dup + mapping | PASS |
| SignController sign → text → TTS pipeline | PASS |
| SpeechController start/stop + result routing | PASS |
| VideoService graceful degradation (no videos) | PASS |
| CameraService live camera start/stop | PASS |
| CameraPanel view — overlays, set_status, null safety | PASS |
| MainWindow widget tree — all 16 required attributes | PASS |
| MainWindow view protocol — all 7 methods callable | PASS |
| gui.py backward-compat shim identity checks | PASS |
| VideoPlayerPanel + VideoService wiring | PASS |
| Window close / cleanup (nothing started) | PASS |
| Sign-to-Speech callback chain (synthetic frames) | PASS |
| Speech-to-Sign animation chain (mock services) | PASS |
| pyttsx3 TTS speak_async + cleanup | PASS |
| Vosk STT thread start/stop + cleanup | PASS |
| SpeechHandler facade init + lifecycle | PASS |
| test_validator_integration.py (pre-existing) | PASS |
| main.py initialization path simulation | PASS |

**Total: 24/24 checks passed. 0 regressions.**

---

## 10. File Summary — What Changed vs. Original

| File | Status | Description |
|------|--------|-------------|
| `src/app/__init__.py` | NEW | App package |
| `src/app/app_state.py` | NEW | Thread-safe centralized state |
| `src/services/__init__.py` | NEW | Services package |
| `src/services/camera_service.py` | NEW | Camera capture loop (from CameraPanel) |
| `src/services/recognition_service.py` | NEW | Temporal smoothing (from CameraPanel) |
| `src/services/speech_service.py` | NEW | STT+TTS facade |
| `src/services/video_service.py` | NEW | Video playback facade |
| `src/controllers/__init__.py` | NEW | Controllers package |
| `src/controllers/camera_controller.py` | NEW | Camera event handler |
| `src/controllers/speech_controller.py` | NEW | Speech event handler |
| `src/controllers/sign_controller.py` | NEW | Sign → text → TTS pipeline |
| `src/ui/panels/__init__.py` | NEW | Panels sub-package |
| `src/ui/panels/camera_panel.py` | NEW | View-only camera display |
| `src/ui/main_window.py` | REPLACED | Was empty stub; now full clean view |
| `src/ui/gui.py` | REPLACED | Was God Class (1100 lines); now 18-line shim |
| `src/main.py` | UNCHANGED | Entry point |
| `src/ui/video_player.py` | UNCHANGED | VideoPlayer + VideoPlayerPanel |
| `src/models/*` | UNCHANGED | Domain models |
| `src/translation/*` | UNCHANGED | Speech handler + sign-to-text |
| `src/utils/landmark_normalizer.py` | ACTIVE | Hand landmark normalization (imported by models/xgboost_classifier.py) |
| `src/utils/constants.py` | INACTIVE | Utility constants (not currently imported) |
| `src/utils/validators.py` | INACTIVE | Validation helpers (not currently imported) |
| `src/utils/logger.py` | INACTIVE | Logging utilities (not currently imported) |
| `src/database/*` | LEGACY | Database module (not used in current system) |
| `src/camera/*` | LEGACY | Camera module stubs (replaced by CameraService) |

