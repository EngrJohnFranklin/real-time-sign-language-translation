# Software Architecture Document

## Real-Time Bidirectional Filipino Sign Language Translation System

### 1. System Overview

The system is an offline desktop application that combines computer vision,
machine learning, speech recognition, text-to-speech, and a CustomTkinter user
interface to support communication between Filipino Sign Language (FSL), spoken
language, and text.

The intended translation directions are:

1. **FSL to Text:** webcam frames are processed into hand landmarks and sign
   predictions, then displayed as text.
2. **Speech to Text:** microphone audio is transcribed locally using Vosk.
3. **Text to Speech:** recognised sign text is spoken through the local pyttsx3
   engine.
4. **Text to FSL:** text is resolved to locally stored whole-word or alphabet
   sign clips for playback.

All core processing uses local resources. No cloud service or network API is
required for normal operation. Vosk models, trained classifiers, word templates,
and media assets are stored within the project environment.

**Current implementation note:** FSL-to-text, speech-to-text, and text-to-speech
are wired into `MainWindow`. The text-to-FSL playback capability is implemented
by `VideoService` and `ui/video_player.py`, but the current GUI constructs
`VideoService(None)`. Consequently, speech results do not currently produce
sign-video playback until a `VideoPlayerPanel` is supplied during GUI setup.

### 2. Component Architecture and Data Flow

```text
+--------------------+       +-----------------------------------------+
| Input Devices      |       | Presentation Layer                      |
| Webcam             |------>| ui/main_window.py                       |
| Microphone         |------>| ui/views/, ui/panels/, ui/video_player  |
| GUI controls       |------>| CustomTkinter views and status display  |
+--------------------+       +------------------+----------------------+
                                                 |
                                                 v
                         +-----------------------+----------------------+
                         | Application Control                          |
                         | CameraController | SpeechController          |
                         | SignController   | AppState                  |
                         +-----------+-------------------+--------------+
                                     |                   |
                 +-------------------+                   +--------------------+
                 v                                                            v
+--------------------------------------+      +--------------------------------------+
| FSL Recognition Pipeline             |      | Speech and Sign-Video Pipeline       |
| CameraService: OpenCV capture        |      | SpeechService -> SpeechHandler       |
| SignRecognizer: MediaPipe landmarks  |      | Vosk: local speech-to-text           |
| XGBoost/word recognizers             |      | pyttsx3: local text-to-speech        |
| RecognitionService: temporal filter  |      | VideoService -> local FSL clips*     |
| SignToTextConverter                  |      +--------------------------------------+
+---------------------+----------------+                         |
                      |                                          v
                      +------------------------------> Text / spoken output

* The video service is available in source code but has no player panel wired
  into the current MainWindow configuration.

Local resources: config/, data/, assets/, model/, model-tl/
```

Camera capture and speech recognition run in background threads. Controllers use
CustomTkinter's `after()` mechanism to marshal user-interface updates to the
main event loop.

### 3. Component Descriptions

| Module | Responsibility |
|---|---|
| `src/main.py` | Configures logging, checks Python and dependencies, prepares local directories, and launches the GUI. |
| `src/app/app_state.py` | Holds shared state for camera activity, speech activity, recognition state, and displayed translation. |
| `src/ui/` | Provides the CustomTkinter main window, home/sign/speech views, camera panel, video player, themes, and widgets. |
| `src/controllers/` | Coordinates UI events with services. Camera, speech, and sign controllers keep workflow logic outside UI widgets. |
| `src/services/` | Wraps camera capture, temporal recognition, speech lifecycle, and video playback behind controller-facing APIs. |
| `src/models/` | Implements gesture detection, model loading, feature inference, XGBoost classification, and word recognition. |
| `src/translation/` | Implements sign-to-text conversion, speech handling, text-to-speech, language mapping, dynamic-time-warping matching, and word-template loading. |
| `src/utils/` | Provides local configuration, paths, camera selection, landmark normalisation, validation, constants, and logging helpers. |
| `src/camera/` | Contains legacy camera-handler and frame-processing support modules. |
| `src/database/` | Contains optional database models and a manager; it is not required by the current GUI workflow. |

### 4. Technology Stack

| Runtime library | Purpose |
|---|---|
| OpenCV (`opencv-python`) | Acquires webcam frames, processes images, and supports video playback. |
| MediaPipe | Extracts hand landmarks from live camera frames. |
| XGBoost | Classifies trained gesture and word-sign features. |
| scikit-learn | Supports trained-model artefacts and machine-learning preprocessing utilities. |
| NumPy | Stores and transforms image, landmark, and feature-array data. |
| CustomTkinter | Builds the desktop graphical user interface. |
| Vosk | Performs offline speech-to-text using locally installed acoustic models. |
| PyAudio | Captures microphone audio supplied to Vosk. |
| pyttsx3 | Produces offline text-to-speech output. |
| Pillow | Converts image frames for presentation in CustomTkinter. |
| PyYAML | Reads YAML configuration files. |
| python-dotenv | Supports local environment-variable configuration. |

### 5. Folder Structure

```text
real-time-sign-language-translation/
|-- src/                         Application source code
|   |-- main.py                  Startup, validation, logging, and GUI launch
|   |-- app/                     Shared application state
|   |-- controllers/             Camera, speech, and sign workflow coordination
|   |-- services/                Camera, recognition, speech, and video facades
|   |-- models/                  Sign detection, classifiers, and model loading
|   |-- translation/             Speech, text, template, and language processing
|   |-- ui/                      CustomTkinter windows, views, panels, and player
|   |-- utils/                   Configuration, paths, normalisation, and helpers
|   |-- camera/                  Legacy camera support
|   `-- database/                Optional persistence support
|-- config/                      YAML and model configuration
|-- data/                        Trained models, training data, and word templates
|-- assets/                      Static visual assets, including sign images
|-- model/                       Local English Vosk model
|-- model-tl/                    Local Filipino/Tagalog Vosk model
|-- scripts/                     Setup, data collection, analysis, and training tools
|-- tests/                       Unit and integration tests
|-- docs/                        Technical and user documentation
|-- logs/                        Runtime log output
|-- requirements.txt             Dependency installation list
|-- pyproject.toml               Packaging and development-tool configuration
|-- setup.py                     Setuptools package configuration
|-- README.md                    Installation and project overview
|-- LICENSE                      Project license
`-- LICENSE-THIRD-PARTY.md       Third-party model attribution and license notice
```

### 6. Local Deployment Requirements

A local deployment requires a supported Python installation, the declared runtime
dependencies, local Vosk model directories, and project data files. A webcam is
required for FSL input; a microphone is required for speech input; and speakers
or headphones are required for text-to-speech output. Sign-video output also
requires local video clips and wiring a `VideoPlayerPanel` to `VideoService`.
