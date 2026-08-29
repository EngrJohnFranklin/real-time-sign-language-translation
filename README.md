# Real-Time Sign Language Translation

A Windows-focused desktop application for local, real-time Filipino Sign Language (FSL) recognition and constrained offline speech input. It uses MediaPipe hand landmarks, XGBoost classifiers, Vosk speech recognition, and pyttsx3 text-to-speech.

All camera, speech, and model processing runs locally. No cloud service or internet connection is required after dependencies are installed.

## What It Does

The application opens with two input modes:

- **Sign Language Input**: starts a webcam feed, detects up to two hands with MediaPipe, and shows a recognized FSL fingerspelling letter. Recognition requires `data/models/sign_model.pkl`, trained from the included landmark-data workflow.
- **Speak Input**: records from the default microphone using Vosk, displays the recognized word or phrase, and shows the matching image from `assets/sign_images/` when available.

The speech recognizer deliberately uses a closed vocabulary:

`good`, `hello`, `i love you`, `love`, `me`, `no`, `please`, `quiet`, `sorry`, `thank you`, and `yes`.

## Included Resources

- English Vosk model: `model/`
- Filipino/Tagalog Vosk model location: `model-tl/`
- Sign images for the speech vocabulary: `assets/sign_images/`
- Word-template recordings: `data/word_templates/`
- Landmark training dataset: `data/training_data/landmark_data.csv`
- Application settings: `config/settings.yaml` and `config/model_config.json`



## Requirements

- Windows 10 or later
- Python 3.9 or newer
- Webcam for Sign Language Input
- Microphone for Speak Input
- Audio output for text-to-speech feedback

The application is configured for CPU inference. A GPU is not required.

## Setup

Clone or download the repository, then create and activate a virtual environment from the project root.

```powershell
py -3.9 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`PyAudio` installation can require Windows build tools on some machines. See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for environment and device troubleshooting.

## Run the Application

From the activated environment in the project root:

```powershell
python src/main.py
```

The startup check verifies the Python version, required runtime packages, and resource directories. Runtime logs are written to `logs/sign_language_app.log`.

### Using the App

1. Choose **Sign Language Input** to open the camera view. The camera starts automatically; use the button to stop or restart it.
2. Present a trained FSL fingerspelling letter clearly to the camera. The recognized letter and confidence-stabilized result appear beside the feed.
3. Choose **Speak Input** to use the microphone. Select **Speak**, say one supported vocabulary item, then wait for the recognition cooldown before the next word.
4. The recognized speech is shown as text and its corresponding sign image is displayed when an asset exists.

## Train Recognition Models

The repository separates letter and static-word model training. Both use 126 normalized features: 63 landmark values per hand.

### FSL Letter Model

Collect labeled samples, inspect them, then train the model used by the live sign-input view:

```powershell
python scripts/verify_gesture_landmarks.py
python scripts/collect_training_data.py
python scripts/analyze_dataset.py
python scripts/train_xgboost_model.py
```

Training writes `data/models/sign_model.pkl`. The trainer performs cross-validation and reports per-class sample counts. Aim for at least 50 samples per label and varied lighting, orientation, and distance.

### Static Word Model

The word-template trainer builds a separate model without changing the letter model:

```powershell
python scripts/collect_word_templates.py
python scripts/train_word_classifier.py
```

It saves `data/models/word_model.pkl` and validates by recording group to avoid frame leakage. Current word templates correspond to the closed speech vocabulary.

Read [docs/COLLECTION_WORKFLOW.md](docs/COLLECTION_WORKFLOW.md), [docs/LANDMARK_NORMALIZATION.md](docs/LANDMARK_NORMALIZATION.md), and [docs/FSL_REFERENCE.md](docs/FSL_REFERENCE.md) before creating or interpreting a dataset. The FSL handshape reference is explicitly provisional and should be verified against authoritative sources.

## Test and Verify

Run the automated test suite:

```powershell
python -m pytest
```

Run the dataset-system verification script:

```powershell
python verify_system.py
```

The integration validator can also be run directly:

```powershell
python test_validator_integration.py
```

## Build a Windows Executable

The PyInstaller specification bundles the application code, configuration, models, sign images, and required binary dependencies. With `.venv` set up:

```powershell
.\build_executable.ps1
```

The distributable is created at `dist/SignLanguageTranslator/SignLanguageTranslator.exe`.

## Project Layout

```text
src/
  app/                 Shared, thread-safe application state
  controllers/         Camera, sign, and speech interaction controllers
  models/              MediaPipe, XGBoost, letter, and word recognition
  services/            Camera, recognition, speech, and video services
  translation/         Speech, text-to-speech, and sign mapping logic
  ui/                  CustomTkinter views, panels, and components
  utils/               Paths, camera selection, and landmark normalization
scripts/               Data collection, validation, analysis, and training
data/
  models/              Generated XGBoost model files
  training_data/       Labeled landmark dataset
  word_templates/      Recorded static-word feature sequences
assets/sign_images/    Word-to-sign display images
model/                 Bundled English Vosk model
model-tl/              Filipino Vosk model location
config/                Runtime configuration
tests/                 Unit and integration tests
docs/                  Setup, architecture, API, and dataset guides
```

## Further Documentation

- [Setup](docs/SETUP.md)
- [Usage](docs/USAGE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API](docs/API.md)
- [Data collection](docs/DATA_COLLECTION.md)
- [Custom dataset setup](docs/CUSTOM_DATASET_SETUP.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## License

This project is released under the [MIT License](LICENSE). See [LICENSE-THIRD-PARTY.md](LICENSE-THIRD-PARTY.md) for notices covering bundled and third-party components.