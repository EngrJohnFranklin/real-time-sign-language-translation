# Setup and Installation Guide

## Requirements

- Windows 10 or later
- Python 3.9 or newer
- Webcam for live FSL letter recognition
- Microphone for offline speech recognition
- Speakers or headphones for text-to-speech

## Installation

From the repository root:

```powershell
py -3.9 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The English Vosk model is bundled under `model/`. A Filipino model can be
installed under `model-tl/`; the application warns when it is unavailable.

Run the desktop application with:

```powershell
python src/main.py
```

For a Windows executable, run `build_executable.ps1` after creating `.venv`.
It produces `dist/SignLanguageTranslator/SignLanguageTranslator.exe`.
