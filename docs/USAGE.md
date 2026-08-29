# Usage Guide

## Start

Activate the virtual environment and run:

```powershell
python src/main.py
```

## Sign Language Input

Select **Sign Language Input** from the home view. The application starts the
camera automatically and shows MediaPipe hand landmarks with the stabilized
letter prediction. The letter model must exist at `data/models/sign_model.pkl`.
Use the camera button to stop or restart capture.

The implemented live classifier recognizes trained FSL alphabet labels only. It
does not perform unrestricted sentence translation. Keep the hand visible,
well-lit, and within the camera frame for best results.

## Speak Input

Select **Speak Input**, then choose **Speak**. The app listens through the
default microphone, displays recognized text, and shows a matching static image
from `assets/sign_images/` when one is available.

Speech recognition is local and restricted to: `good`, `hello`, `i love you`,
`love`, `me`, `no`, `please`, `quiet`, `sorry`, `thank you`, and `yes`.
Wait for the listening cooldown before saying the next word or phrase.

## Logs

Startup and runtime messages are written to `logs/sign_language_app.log`.
