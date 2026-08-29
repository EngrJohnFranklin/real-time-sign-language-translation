# Real-Time Sign Language Translation Documentation

## Overview
A local desktop application with two input modes: trained FSL alphabet recognition
from a webcam, and constrained offline speech recognition that displays a matching
static sign image.

## Table of Contents
- [Setup](SETUP.md)
- [Usage](USAGE.md)
- [API](API.md)
- [Architecture](ARCHITECTURE.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## Features
- Real-time sign language recognition
- FSL alphabet labels (`A` through `Z`) recognized from the webcam when
	`data/models/sign_model.pkl` is available
- Text-to-speech output
- Offline Vosk speech recognition for the configured closed vocabulary
- Static sign-image display from `assets/sign_images/` for recognized speech
- CustomTkinter home, sign-input, and speech-input views
