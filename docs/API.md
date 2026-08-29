# API Documentation

## Application Entry Point

- `src.main.main()` initializes logging, validates the environment, and launches
	`ui.main_window.MainWindow`.

## Recognition

- `models.sign_detector.SignRecognizer` detects MediaPipe hand landmarks and
	loads `data/models/sign_model.pkl` when present.
- `models.word_recognizer.WordRecognizer` loads the optional static-word model
	at `data/models/word_model.pkl`.
- `services.recognition_service.RecognitionService` applies temporal filtering
	before a letter prediction is shown.

## Speech

- `translation.speech_handler.VoskSpeechRecognizer` performs local, grammar-
	constrained speech recognition with a local Vosk model.
- `translation.speech_handler.TextToSpeechEngine` speaks recognized sign output
	through pyttsx3.
- `translation.speech_handler.SpeechLanguage` defines English and Filipino
	model selection.

## User Interface

- `ui.main_window.MainWindow` owns view switching and controller wiring.
- `ui.views.sign_view.SignView` displays the live camera and recognized letter.
- `ui.views.speech_view.SpeechView` displays recognized speech and a matching
	`ui.components.sign_image_display.SignImageDisplay` image.

`ui.video_player` and `services.video_service` are legacy components and are
not wired into the current GUI.
