# API Documentation

## Core Modules

### Camera Module
- `capture_frame()` - Get current frame from webcam
- `process_frame()` - Preprocess frame for inference

### Models Module
- `predict_gesture()` - Run inference on frame
- `load_model()` - Load trained model

### Translation Module
- `sign_to_text()` - Convert detected signs to text
- `text_to_speech()` - Convert text to audio

### UI Module
- `display_video()` - Show video stream
- `update_translation()` - Update translation output
