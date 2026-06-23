# System Architecture

## Component Overview
- Camera Capture Layer
- Frame Processing Layer
- Model Inference Layer
- Translation Layer
- UI/Display Layer
- Data Persistence Layer

## Data Flow
1. Webcam captures video frames
2. Frames preprocessed and normalized
3. ML model performs gesture recognition
4. Detected signs translated to text
5. Text converted to speech (optional)
6. Results displayed in UI
7. Translation history logged

## Technology Stack
- Python 3.9+
- PyQt/Tkinter for GUI
- TensorFlow/PyTorch for ML models
- OpenCV for video processing
