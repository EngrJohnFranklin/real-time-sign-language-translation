# Real-Time Sign Language Translation System

A comprehensive desktop application for real-time bidirectional sign language translation using computer vision, speech recognition, and text-to-speech technology.

## Features

- **📷 Real-Time Hand Detection**: Uses MediaPipe for accurate hand landmark detection with visual overlay
- **🤖 Sign Recognition**: Recognizes 10 common sign language gestures (Hello, Thank You, Yes, No, Help, Good Morning, Sorry, Please, Goodbye, I Love You)
- **🎤 Speech-to-Text**: Offline speech recognition using Vosk (no internet required)
- **🔊 Text-to-Speech**: Converts recognized signs to spoken words using pyttsx3
- **🎬 Video Demonstrations**: Built-in video player for learning sign language gestures
- **💻 Modern GUI**: Dark-themed CustomTkinter interface with intuitive controls
- **⚡ Real-Time Processing**: Multi-threaded background processing for smooth UI

## System Requirements

- **OS**: Windows 10 or later
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Webcam**: Required for real-time hand detection
- **Microphone**: Required for speech recognition
- **Disk Space**: 2GB (includes Vosk model)

## Installation Guide (Windows)

### Step 1: Install Python

1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. **Important**: During installation, check the box "Add Python to PATH"
3. Click "Install Now"
4. Verify installation by opening Command Prompt and typing:
   ```bash
   python --version
   ```

### Step 2: Download the Project

**Option A: Using Git**
```bash
git clone https://github.com/yourusername/real-time-sign-language-translation.git
cd real-time-sign-language-translation
```

**Option B: Manual Download**
1. Download the project as ZIP from GitHub
2. Extract to your desired location
3. Open Command Prompt in the project folder

### Step 3: Create Python Virtual Environment

Creating a virtual environment isolates dependencies for this project.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt line, indicating the environment is active.

### Step 4: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Installation may take 10-15 minutes** as it downloads and compiles packages. Common packages being installed:

| Package | Purpose |
|---------|---------|
| opencv-python | Video capture and frame processing |
| mediapipe | Hand landmark detection |
| customtkinter | Modern GUI framework |
| vosk | Offline speech recognition |
| pyttsx3 | Text-to-speech synthesis |
| Pillow | Image processing |
| numpy | Numerical computing |
| pyaudio | Audio input/output |

### Step 5: Download Vosk Speech Recognition Model

Vosk requires a language model for speech recognition. Models are downloaded separately.

#### Option A: Automatic Download (Recommended)

1. Open Command Prompt in the project folder
2. Create a models directory:
   ```bash
   mkdir models
   cd models
   ```

3. Download the English model (this is a large file, ~50MB):
   ```bash
   # Using PowerShell (recommended for Windows)
   $url = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42.zip"
   $output = "vosk-model-en-us.zip"
   (New-Object System.Net.WebClient).DownloadFile($url, $output)
   
   # Or using curl (if available)
   curl -L https://alphacephei.com/vosk/models/vosk-model-en-us-0.42.zip -o vosk-model-en-us.zip
   ```

4. Extract the downloaded file:
   ```bash
   # Using PowerShell
   Expand-Archive vosk-model-en-us.zip -DestinationPath .
   
   # Or use Windows File Explorer to extract
   # Right-click → Extract All
   ```

5. Verify the model structure (should look like):
   ```
   models/
   └── vosk-model-en-us-0.42/
       ├── conf/
       ├── graph/
       ├── ivector/
       └── ...
   ```

6. Set environment variable (optional but recommended):
   ```bash
   # In Command Prompt
   setx VOSK_MODEL_PATH "%cd%\vosk-model-en-us-0.42"
   ```

#### Option B: Manual Download

1. Visit [Vosk Models](https://alphacephei.com/vosk/models)
2. Download the English model (vosk-model-en-us-0.42.zip or latest)
3. Create `models/` folder in project root
4. Extract the ZIP file into the `models/` folder

### Step 6: Create Videos Directory (Optional)

For sign language demonstration videos:

```bash
mkdir videos
```

Place your `.mp4` video files in this directory with these filenames:
- `hello.mp4`
- `thankyou.mp4`
- `yes.mp4`
- `no.mp4`
- `help.mp4`
- `goodmorning.mp4`
- `sorry.mp4`
- `please.mp4`
- `goodbye.mp4`
- `iloveyou.mp4`

**Note**: Videos are optional. The app will work without them.

## Running the Application

### Method 1: Command Line (Recommended)

```bash
# Make sure virtual environment is activated (see step 3)
# You should see (venv) at the start of your command prompt

# Navigate to project directory
cd real-time-sign-language-translation

# Run the application
python src/main.py
```

### Method 2: Direct Execution

```bash
# From project root with activated venv
python -m src.main
```

### First Time Launch

On first launch, the application will:
1. Verify all dependencies
2. Check Python version
3. Validate resource directories
4. Initialize modules
5. Launch the GUI (takes 10-30 seconds)

**Be patient on first launch** as modules need to initialize.

## Application Usage

### Main Window Layout

```
┌─────────────────────┬──────────────────────────┐
│                     │ 🤖 Recognized Signs      │
│   📷 Live Camera    │                          │
│   Feed              ├──────────────────────────┤
│   (with landmarks)  │ 🎤 Speech-to-Text        │
│                     │                          │
│                     ├──────────────────────────┤
│                     │ 🎬 Sign Language Videos  │
│                     │                          │
└─────────────────────┴──────────────────────────┘
        Control Buttons
```

### Control Buttons

| Button | Function |
|--------|----------|
| **▶ Start Camera** | Toggle webcam feed and hand detection |
| **🎤 Listen** | Start/stop speech recognition |
| **🔊 Speak** | Speak the text in the speech display |
| **🗑 Clear** | Clear all text displays |
| **❌ Exit** | Close the application |

### Workflow Example

1. **Click "▶ Start Camera"** - Webcam starts, hand landmarks display
2. **Make a sign** - Recognition appears in the "Recognized Signs" panel
3. **Click "🎤 Listen"** - Microphone activates
4. **Speak into microphone** - Text appears in "Speech-to-Text" panel
5. **Click "🔊 Speak"** - Application speaks the recognized text
6. **Select video** - Choose a sign from dropdown to see demonstration

## Project Folder Structure

```
real-time-sign-language-translation/
│
├── src/                          # Source code
│   ├── main.py                   # Application entry point
│   ├── __init__.py
│   │
│   ├── models/                   # Machine learning modules
│   │   ├── __init__.py
│   │   ├── sign_detector.py      # Hand detection & sign recognition
│   │   ├── gesture_recognizer.py
│   │   ├── inference.py
│   │   └── model_loader.py
│   │
│   ├── translation/              # Translation modules
│   │   ├── __init__.py
│   │   ├── speech_handler.py     # Speech-to-text & text-to-speech
│   │   ├── sign_to_text.py
│   │   ├── text_to_speech.py
│   │   └── language_mapper.py
│   │
│   ├── ui/                       # User interface
│   │   ├── __init__.py
│   │   ├── gui.py                # Main GUI window
│   │   ├── video_player.py       # Video playback
│   │   ├── main_window.py
│   │   ├── styles.py
│   │   ├── widgets.py
│   │   └── assets/               # Images, icons, fonts
│   │
│   ├── camera/                   # Camera handling
│   │   ├── __init__.py
│   │   ├── camera_handler.py
│   │   └── frame_processor.py
│   │
│   ├── database/                 # Database modules
│   │   ├── __init__.py
│   │   ├── db_manager.py
│   │   └── models.py
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── logger.py
│       └── validators.py
│
├── config/                       # Configuration files
│   ├── logging.yaml
│   ├── model_config.json
│   └── settings.yaml
│
├── data/                         # Data directory
│   ├── dictionaries/
│   ├── models/                   # Pre-trained models
│   └── training_data/
│
├── models/                       # Vosk speech models (create this)
│   └── vosk-model-en-us-0.42/    # Downloaded Vosk model
│
├── videos/                       # Sign language demo videos (create this)
│   ├── hello.mp4
│   ├── thankyou.mp4
│   ├── yes.mp4
│   └── ... (more videos)
│
├── logs/                         # Application logs (created at runtime)
│   └── sign_language_app.log
│
├── tests/                        # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_camera.py
│   ├── test_models.py
│   ├── test_translation.py
│   ├── test_ui.py
│   └── integration/
│       └── test_end_to_end.py
│
├── docs/                         # Documentation
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── README.md
│   ├── SETUP.md
│   ├── TROUBLESHOOTING.md
│   └── USAGE.md
│
├── scripts/                      # Utility scripts
│   ├── data_preprocessing.py
│   ├── download_models.py
│   ├── setup_environment.py
│   └── train_model.py
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Project configuration
├── LICENSE                      # License file
└── README.md                    # This file
```

## Troubleshooting

### Installation Issues

#### Problem: "Python is not recognized"
**Solution**: 
- Python is not in PATH. Reinstall Python and **check "Add Python to PATH"** during installation
- Restart Command Prompt after reinstalling Python
- Verify: `python --version`

#### Problem: "pip: command not found"
**Solution**:
```bash
# Use Python module directly
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Problem: Virtual environment won't activate
**Solution**:
```bash
# Delete and recreate
rmdir /s venv
python -m venv venv
venv\Scripts\activate
```

#### Problem: "Permission denied" when installing packages
**Solution**:
```bash
# Run Command Prompt as Administrator
# Then activate venv and reinstall
pip install --upgrade pip
pip install -r requirements.txt
```

### Vosk Model Issues

#### Problem: "Vosk model not found" error
**Solution**:
1. Verify model exists in `models/` folder:
   ```bash
   dir models
   ```
2. Model folder should be named `vosk-model-en-us-0.42` (or similar)
3. Set environment variable:
   ```bash
   setx VOSK_MODEL_PATH "%cd%\vosk-model-en-us-0.42"
   ```
4. Restart Command Prompt and try again

#### Problem: "Vosk model download failed"
**Solution**:
- Check internet connection
- Try downloading model manually from [Vosk website](https://alphacephei.com/vosk/models)
- Alternatively, use a smaller model: `vosk-model-small-en-us-0.15`

### Camera/Microphone Issues

#### Problem: "Cannot open camera"
**Solution**:
- Check if another application is using the camera
- Close video conferencing apps (Zoom, Teams, Skype)
- Unplug and reconnect the webcam
- Try a different USB port
- Restart computer

#### Problem: "No audio input devices available"
**Solution**:
```bash
# Reinstall PyAudio
pip uninstall pyaudio
pip install pipwin
pipwin install pyaudio
```

#### Problem: Microphone not working
**Solution**:
- Check Windows Settings → Privacy → Microphone
- Ensure the app has microphone permission
- Test microphone in other applications first (Skype, Discord)
- Update audio drivers from manufacturer's website

### Application Launch Issues

#### Problem: "ModuleNotFoundError: No module named 'src'"
**Solution**:
- Run from project root directory:
  ```bash
  cd real-time-sign-language-translation
  python src/main.py
  ```

#### Problem: GUI doesn't start / freezes
**Solution**:
- Check logs: `cat logs/sign_language_app.log`
- Ensure all dependencies are installed: `pip list`
- Try updating packages: `pip install --upgrade -r requirements.txt`
- Restart application and computer if needed

#### Problem: "ImportError: No module named 'customtkinter'"
**Solution**:
```bash
# Ensure venv is activated (see (venv) in prompt)
pip install customtkinter --upgrade

# If still failing
pip uninstall customtkinter
pip install customtkinter==5.2.0
```

### Performance Issues

#### Problem: Low FPS / jerky camera feed
**Solution**:
- Close unnecessary applications (browsers, heavy software)
- Lower video resolution in settings
- Disable overlays and animations
- Check CPU/RAM usage (Task Manager)

#### Problem: High CPU usage
**Solution**:
- Reduce video resolution
- Stop all unnecessary background processes
- Check for memory leaks in logs
- Restart application

### Speech Recognition Issues

#### Problem: Speech recognition not working
**Solution**:
1. Check microphone is selected in Windows Sound Settings
2. Test microphone: `Settings → Privacy → Microphone`
3. Check Vosk model is properly installed
4. Enable microphone permission for Python in Windows Defender

#### Problem: Poor speech recognition accuracy
**Solution**:
- Speak clearly and at normal pace
- Reduce background noise
- Move microphone closer
- Use a better quality microphone

### Video Playback Issues

#### Problem: "Videos folder not found"
**Solution**:
- Create `videos/` folder in project root:
  ```bash
  mkdir videos
  ```
- This is optional; app works without it

#### Problem: Videos won't play
**Solution**:
- Verify video format is `.mp4`
- Check filenames match expected names
- Try opening video in Windows Media Player to verify it's valid
- Ensure file isn't corrupted

## Checking the Log File

If the application encounters errors, check the detailed log:

```bash
# View last 50 lines of log
type logs\sign_language_app.log | tail -50

# Or open directly with Notepad
notepad logs\sign_language_app.log
```

Look for lines with `ERROR` or `WARNING` to understand what went wrong.

## GPU Acceleration (Optional)

For faster hand detection on systems with NVIDIA GPU:

```bash
# Install CUDA version of MediaPipe (if applicable)
pip install mediapipe-gpu

# Or use TensorFlow GPU
pip install tensorflow-gpu
```

## Performance Tips

1. **Use wired microphone** instead of built-in for better speech recognition
2. **Close background applications** that use GPU/CPU
3. **Ensure good lighting** for better hand detection accuracy
4. **Speak clearly** to microphone for accurate speech recognition
5. **Update drivers** - especially graphics and audio drivers
6. **Use USB 3.0 webcam** if available for better performance

## Common Questions

### Q: Can I use this on Mac/Linux?
**A**: The code is cross-platform, but this guide is Windows-specific. PyAudio installation may require different steps on Mac/Linux.

### Q: Do I need internet?
**A**: No. Everything runs offline except optional video downloads and model downloads (one-time only).

### Q: Can I customize recognized signs?
**A**: Yes! Edit `src/models/sign_detector.py` to add new signs by defining new geometric rules.

### Q: What if I don't have a webcam?
**A**: You'll need a webcam for hand detection. USB webcams are inexpensive ($20-50).

### Q: Can I add my own videos?
**A**: Yes! Place `.mp4` files in the `videos/` folder with proper naming.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the Troubleshooting section above
2. Check application logs in `logs/sign_language_app.log`
3. Open an issue on GitHub with error details and log excerpts

## References

- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [Vosk Speech Recognition](https://alphacephei.com/vosk/)
- [CustomTkinter GitHub](https://github.com/TomSchimansky/CustomTkinter)
- [OpenCV Documentation](https://docs.opencv.org/)
- [pyttsx3 Documentation](https://pyttsx3.readthedocs.io/)

## Acknowledgments

This project uses:
- **MediaPipe** - Hand landmark detection
- **OpenCV** - Video processing
- **Vosk** - Speech recognition
- **pyttsx3** - Text-to-speech
- **CustomTkinter** - Modern GUI framework

---

**Last Updated**: June 23, 2026  
**Version**: 1.0.0
