# Troubleshooting Guide

## Common Issues

### Camera Not Detected
- Check camera permissions
- Verify camera is connected
- Try selecting camera manually

### Model Loading Failed
- Confirm the bundled English Vosk model exists under `model/`
- Train the alphabet classifier to create `data/models/sign_model.pkl`
- Confirm generated models and resource folders are readable

### Poor Recognition Accuracy
- Improve lighting conditions
- Ensure proper camera angle
- Keep the whole hand visible and collect more varied samples before retraining
- Confirm that the intended alphabet label is present and balanced in the CSV
- Reduce background noise when using Speak Input

### Performance Issues
- Close other camera- or CPU-intensive applications
- Reduce camera resolution in `config/settings.yaml` if necessary
- Close other applications
- Check system resources

### No Sign Image Appears
- Confirm the recognized word is in the supported speech vocabulary
- Check `assets/sign_images/` for a matching lowercase `.png` filename
- Multi-word phrases require a matching full-phrase image, such as
	`thank_you.png`; they do not fall back to the first word

### Filipino Speech Recognition Unavailable
- Install a valid Filipino Vosk model under `model-tl/`
- English speech recognition uses the bundled `model/` directory
