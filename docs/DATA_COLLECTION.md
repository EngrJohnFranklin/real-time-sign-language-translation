# Enhanced Data Collection Guide

## Overview

The upgraded `collect_training_data.py` provides a professional-grade data collection interface for the sign language classifier. It includes real-time quality validation, flexible capture modes, and duplicate prevention.

## Display Elements

The camera window shows comprehensive information:

```
┌─────────────────────────────────────────────────────────────┐
│ Sign: Hello           [1/10] Samples: 5         ◉ SINGLE     │
│ Left hand (1) | Conf: 0.92 | FPS: 28.3                      │
│ ⚠ Not fully visible                                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│                  [Camera feed with landmarks]                │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│ SPACE: capture | R: continuous | S: stop | N: next |...     │
└─────────────────────────────────────────────────────────────┘
```

### Status Indicators

| Element | Meaning |
|---------|---------|
| **Sign Name** | Current sign being collected (e.g., "Hello") |
| **[1/10]** | Current sign (1) out of total (10) |
| **Samples: 5** | Number of samples collected for current sign |
| **◉ SINGLE** | Single capture mode (saves 1 sample per SPACE) |
| **● CONTINUOUS** | Continuous mode (saves every frame) |
| **⊘ IDLE** | No active recording |
| **Conf: 0.92** | Hand detection confidence (0.0-1.0) |
| **FPS: 28.3** | Frames per second (useful for latency analysis) |

### Validation Messages

| Message | Cause | Action |
|---------|-------|--------|
| "Left hand not fully visible" | Hand partially outside frame | Adjust position |
| "Right hand not fully visible" | Hand partially outside frame | Adjust position |
| "Confidence too low (0.68)" | Detection confidence < 0.7 | Improve lighting/angle |
| "Sample too similar to previous" | Duplicate detected (cosine sim > 0.95) | Move hand or try different angle |

## Keyboard Controls

### Capture Modes

| Key | Function |
|-----|----------|
| **SPACE** | Capture a single sample and return to idle |
| **R** | Start continuous capture (saves every valid frame) |
| **S** | Stop continuous capture and return to idle |

### Navigation

| Key | Function |
|-----|----------|
| **N** | Move to next sign, reset sample count |
| **P** | Move to previous sign, reset sample count |
| **ESC** | Exit and save all collected data |

### Example Workflow

```
1. Start script
2. Press R to begin continuous capture for "Hello"
3. Perform "Hello" gesture 20-30 times with variations
4. Press S to stop
5. Press N to move to next sign
6. Repeat 2-4 for each sign
7. Press ESC when done
```

## Quality Validation

The script automatically validates samples before saving:

### Hand Detection
- ✓ Hand must be detected by MediaPipe
- ✓ Detection confidence ≥ 0.7 (configurable via `MIN_HAND_CONFIDENCE`)

### Hand Visibility
- ✓ All 21 landmarks must be within frame (0 < x < 1, 0 < y < 1)
- ✗ Hands partially outside frame are rejected

### Duplicate Prevention
- ✓ Cosine similarity between current and previous sample < 0.95
- ✗ Nearly identical samples are rejected to avoid redundancy
- Uses: `similarity = (sample1 · sample2) / (||sample1|| × ||sample2||)`

## Data Format

### CSV Output

Saved to `data/training_data/landmark_data.csv`:

```
label,left_x0,left_y0,left_z0,...,left_z20,right_x0,right_y0,...,right_z20
Hello,0.15,0.22,0.01,...,-0.03,0.0,0.0,...,0.0
Hello,0.16,0.23,0.02,...,-0.02,0.0,0.0,...,0.0
Thank You,0.25,0.18,0.00,...,0.05,0.0,0.0,...,0.0
...
```

**Columns:**
- 1 label column (sign name)
- 63 left hand features (21 landmarks × 3 coordinates)
- 63 right hand features (21 landmarks × 3 coordinates)
- **Total: 127 columns**

### Normalization Applied

Each hand's landmarks are normalized:

1. **Translation**: Shifted to wrist origin (landmark 0)
2. **Scaling**: Divided by hand size (wrist-to-middle-finger-MCP distance)
3. **Flattening**: 21 landmarks × 3 axes = 63 elements per hand

Result: Scale and translation invariant features

## Data Collection Tips

### For Best Quality

1. **Lighting**: Good lighting helps MediaPipe detect hands reliably
2. **Angles**: Vary hand angles and distances (20-50 cm from camera)
3. **Speed**: Perform signs at natural speed, not too fast
4. **Coverage**: Capture signs from multiple angles and distances
5. **Hands**: Include both single-hand and two-hand variations if applicable

### Sample Count Recommendations

| Category | Samples | Notes |
|----------|---------|-------|
| Development | 20-50 | For quick testing |
| Training | 50-100 | Per sign for decent accuracy |
| Production | 200-500 | Per sign for high accuracy |

### Avoiding Common Mistakes

- ❌ Collecting samples at same distance/angle only → limited generalization
- ❌ Capturing low-confidence hands → poor training data
- ❌ Ignoring validation messages → wasted samples
- ✅ Varying conditions (distance, angle, speed)
- ✅ Collecting diverse hand positions
- ✅ Keeping hand fully visible

## Configuration

Edit script for customization:

```python
# Hand detection confidence threshold
MIN_HAND_CONFIDENCE = 0.7  # Range: 0.5-1.0

# Output file path
OUTPUT_PATH = _PROJECT_ROOT / "data" / "training_data" / "landmark_data.csv"

# Duplicate detection threshold
# In _is_sample_duplicate(), cosine similarity > threshold = duplicate
threshold = 0.95  # Range: 0.9-0.99 (higher = stricter)
```

## Workflow Example

### Session 1: Initial Collection

```bash
$ python scripts/collect_training_data.py

=== Enhanced Sign Language Data Collection ===
Signs: Hello, Thank You, Yes, No, Help, ...
Output: data/training_data/landmark_data.csv

Controls:
  SPACE = Capture single sample
  R = Start continuous capture
  S = Stop capture
  N = Next sign
  P = Previous sign
  ESC = Exit

Ready to collect: Hello

[User presses R]
Starting continuous capture for 'Hello'...
✓ Hello: sample 1
✓ Hello: sample 2
✓ Hello: sample 3
...
✓ Hello: sample 25

[User presses S]
Stopped capture

[User presses N]
Next: Thank You

[Repeat for all signs]

[User presses ESC]
Exiting...

============================================================
✓ Saved 150 new samples to data/training_data/landmark_data.csv
============================================================

Samples per sign:
  ✓ Hello               :  25 samples
  ✓ Thank You           :  20 samples
  ✓ Yes                 :  22 samples
  ...
```

### Session 2: Augment Existing Data

```bash
$ python scripts/collect_training_data.py

Existing samples: 150

[Script appends to existing CSV]
```

## Troubleshooting

### "Cannot open webcam"
- Check webcam connection
- Try another camera app first
- May be permissions issue on some systems

### "Hand detected but samples rejected"
- Check validation messages (hand not fully visible?)
- Improve lighting
- Move hand closer to camera (but keep it fully visible)

### "Confidence too low"
- Improve lighting conditions
- Face camera more directly
- Hold hand steadier

### "Sample too similar to previous"
- Move hand more between captures
- Try different angle or position
- Wait a moment before next capture

### Slow FPS (<15)
- Close other applications
- Check webcam resolution setting (640x480 is default)
- CPU might be under load

## Performance Metrics

Typical performance on modern hardware:

- **Hand detection**: 10-30 ms per frame
- **Landmark normalization**: 1-2 ms per frame
- **Total latency**: 15-50 ms
- **Throughput**: 20-30 FPS

## Next Steps

After collecting data:

```bash
# 1. Train the XGBoost model
python scripts/train_xgboost_model.py

# 2. Test the system
python src/main.py

# 3. If accuracy is low, collect more data and retrain
```

## Related Files

- **Normalization**: `src/utils/landmark_normalizer.py`
- **Training**: `scripts/train_xgboost_model.py`
- **Inference**: `src/models/sign_detector.py`
- **Features**: `src/models/xgboost_classifier.py`
- **Configuration**: `config/model_config.json`
