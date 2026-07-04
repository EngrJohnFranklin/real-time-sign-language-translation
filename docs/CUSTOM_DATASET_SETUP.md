# Custom Static Gesture Dataset Setup

## Overview

This guide explains how to create a custom static gesture dataset optimized for MediaPipe Hands + XGBoost classification. The system is designed for real-time recognition of 10 distinct hand gestures.

## The 10 Gestures

| # | Gesture | Meaning | Emoji | Difficulty |
|---|---------|---------|-------|------------|
| 1 | Closed Fist | Sorry | ✊ | ⭐ Easy |
| 2 | Open Palm | Stop | ✋ | ⭐ Easy |
| 3 | Thumbs Up | Yes | 👍 | ⭐ Easy |
| 4 | Thumbs Down | No | 👎 | ⭐ Easy |
| 5 | Index Finger | Help | ☝️ | ⭐⭐ Medium |
| 6 | Shaka | Hello | 🤙 | ⭐⭐ Medium |
| 7 | I Love You | I Love You | 🤟 | ⭐⭐ Medium |
| 8 | Peace Sign | Thank You | ✌️ | ⭐⭐⭐ Hard |
| 9 | OK Sign | Good | 👌 | ⭐⭐⭐ Hard |
| 10 | Vulcan Salute | Goodbye | 🖖 | ⭐⭐⭐ Hard |

## Quick Start (5 minutes)

### 1. Verify Gesture Definitions
```bash
python scripts/verify_gesture_landmarks.py
```
This tool displays each gesture's hand configuration with:
- Key features to look for
- Finger positions for each digit
- Helpful hints during collection

**Controls:**
- N = Next gesture
- P = Previous gesture
- SPACE = Show detailed finger positions
- ESC = Exit

### 2. Collect Initial Test Data
```bash
python scripts/collect_training_data.py
```
- Position your hand for each gesture (gesture shown on screen)
- Press SPACE to capture single sample or R for continuous
- Watch for quality feedback (green = good, red = rejected)
- Press N to next gesture
- Press ESC when done

### 3. Analyze Your Dataset
```bash
python scripts/analyze_dataset.py
```
Shows:
- Total samples per gesture
- Dataset balance and quality
- Recommendations for improvement
- Training readiness score

### 4. Train Model
```bash
python scripts/train_xgboost_model.py
```
Trains XGBoost classifier on your collected data

### 5. Test Real-Time
```bash
python src/main.py
```
Live recognition of your gestures with confidence scores

## Complete Workflow

### Phase 1: Preparation (5 min)

1. **Review gesture definitions**
   ```bash
   python scripts/verify_gesture_landmarks.py
   ```

2. **Prepare environment**
   - Good lighting (bright, no harsh shadows)
   - Clear background (white wall preferred)
   - Camera 1-1.5 meters away
   - Webcam at chest height

3. **Test setup**
   ```bash
   python scripts/collect_training_data.py
   # Then press ESC immediately (just testing)
   ```

### Phase 2: Data Collection (30-60 min)

**Minimum viable dataset:** 200 samples (20 per gesture)
**Recommended dataset:** 500 samples (50 per gesture)
**Production dataset:** 1000+ samples (100+ per gesture)

**Collection strategy:**
```bash
python scripts/collect_training_data.py
```

**For each gesture:**
1. Read the gesture description shown on screen
2. Get into correct hand configuration
3. Press SPACE for single capture or R for continuous
4. Capture 20-50 samples (watch for "Too similar" rejections = duplicates)
5. Vary hand position, rotation, and distance between captures
6. Press N to move to next gesture
7. Watch acceptance rate (target > 85%)

**Recommended order (easy → hard):**
1. Open Palm
2. Closed Fist
3. Thumbs Up
4. Thumbs Down
5. Index Finger
6. Shaka
7. I Love You
8. Peace Sign
9. OK Sign
10. Vulcan Salute

**Pro tips:**
- Collect with multiple users if possible (better generalization)
- Capture both left and right hands
- Vary distances (0.5m, 1m, 1.5m)
- Vary lighting (natural, artificial, low-light)
- Take breaks between gestures
- Don't force gestures - be natural

### Phase 3: Quality Analysis (2 min)

```bash
python scripts/analyze_dataset.py
```

**What to look for:**
- ✓ Total samples ≥ 200 (ideally ≥ 500)
- ✓ All 10 gestures present
- ✓ Balanced distribution (each gesture has similar sample count)
- ✓ Acceptance rate ≥ 85% during collection

**If analysis shows issues:**
- Too few samples → Collect more
- Missing gestures → Add them
- Imbalanced distribution → Collect more of underrepresented gestures
- Low acceptance rate → Improve lighting/positioning

### Phase 4: Model Training (2-5 min)

```bash
python scripts/train_xgboost_model.py
```

**Expected performance:**
- Training accuracy: 95%+
- Cross-validation accuracy: 85-92% (depending on data quality)
- Inference time: 1-5ms per gesture (real-time capable)

**Interpretation:**
- Accuracy ≥ 90% → Excellent data quality
- Accuracy 85-89% → Good data quality
- Accuracy < 85% → Re-check data quality or collect more

### Phase 5: Real-Time Testing (5+ min)

```bash
python src/main.py
```

**Test scenarios:**
- [ ] Make each gesture in turn
- [ ] Verify correct recognition
- [ ] Test different lighting conditions
- [ ] Test different distances
- [ ] Test different hand sizes
- [ ] Test with both hands

**Expected behavior:**
- Gesture recognized within 100ms
- Shows gesture name with confidence score
- Plays TTS output
- Handles natural hand variations

## Detailed Resources

### 📖 Documentation
- **[COLLECTION_WORKFLOW.md](COLLECTION_WORKFLOW.md)** - Step-by-step guide
- **[STATIC_GESTURE_DATASET.md](STATIC_GESTURE_DATASET.md)** - Detailed reference
- **[GESTURE_QUICK_REFERENCE.md](GESTURE_QUICK_REFERENCE.md)** - Visual guide
- **[DATA_COLLECTION.md](DATA_COLLECTION.md)** - Data collection best practices
- **[LANDMARK_NORMALIZATION.md](LANDMARK_NORMALIZATION.md)** - Technical details

### 🛠 Tools
- **verify_gesture_landmarks.py** - Display gesture configurations
- **collect_training_data.py** - Data collection with quality validation
- **analyze_dataset.py** - Analyze dataset quality and balance
- **train_xgboost_model.py** - Train the classifier
- **src/main.py** - Real-time gesture recognition

## Dataset Structure

### CSV Format
```csv
label,left_x0,left_y0,left_z0,...,left_z20,right_x0,right_y0,...,right_z20
Closed Fist,0.45,0.32,0.001,...,0.75,-0.15,-0.08,...,-0.25
Open Palm,0.50,0.50,0.001,...,0.95,-0.50,-0.50,...,-0.95
```

**Columns:**
- 1st: Label (gesture name)
- 2-64: Left hand landmarks (21 × 3 coordinates)
- 65-127: Right hand landmarks (21 × 3 coordinates)

**Preprocessing:**
- Landmarks normalized to wrist origin
- Scaled by hand size (wrist-to-MCP distance)
- Missing hands zero-padded

**File location:** `data/training_data/landmark_data.csv`

## Quality Requirements

### Automatic Rejection (via SampleValidator)
The collection script automatically rejects samples:
- ❌ Hand detection confidence < 0.7
- ❌ Hand partially outside frame (not fully visible)
- ❌ Landmark confidence < 0.5 (blurry/unclear)
- ❌ Duplicate detection (cosine similarity > 0.95)

### Manual Quality Checks
Before accepting a collection session:
- ✓ All finger landmarks clearly visible
- ✓ No motion blur
- ✓ Hand contrast with background
- ✓ Correct gesture configuration
- ✓ Even lighting (no harsh shadows)

## Expected Performance

### By Dataset Size
| Dataset Size | Training Acc | Cross-Val Acc | Status |
|---|---|---|---|
| 100 samples | 92% | 75% | Proof of concept |
| 200 samples | 94% | 82% | Minimum viable |
| 500 samples | 96% | 88% | Recommended |
| 1000+ samples | 97% | 91% | Production |

### By Gesture
Most difficult gestures to classify:
1. **Vulcan Salute** - Complex 5-finger pattern with gap
2. **OK Sign** - Requires precise thumb-index proximity
3. **Peace Sign** - Distinguishing from other 2-finger gestures

Most distinct gestures:
1. **Open Palm** - Maximum spread
2. **Closed Fist** - Maximum cluster
3. **Thumbs Up/Down** - Vertical extremes

## Troubleshooting

### Collection Issues

**Problem: High rejection rate (> 30%)**
- Low confidence → Improve lighting, move closer
- Not fully visible → Center hand in frame
- Too similar → Vary hand position/rotation between captures

**Problem: Missing gesture recognition**
- Gesture not collected → Add it to collection
- Gesture imbalance → Collect more of underrepresented gesture
- Wrong gesture config → Review verify_gesture_landmarks.py

### Model Issues

**Problem: Low accuracy (< 80%)**
- Poor data quality → Check acceptance rate (> 85% needed)
- Gesture execution wrong → Review collected video
- Insufficient samples → Collect more
- Class imbalance → Balance dataset

**Problem: Slow inference (> 100ms)**
- System overloaded → Close other applications
- CPU limited → Check CPU usage (should be < 20%)
- Resolution too high → Lower video resolution

### Technical Issues

**Problem: "Cannot open webcam"**
- Webcam in use → Close other apps (Zoom, Teams, etc.)
- Permissions denied → Grant camera access in OS settings
- Camera disconnected → Check USB connection

**Problem: "ModuleNotFoundError"**
- Dependencies missing → Run: `pip install -r requirements.txt`
- Environment not activated → Activate venv: `.\venv\Scripts\activate`

## Feature Highlights

✅ **High accuracy** - 85-92% cross-validation on balanced datasets
✅ **Real-time** - 1-5ms inference per gesture (200+ fps)
✅ **Lightweight** - Fits on CPU-only machines
✅ **Robust** - Normalized features handle size/position/distance variation
✅ **Quality control** - Automatic rejection of low-quality samples
✅ **User-friendly** - Interactive collection UI with live feedback
✅ **Production-ready** - Multi-user, edge-case tested

## Next Steps

1. **Start now:**
   ```bash
   python scripts/verify_gesture_landmarks.py
   ```

2. **Collect data:**
   ```bash
   python scripts/collect_training_data.py
   ```

3. **Check quality:**
   ```bash
   python scripts/analyze_dataset.py
   ```

4. **Train model:**
   ```bash
   python scripts/train_xgboost_model.py
   ```

5. **Test real-time:**
   ```bash
   python src/main.py
   ```

## Support & Resources

- **Gesture not recognized?** → Run verify_gesture_landmarks.py
- **Model accuracy low?** → Check analyze_dataset.py output
- **Need more details?** → See STATIC_GESTURE_DATASET.md
- **Step-by-step guide?** → See COLLECTION_WORKFLOW.md
- **Technical questions?** → See LANDMARK_NORMALIZATION.md

---

**Happy collecting! Build your custom gesture dataset in 1-2 hours and get 85%+ accuracy with real-time recognition! 🎉**
