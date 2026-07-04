# Static Gesture Dataset Collection Guide

## Overview

This guide explains how to collect a custom static gesture dataset optimized for MediaPipe Hands + XGBoost classification. The dataset consists of 10 single-frame hand gestures, each with distinct landmark patterns suitable for real-time recognition.

**Key Characteristics:**
- **Static gestures only** - No hand movement within a frame
- **21 landmarks per hand** - x, y, z coordinates from MediaPipe
- **126-element feature vectors** - Normalized (wrist origin + hand size scaled)
- **High class separability** - Each gesture has unique hand configuration
- **Real-time optimized** - Fast inference on CPU with XGBoost

## The 10 Gestures

### Hand Configuration Reference

```
Landmark Index Positions (MediaPipe 21-point model):
  0 = Wrist
  1-4 = Thumb (base to tip)
  5-8 = Index finger (base to tip)
  9-12 = Middle finger (base to tip)
  13-16 = Ring finger (base to tip)
  17-20 = Pinky finger (base to tip)
```

### 1. Closed Fist (Sorry)
**Hand Configuration:**
- All fingers fully flexed/closed
- Thumb tucked inside or resting on fingers
- Palm facing forward or to side
- Clear spherical/compact hand shape

**Distinctiveness:** Very compact, minimal spread
**Difficulty:** ⭐ Easy - All landmarks clustered tightly
**Key Landmarks:** All 21 landmarks in tight cluster

---

### 2. Open Palm (Stop)
**Hand Configuration:**
- All 5 fingers fully extended and spread
- Palm flat and facing camera/forward
- Fingers slightly separated (natural spread)
- Wrist straight

**Distinctiveness:** Maximum spread, all landmarks extended
**Difficulty:** ⭐ Very Easy - Maximum separation
**Key Landmarks:** All finger tips (4, 8, 12, 16, 20) widely separated

---

### 3. Thumbs Up (Yes)
**Hand Configuration:**
- Thumb pointing upward (extended)
- Other 4 fingers flexed/closed
- Wrist in neutral or slightly extended position
- Thumb creates vertical line from wrist

**Distinctiveness:** Single extended digit (thumb), very recognizable
**Difficulty:** ⭐ Very Easy - Clear vertical thumb
**Key Landmarks:** Thumb tip (4) high above wrist (0), other fingers clustered

---

### 4. Thumbs Down (No)
**Hand Configuration:**
- Thumb pointing downward (extended)
- Other 4 fingers flexed/closed
- Mirror of Thumbs Up gesture
- Wrist in neutral or extended position

**Distinctiveness:** Single extended digit pointing opposite direction
**Difficulty:** ⭐ Very Easy - Clear distinction from Thumbs Up
**Key Landmarks:** Thumb tip (4) low relative to wrist (0), other fingers clustered

---

### 5. Index Finger Up (Help)
**Hand Configuration:**
- Index finger fully extended pointing upward
- All other fingers (middle, ring, pinky) flexed/closed
- Thumb may be flexed or slightly extended
- Wrist in neutral position

**Distinctiveness:** Single extended digit (index), different from thumb
**Difficulty:** ⭐ Easy - Clear single extension different from thumb
**Key Landmarks:** Index tip (8) high above wrist, other landmarks clustered

---

### 6. Peace Sign (Thank You)
**Hand Configuration:**
- Index and middle fingers extended in a "V" shape
- Fingers spread apart (distinct gap between them)
- Ring and pinky fingers flexed/closed
- Thumb may be flexed or extended slightly
- Palm facing forward or to side

**Distinctiveness:** Two extended digits with characteristic gap
**Difficulty:** ⭐⭐ Medium - Two extended fingers with clear separation
**Key Landmarks:** Index tip (8) and middle tip (12) separated, ring/pinky clustered

---

### 7. OK Sign (Good)
**Hand Configuration:**
- Thumb and index finger form a circle (touching or very close)
- Middle, ring, and pinky fingers fully extended upward
- Fingers held separately, not touching
- Hand orientation typically palm-down or palm-facing-camera

**Distinctiveness:** Unique circle formation + three extended fingers
**Difficulty:** ⭐⭐ Medium - Requires precise thumb-index relationship
**Key Landmarks:** Thumb tip (4) and index tip (8) close together, middle/ring/pinky (12, 16, 20) extended high

---

### 8. I Love You (I ♡ U)
**Hand Configuration:**
- Thumb, index, and pinky fingers extended
- Middle and ring fingers flexed/folded down
- Creates distinctive three-point pattern
- All extended fingers separated and clear

**Distinctiveness:** Unique three-finger pattern (non-consecutive)
**Difficulty:** ⭐⭐ Medium - Three extended digits with gap in middle
**Key Landmarks:** Thumb (4), index (8), and pinky (20) extended; middle (12) and ring (16) clustered

---

### 9. Shaka (Hello)
**Hand Configuration:**
- Thumb and pinky extended
- Index, middle, and ring fingers flexed/closed
- Creates distinctive "hang loose" gesture
- Wrist may be rotated, hand orientation variable

**Distinctiveness:** Thumb + pinky extended, middle fingers closed
**Difficulty:** ⭐⭐ Medium - Two extended non-adjacent digits
**Key Landmarks:** Thumb (4) and pinky (20) extended; index/middle/ring (8, 12, 16) clustered

---

### 10. Vulcan Salute (Goodbye)
**Hand Configuration:**
- Middle and ring fingers form a "V" gap (not touching each other)
- Thumb, index, and pinky fully extended
- Wrist neutral, hand raised
- Gap between middle and ring creates distinctive pattern

**Distinctiveness:** Five fingers present but with unique gap
**Difficulty:** ⭐⭐⭐ Hard - Complex multi-finger configuration
**Key Landmarks:** All five fingers extended with characteristic gap between landmarks 12 (middle) and 16 (ring)

---

## Collection Strategy

### Phase 1: Core Collection

**Objective:** Gather initial diverse samples for each gesture

**Requirements per gesture:**
- Minimum 20 samples to start
- Multiple users (2-3)
- Both hands (left & right) 
- Various distances (0.5m - 2m from camera)
- Different lighting conditions (natural, artificial, low-light)
- Different backgrounds (white wall, cluttered, outdoors)

**Example collection sequence:**
```
User 1, Right Hand:
  - Closed Fist (5 samples x 3 distances x 2 lighting = 30 samples)
  - Open Palm (5 samples x 3 distances x 2 lighting = 30 samples)
  - [repeat for all 10 gestures]

User 1, Left Hand:
  - [repeat all 10 gestures]

User 2, Right Hand:
  - [repeat all 10 gestures]

User 2, Left Hand:
  - [repeat all 10 gestures]
```

### Phase 2: Quality Enhancement

**Objective:** Improve class separability and balance

For each gesture:
1. **Analyze mistakes** - Which gestures are misclassified?
2. **Add hard negatives** - Collect examples of easily confused pairs
3. **Improve underrepresented conditions** - Add more low-light, far-distance, etc.
4. **Balance dataset** - Equalize samples across classes

### Phase 3: Edge Cases

**Objective:** Handle real-world variations

Collect samples with:
- Partial hand occlusion (fingers partially cut off)
- Fast hand motion (introduce slight blur but keep recognizable)
- Unusual hand sizes (very large or small hands)
- Hand rotations (different viewing angles)
- Overlapping hands (both hands partially visible)

---

## Quality Requirements

### Rejection Criteria

Automatically rejected by SampleValidator:

✗ **Low Confidence:** Hand detection confidence < 0.7
- **Fix:** Improve lighting, reduce distance, ensure full hand visibility

✗ **Incomplete Hand:** Any landmark outside 0-1 normalized frame
- **Fix:** Move hand fully into camera view

✗ **Low Landmark Confidence:** Per-landmark confidence < 0.5
- **Fix:** Improve lighting, reduce motion blur, adjust hand position

✗ **Duplicate Samples:** Cosine similarity > 0.95 to previous sample
- **Fix:** Vary hand position/angle slightly between captures

### Manual Quality Checks

Before accepting a collection session:

| Check | Good | Bad |
|-------|------|-----|
| **Hand Position** | Fully visible in frame | Partial/cut off |
| **Finger Clarity** | All fingers clearly defined | Blurred/merged fingers |
| **Motion Blur** | No visible blur | Visible hand movement |
| **Lighting** | Even illumination | Shadows/glare on hand |
| **Background** | Contrasts with hand | Similar color to hand |
| **Gesture Definition** | Correct hand config | Ambiguous/wrong pose |

---

## Maximum Class Separability Tips

### Distance Metrics (Euclidean in Landmark Space)

**Expected separability per gesture pair:**

| Gesture Pair | Separability | Reason |
|---|---|---|
| Open Palm ↔ Closed Fist | Excellent | Complete opposite |
| Thumbs Up ↔ Thumbs Down | Excellent | Mirror gestures |
| Thumbs Up ↔ Index Finger | Good | Different extended finger |
| Peace Sign ↔ I Love You | Medium | Both have 2 extended, different fingers |
| Vulcan Salute ↔ I Love You | Medium | Similar finger count, different pattern |

### Confusion Matrices to Monitor

**Likely confusions** (watch during training):
1. Peace Sign ↔ Vulcan Salute (both have multiple extended fingers)
2. I Love You ↔ Shaka (both have non-consecutive fingers extended)
3. Thumbs Up ↔ Index Finger (both single upward extension)

**Mitigation:**
- Collect extra samples of confusing pairs
- Ensure maximum hand size variation (normalizes these differences)
- Verify gesture execution (correct hand configuration)

---

## Optimal Sample Distribution

### Minimum Viable Dataset

For initial testing:
```
Total: 200 samples (20 per gesture)
Distribution: 10 gestures × 20 samples = 200
Split: 80% train (160) / 20% test (40)
Acceptance Rate Target: 85%+ (reject low quality)
```

### Recommended Dataset

For production deployment:
```
Total: 500 samples (50 per gesture)
Distribution:
  - 10 gestures × 50 samples = 500
  - Per gesture breakdown:
    * 20 samples: User 1, Right Hand
    * 20 samples: User 1, Left Hand
    * 10 samples: User 2, Right Hand
  - Variations included:
    * 3 distances: 0.5m, 1m, 1.5m (each 2-3 samples)
    * 2 lighting: Natural, Artificial (mix throughout)
    * 2+ backgrounds: White wall, Outdoor, Cluttered (distribute)

Split: 80% train (400) / 20% test (100)
Acceptance Rate Target: 90%+
Cross-validation: 5-fold, expect >85% accuracy
```

### Production Dataset

For commercial deployment:
```
Total: 1000+ samples (100+ per gesture)
Distribution:
  - 10 gestures × 100 samples = 1000
  - 5+ users (diverse hand sizes/shapes)
  - Both hands evenly distributed
  - All distance/lighting/background combinations
  - Edge cases (occlusion, motion, angles)

Split: 80% train (800) / 20% test (200)
Acceptance Rate Target: 95%+
Cross-validation: 5-fold, expect >92% accuracy
Robustness: Tested across devices/lighting
```

---

## Collection Workflow

### Using collect_training_data.py

```bash
cd /real-time-sign-language-translation
python scripts/collect_training_data.py
```

**Keyboard Controls:**
- **SPACE** - Capture single sample
- **R** - Start continuous capture (saves every valid frame)
- **S** - Stop capture
- **N** - Next gesture (auto-reset duplicate detection)
- **P** - Previous gesture (useful if you make a mistake)
- **ESC** - Exit and save

### Collection Session Template

```
1. Read gesture 1 description (Closed Fist)
2. Get into correct hand configuration
3. Press SPACE or R to capture 5-10 samples
4. Watch for feedback:
   - ✓ Sample accepted (green)
   - ✗ Sample rejected with reason (red)
5. Adjust pose if rejected, try again
6. Press N to move to next gesture
7. Repeat for all 10 gestures
8. Press ESC to save and exit
```

### Tips for Successful Collection

**Before Collection:**
- [ ] Review all 10 gesture configurations
- [ ] Test hand visibility (full hand in frame)
- [ ] Adjust camera distance (1-1.5m optimal)
- [ ] Check lighting (bright but not glaring)
- [ ] Clean webcam lens
- [ ] Ensure background contrast

**During Collection:**
- [ ] Hold each gesture steady (no motion)
- [ ] Keep hand fully visible throughout
- [ ] Vary subtle aspects (hand rotation, position)
- [ ] Capture multiple hand sizes per gesture
- [ ] Watch overlay feedback (green = good, red = rejected)
- [ ] Don't repeat identical poses (duplicates rejected)

**After Each Gesture:**
- [ ] Review collected samples count
- [ ] Check for any rejected samples and reason
- [ ] Take a short break before next gesture

**Data Quality Metrics:**
- Acceptance rate > 85% = High quality
- Acceptance rate 70-85% = Acceptable (may need re-collection)
- Acceptance rate < 70% = Poor quality (redo collection)

---

## CSV Output Format

**File:** `data/training_data/landmark_data.csv`

**Structure:**
```csv
label,left_x0,left_y0,left_z0,left_x1,left_y1,left_z1,...,right_x0,right_y0,right_z0,...
Closed Fist,0.45,0.32,0.001,0.43,0.30,0.002,...,-0.15,-0.08,0.05,...
Open Palm,0.50,0.50,0.001,0.65,0.45,0.002,...,-0.50,-0.50,0.001,...
[... one row per sample]
```

**Columns:**
- Column 0: `label` - Gesture name (one of the 10 gestures)
- Columns 1-63: Left hand landmarks (21 landmarks × 3 coordinates)
- Columns 64-126: Right hand landmarks (21 landmarks × 3 coordinates)

**Preprocessing:**
- All coordinates are **normalized**:
  - Translated so wrist (landmark 0) is at origin
  - Scaled by hand size (wrist-to-middle-finger-MCP distance)
  - Missing hands are zero-padded (63 zeros)
- This normalization ensures invariance to:
  - Hand size variation
  - Hand position in frame
  - Camera distance

---

## Training & Validation

### Training Command

```bash
python scripts/train_xgboost_model.py
```

**Expected Performance:**
- Training accuracy: 95%+ (128 features, normalized input)
- Cross-validation accuracy: 85-92% (varies by collection quality)
- Inference time: 1-5ms per sample (CPU, real-time capable)

### Validation Strategy

1. **Monitor per-class performance:**
   ```
   Closed Fist: 95%
   Open Palm: 98%
   Thumbs Up: 92%
   [...]
   ```

2. **Analyze confusion matrix:**
   - Identify which gestures are misclassified as what
   - Collect more samples for confused pairs

3. **Test real-time inference:**
   ```bash
   python src/main.py
   ```
   - Verify each gesture recognized correctly
   - Check latency (should be < 100ms per frame)

### If Accuracy is Low

1. **Check data quality:**
   - Acceptance rate should be > 85%
   - Check rejection statistics in collection output
   - Re-collect with better quality

2. **Check gesture execution:**
   - Verify each gesture matches exact hand configuration
   - Record a video to confirm poses
   - Cross-reference with gesture definitions

3. **Check dataset balance:**
   - Each gesture should have ~equal samples
   - If one gesture has 100 samples and another has 10, balance first

4. **Increase dataset size:**
   - Start with 20 per gesture, increase to 50
   - Add more user variation
   - Include more lighting/distance conditions

---

## Advanced: Feature Analysis

### Landmark Importance per Gesture

**Closed Fist:**
- Important: All finger tips (4, 8, 12, 16, 20) - should be close to wrist
- Less important: Individual finger joints (varies)

**Open Palm:**
- Important: All finger tips (4, 8, 12, 16, 20) - should be far from wrist
- Important: Finger spread (distance between tips)
- Less important: Individual joint angles

**Thumbs Up:**
- Important: Thumb tip (4) - should be high above wrist
- Important: Other finger tips (8, 12, 16, 20) - should be low/clustered
- Critical: Thumb position > other fingers

**[... similar analysis for other gestures]**

### Normalization Benefits

Why we normalize landmarks:

1. **Scale Invariance:** Large hands and small hands look identical
2. **Position Invariance:** Hand 1 inch from camera or 2 feet away looks identical
3. **Rotation Invariance:** Hand tilted differently still recognizable
4. **Reduced Dimensionality:** 126 features instead of raw coordinates

### Class Separability Measurement

Formula: **Cosine similarity between gesture centroids**
- Value 0 = completely different gestures (good)
- Value 1 = identical gestures (bad)
- Target: < 0.7 between any two gestures

---

## Troubleshooting

### "Low confidence" rejections (> 50%)
**Cause:** Hand not properly detected
- **Solution:** Improve lighting (avoid shadows), move closer (0.5-1m), ensure hand fully visible

### "Not fully visible" rejections (> 20%)
**Cause:** Hand landmarks outside frame or too close to edge
- **Solution:** Position hand in center of frame, use wider angle lens, move further back

### "Too similar to previous" rejections (> 30%)
**Cause:** Capturing identical poses repeatedly
- **Solution:** Vary hand position/rotation slightly between captures, move through frame naturally

### Model accuracy < 80%
**Cause:** Poor data quality or ambiguous gestures
- **Solution:** Re-collect with stricter quality control, verify gesture definitions, increase dataset size

### Slow inference (> 100ms)
**Cause:** System issue, not data related
- **Solution:** Close other applications, check CPU usage, reduce video resolution

---

## Summary Checklist

- [ ] Updated gesture vocabulary (10 new gestures)
- [ ] Review all 10 gesture configurations
- [ ] Set up collection environment (good lighting, clear background)
- [ ] Collect Phase 1: 20+ samples per gesture (all users, both hands)
- [ ] Verify acceptance rate > 85%
- [ ] Analyze dataset statistics
- [ ] Train initial model
- [ ] Validate performance (> 85% accuracy)
- [ ] Collect Phase 2: Hard negatives and edge cases
- [ ] Retrain model if accuracy improved
- [ ] Collect Phase 3: Full production dataset (100+ per gesture)
- [ ] Final validation and deployment testing

---

## Next Steps

1. **Start collection:** Run `collect_training_data.py`
2. **Monitor quality:** Watch for rejection reasons
3. **Train model:** Run `train_xgboost_model.py` after collecting
4. **Test inference:** Run `src/main.py` to verify real-time recognition
5. **Iterate:** Improve dataset based on model performance

Good luck with your dataset! The combination of static gestures, normalized landmarks, and XGBoost should provide excellent real-time recognition.
