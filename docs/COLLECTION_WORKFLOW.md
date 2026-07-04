# Dataset Collection Workflow

## Quick Start Guide

This workflow will guide you through collecting a complete static gesture dataset optimized for MediaPipe Hands + XGBoost classification.

**Total time:** 1-2 hours (for 500 samples across 10 gestures)

---

## Step 1: Preparation (5 minutes)

### 1.1 Review Gestures
```bash
# Start the verification tool to see what each gesture looks like
python scripts/verify_gesture_landmarks.py
```

**Controls:**
- N = Next gesture
- P = Previous gesture  
- SPACE = Show detailed finger positions
- ESC = Exit

**What to do:**
- Browse through all 10 gestures
- Pay attention to "Key Features" and "Finger Positions"
- Watch the hints carefully

### 1.2 Prepare Collection Environment
Checklist:
- [ ] Good lighting (bright room, avoid harsh shadows)
- [ ] Clear background (white wall preferred, contrasts with hand)
- [ ] Webcam positioned at chest height
- [ ] Camera 1-1.5 meters away from you
- [ ] Background space clear (no clutter)
- [ ] Clean webcam lens

### 1.3 Setup Collection Script
```bash
# Verify collection script works
python scripts/collect_training_data.py
# (You'll see the interface but won't save yet - press ESC to cancel)
```

---

## Step 2: Test Run (10 minutes)

### 2.1 Collect 2 Sample Gestures
Start fresh collection:
```bash
python scripts/collect_training_data.py
```

Choose: **Start with "Closed Fist"** (easiest)

### 2.2 Capture Process
1. **Position your hand**
   - Make the gesture (all fingers curled inward)
   - Hold steady, no movement
   - Full hand visible in camera
   - Centered in frame

2. **Capture a sample**
   - Press SPACE for single capture
   - Watch the overlay for feedback
   - Should see: "✓ Accepted" (green)

3. **Repeat 5 times**
   - Vary hand position slightly between captures
   - Move hand up/down/left/right slightly
   - Change hand rotation slightly
   - Watch acceptance rate

### 2.3 Check Feedback
**Expected messages:**
- ✓ "Accepted (#1)" = Good! Sample saved
- ✗ "Low confidence (0.62 < 0.7)" = Move closer or improve lighting
- ✗ "Not fully visible" = Move hand fully into frame
- ✗ "Too similar to previous" = Vary hand position/rotation

### 2.4 Move to Next Gesture
- Press N to move to "Open Palm"
- Repeat capture process
- Collect 5 samples

### 2.5 Exit and Save
- Press ESC to finish
- You'll see summary:
  ```
  Samples per sign:
    ✓ Closed Fist : 5 samples
    ✓ Open Palm   : 5 samples
    
  Quality control statistics:
    • Total attempts    : 12
    • Accepted          : 10
    • Acceptance rate   : 83.3%
  ```

**Your CSV file:** `data/training_data/landmark_data.csv`

---

## Step 3: Full Dataset Collection

### 3.1 Collection Plan

**Phase 1: Initial Dataset (30 mins)**
- 20 samples per gesture = 200 total samples
- Single user (you)
- Right hand only (for now)
- Natural lighting

**Phase 2: Enhancement (30 mins)**
- 30 additional samples per gesture = 300 total
- Both hands (if possible, or ask a friend)
- Different lighting conditions
- Different distances

**Phase 3: Production (30 mins)**
- 50+ samples per gesture = 500+ total
- Multiple variations

### 3.2 Collection Session 1 (Right Hand, Initial)

**Duration:** 30 minutes

```bash
python scripts/collect_training_data.py
```

**Gesture order:** Easy to hard
1. ✋ Open Palm (2 min)
2. ✊ Closed Fist (2 min)
3. 👍 Thumbs Up (2 min)
4. 👎 Thumbs Down (2 min)
5. ☝️ Index Finger (3 min)
6. 🤙 Shaka (3 min)
7. 🤟 I Love You (3 min)
8. ✌️ Peace Sign (3 min)
9. 👌 OK Sign (4 min)
10. 🖖 Vulcan Salute (5 min)

**Per gesture:**
- Capture 20 samples
- Press SPACE for each sample
- Vary position/rotation/distance slightly
- Watch for "Too similar" rejections (means duplicate - skip)
- Typical acceptance rate: 85-90%

**Tips for success:**
- Take breaks between gestures
- Adjust lighting if many rejects
- Keep hand centered and fully visible
- Don't force gestures - be natural

### 3.3 Collection Session 2 (If Available: Left Hand or Different User)

**Duration:** 30-60 minutes

```bash
python scripts/collect_training_data.py
# Note: Appends to existing CSV (won't overwrite)
```

**Collect 30 additional samples per gesture**
- Same 10 gestures in same order
- Use left hand if different from Session 1
- OR use different person/lighting
- OR use different distance (0.5m and 2m to test range)

### 3.4 Collection Session 3 (Additional Edge Cases)

**Duration:** 30 minutes

Focus on:
- Difficult gestures (Peace Sign, OK Sign, Vulcan Salute)
- Low-light conditions
- Far distance (test robustness)
- Unusual hand sizes

```bash
python scripts/collect_training_data.py
```

---

## Step 4: Dataset Validation

### 4.1 Check CSV File
```bash
# Count total samples
wc -l data/training_data/landmark_data.csv

# Expected output for ~500 samples:
# 501 data/training_data/landmark_data.csv
# (1 header + 500 data rows)

# Check format (first 3 lines)
head -3 data/training_data/landmark_data.csv
```

**Expected format:**
```csv
label,left_x0,left_y0,left_z0,left_x1,...right_z20
Closed Fist,0.45,0.32,0.001,0.43,...
Open Palm,0.50,0.50,0.001,0.65,...
```

### 4.2 Verify Sample Distribution
```bash
# Count samples per gesture
python scripts/check_dataset_balance.py
```

**Expected output:**
```
Gesture Distribution:
  Closed Fist    : 50 samples (10.0%)
  Open Palm      : 50 samples (10.0%)
  Thumbs Up      : 50 samples (10.0%)
  Thumbs Down    : 50 samples (10.0%)
  Index Finger   : 50 samples (10.0%)
  Peace Sign     : 50 samples (10.0%)
  OK Sign        : 50 samples (10.0%)
  I Love You     : 50 samples (10.0%)
  Shaka          : 50 samples (10.0%)
  Vulcan Salute  : 50 samples (10.0%)
  
Total: 500 samples
Balance: GOOD (all within ±10%)
```

### 4.3 Quality Metrics
**What to look for:**
- ✅ Total samples >= 200 (minimum viable)
- ✅ All gestures represented (not missing any)
- ✅ Balanced distribution (each gesture similar count)
- ✅ Acceptance rate >= 85% during collection

---

## Step 5: Training

### 5.1 Train XGBoost Model
```bash
python scripts/train_xgboost_model.py
```

**Expected output:**
```
Loading training data from CSV...
  Samples: 500
  Features: 126 (normalized landmarks)
  Classes: 10 (gestures)

Training XGBoost classifier...
  Estimators: 300
  Max depth: 5
  Learning rate: 0.08
  Subsample: 0.8

Cross-validation scores: 
  Fold 1: 88.5%
  Fold 2: 87.2%
  Fold 3: 89.1%
  Fold 4: 86.8%
  Fold 5: 88.3%
  
Average accuracy: 88.0% ± 1.0%

Model saved to: data/models/sign_model.pkl
```

### 5.2 Interpret Results

**Expected accuracy ranges:**
- Dataset size 200 samples: 75-80% (proof of concept)
- Dataset size 500 samples: 85-92% (recommended)
- Dataset size 1000+ samples: 90-95% (production)

**If accuracy is low (< 80%):**
1. ✓ Check dataset quality (acceptance rate during collection)
2. ✓ Check gesture execution (review video of your poses)
3. ✓ Check dataset balance (use check_dataset_balance.py)
4. ✓ Collect more samples (especially for difficult gestures)

**If accuracy is good (> 85%):**
- Model is ready for deployment!

---

## Step 6: Real-Time Testing

### 6.1 Run Recognition System
```bash
python src/main.py
```

**What to do:**
1. Stand in front of camera
2. Make each gesture in turn
3. Watch for recognition
4. Test accuracy by comparing to expected output

**Expected behavior:**
- ✓ Recognized within 100ms
- ✓ Shows gesture name and confidence
- ✓ Plays TTS (text-to-speech) output
- ✓ Handles variations (hand size, position, lighting)

### 6.2 Test Real-World Conditions
Try these scenarios:
- [ ] Different lighting (normal, bright, dim)
- [ ] Different distances (0.5m, 1m, 2m)
- [ ] Different backgrounds
- [ ] Left hand vs right hand
- [ ] Fast gestures vs slow
- [ ] Different hand sizes

### 6.3 Log Misclassifications
If a gesture is recognized incorrectly:
1. Note which gesture was intended
2. Note which gesture was recognized
3. Try to collect more similar examples
4. Retrain model

---

## Step 7: Optimization (Optional)

### 7.1 Collect Hard Negatives
If model confuses gestures A and B:
1. Collect 20 additional samples of both A and B
2. Emphasize their differences
3. Retrain model

### 7.2 Expand to Production Dataset
Collect 1000+ samples total:
- Multiple users (5+)
- Both hands evenly
- All lighting conditions
- All distances
- Edge cases

```bash
# Repeat collection sessions with different people
python scripts/collect_training_data.py
```

### 7.3 Retrain and Validate
```bash
python scripts/train_xgboost_model.py
python src/main.py
```

---

## Troubleshooting

### "Too many rejections"

**Problem:** > 30% of samples rejected

**Causes & fixes:**
- [ ] Low confidence rejection → Improve lighting, move closer
- [ ] Not fully visible → Move hand more toward center
- [ ] Too similar to previous → Vary hand position more between captures

**Action:** 
1. Adjust environment (lighting, distance)
2. Re-read gesture definition (might be wrong)
3. Redo the gesture more carefully

### "Low model accuracy (< 80%)"

**Problem:** Model is misclassifying gestures

**Causes & fixes:**
- [ ] Poor data quality → Re-check acceptance rate, look for rejects
- [ ] Gesture execution wrong → Review via camera, compare to reference
- [ ] Insufficient samples → Collect more (especially difficult gestures)
- [ ] Dataset imbalance → Check gesture counts, balance if needed

**Action:**
1. Check collection acceptance rate (target >= 85%)
2. Verify gesture configurations
3. Collect more data
4. Retrain

### "Slow inference (> 100ms)"

**Problem:** Real-time performance is poor

**Note:** This is NOT a data collection issue. This indicates:
- System CPU overloaded
- Other applications running
- Video resolution too high

**Action:**
1. Close other applications
2. Check CPU usage (should be < 20%)
3. Lower video resolution if needed

---

## Success Checklist

- [ ] Reviewed all 10 gesture definitions
- [ ] Environment prepared (lighting, background, camera)
- [ ] Tested collection script
- [ ] Collected 200+ samples
- [ ] All 10 gestures represented
- [ ] Gesture distribution balanced
- [ ] Acceptance rate >= 85%
- [ ] Trained model successfully
- [ ] Cross-validation accuracy >= 85%
- [ ] Real-time testing works
- [ ] Recognized gestures correctly

---

## Next Steps

### Option 1: Deploy Now
If model accuracy > 85%:
1. Use `python src/main.py` for real-time recognition
2. Integrate into your application
3. Continue collecting feedback data

### Option 2: Optimize First
If model accuracy < 85% or want better:
1. Analyze confusion matrix (which gestures confuse the model?)
2. Collect more hard negative samples
3. Retrain model
4. Repeat until satisfied

### Option 3: Expand Dataset
For production deployment:
1. Recruit multiple users
2. Collect more samples per gesture (100+)
3. Include more edge cases (lighting, distance, occlusion)
4. Retrain with larger dataset
5. Validate against test set

---

## Data Files

**CSV file:** `data/training_data/landmark_data.csv`
- Format: label, 126 normalized landmark features
- One sample per row
- 127 columns total (label + features)

**Model file:** `data/models/sign_model.pkl`
- XGBoost classifier
- Label encoder
- Saved automatically after training

**Logs:** Check `logs/` directory
- Collection logs (if logging enabled)
- Training logs
- Inference logs

---

## Additional Resources

- **Gesture Reference:** `docs/GESTURE_QUICK_REFERENCE.md`
- **Detailed Guide:** `docs/STATIC_GESTURE_DATASET.md`
- **Data Collection Details:** `docs/DATA_COLLECTION.md`
- **Landmark Normalization:** `docs/LANDMARK_NORMALIZATION.md`

---

## Support

**Stuck on a gesture?**
- Run `python scripts/verify_gesture_landmarks.py`
- Review detailed finger positions
- Compare your hand to the description

**Model not working?**
- Check acceptance rate during collection (> 85% needed)
- Verify gesture execution (review video)
- Check dataset balance
- Collect more samples

**Questions about landmarks?**
- See `docs/LANDMARK_NORMALIZATION.md`
- Check MediaPipe documentation
- Review the landmark indices (0-20)

---

Good luck with your dataset collection! 🎉
