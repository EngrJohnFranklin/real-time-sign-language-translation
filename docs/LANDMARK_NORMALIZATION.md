# Landmark Normalization Preprocessing

## Overview

The `landmark_normalizer` module provides consistent preprocessing of MediaPipe hand landmarks across the entire pipeline:
- **Data collection** (`scripts/collect_training_data.py`)
- **Model training** (`scripts/train_xgboost_model.py`)
- **Real-time inference** (`src/models/sign_detector.py`)

Using the same normalization everywhere prevents train/test skew and makes the model robust to hand size and distance variations.

## Normalization Process

### Single-Hand Normalization

Each hand's 21 landmarks are normalized independently:

1. **Translate to wrist origin**
   - Wrist landmark (index 0) becomes (0, 0, 0)
   - All other landmarks are shifted by the same amount
   - Result: Translation-invariant features

2. **Scale by hand size**
   - Hand size = Euclidean distance from wrist to middle-finger MCP (landmark 9)
   - All coordinates divided by hand size
   - Result: Scale-invariant features (robust to camera distance)

3. **Flatten to 63-element vector**
   - 21 landmarks × 3 axes (x, y, z) = 63 features
   - Includes z-axis (depth) for better pose discrimination

### Example

Raw wrist: (640.0, 480.0, 0.5)  
Raw middle-finger MCP: (650.0, 400.0, 0.5)  

After translation:
```
Wrist: (0, 0, 0)
Middle MCP: (10, -80, 0)
```

Hand size = √(10² + 80² + 0²) ≈ 80.62

After scaling:
```
Wrist: (0, 0, 0)
Middle MCP: (0.124, -0.993, 0)
```

### Dual-Hand Features

Two hands (or one + padding) are combined into a 126-element vector:

```
Features = [Left Hand (63) | Right Hand (63)]
         = [left_x0, left_y0, left_z0, ..., left_z20, right_x0, right_y0, ..., right_z20]
```

- If a hand is missing, its 63 elements are filled with zeros
- Left hand is always first (consistent ordering)

## API Reference

### `normalize_landmarks(landmarks, min_scale=1e-6)`

Normalize a single hand's landmarks.

**Parameters:**
- `landmarks`: List of 21 (x, y, z) tuples from MediaPipe, or None
- `min_scale`: Minimum hand size threshold (prevents divide-by-zero)

**Returns:**
- 63-element numpy array (float32) if valid, else None
- Returns None if input is invalid or hand size too small

**Example:**
```python
from utils.landmark_normalizer import normalize_landmarks

landmarks = [(x, y, z), ...]  # 21 MediaPipe hand landmarks
normalized = normalize_landmarks(landmarks)
if normalized is not None:
    print(f"Normalized shape: {normalized.shape}")  # (63,)
```

### `normalize_dual_hand_features(left_landmarks=None, right_landmarks=None, min_scale=1e-6)`

Normalize up to 2 hands and combine into a 126-element vector.

**Parameters:**
- `left_landmarks`: List of 21 (x, y, z) tuples for left hand, or None
- `right_landmarks`: List of 21 (x, y, z) tuples for right hand, or None
- `min_scale`: Minimum hand size threshold

**Returns:**
- Always returns 126-element numpy array (float32)
- Missing hands are represented as 63 zeros
- Never returns None

**Example:**
```python
from utils.landmark_normalizer import normalize_dual_hand_features

left_lms = [...]  # 21 tuples or None
right_lms = [...]  # 21 tuples or None
features = normalize_dual_hand_features(left_lms, right_lms)
print(f"Features shape: {features.shape}")  # (126,)
```

### `normalize_batch_landmarks(landmarks_batch, min_scale=1e-6)`

Normalize a batch of landmarks for training.

**Parameters:**
- `landmarks_batch`: List of landmark sequences, each with 21 (x, y, z) tuples
- `min_scale`: Minimum hand size threshold

**Returns:**
- Array of shape (batch_size, 63) with normalized features
- Invalid landmarks replaced with zeros

**Example:**
```python
from utils.landmark_normalizer import normalize_batch_landmarks

batch = [[...], [...], [...]]  # 3 hands, each with 21 landmarks
normalized = normalize_batch_landmarks(batch)
print(f"Batch shape: {normalized.shape}")  # (3, 63)
```

## Integration Points

### 1. Data Collection (`scripts/collect_training_data.py`)

Collects raw landmarks and saves normalized features to CSV:

```python
from models.xgboost_classifier import XGBoostSignClassifier

# Record raw landmarks from MediaPipe
left_landmarks = [(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark]

# Normalize and extract features
features = XGBoostSignClassifier.extract_features(left_landmarks, right_landmarks)
# features is already normalized via landmark_normalizer

# Save to CSV (127 columns: label + 126 features)
row = [sign_name] + features.tolist()
csv_writer.writerow(row)
```

**Output:** `data/training_data/landmark_data.csv` with normalized features

### 2. Model Training (`scripts/train_xgboost_model.py`)

Loads pre-normalized features from CSV:

```python
from scripts.train_xgboost_model import load_dataset

X, y = load_dataset("data/training_data/landmark_data.csv")
# X shape: (num_samples, 126) — already normalized
# y: string labels (sign names)

# Train XGBoost
xgb_model = xgb.XGBClassifier(...)
xgb_model.fit(X, y_encoded)
```

**Output:** `data/models/sign_model.pkl` trained on normalized features

### 3. Real-Time Inference (`src/models/sign_detector.py`)

Normalizes landmarks on-the-fly during prediction:

```python
from models.xgboost_classifier import XGBoostSignClassifier

# Receive raw landmarks from MediaPipe
left_landmarks = [...]  # 21 tuples
right_landmarks = [...]  # 21 tuples

# Normalize and predict
sign, confidence = classifier.predict_both_hands(left_landmarks, right_landmarks)
# Normalization happens internally via landmark_normalizer
```

## Robustness Features

### Divide-by-Zero Prevention

If hand size < `min_scale` (default 1e-6), returns None:

```python
from utils.landmark_normalizer import normalize_landmarks

degenerate = [(0.0, 0.0, 0.0)] * 21  # All landmarks at origin
result = normalize_landmarks(degenerate)
# result is None — detected invalid hand pose
```

### Missing Hand Handling

If a hand is not detected, it's represented as 63 zeros:

```python
from utils.landmark_normalizer import normalize_dual_hand_features

only_left = [...]  # 21 tuples
features = normalize_dual_hand_features(only_left, None)
# features[:63] = normalized left hand
# features[63:] = all zeros (right hand padding)
print(features[63:] == 0)  # All True
```

### Consistent Preprocessing

Same normalization used everywhere:

```
Collection → normalize_landmarks → CSV features
    ↓
Training → load_dataset → read CSV features → train
    ↓
Inference → normalize_landmarks → predict
```

**Result:** No train/test skew from preprocessing differences.

## Performance Characteristics

- **Normalization time per hand:** < 0.1 ms (NumPy operations)
- **Total pipeline latency:** < 5 ms (hand detection + normalization + XGBoost inference)
- **Memory per sample:** 126 float32 values = 504 bytes

## Tips

1. **Always normalize landmarks before training:** Use `normalize_dual_hand_features()` or `XGBoostSignClassifier.extract_features()`

2. **Use consistent ordering:** Left hand always first (indices 0-62), right hand second (indices 63-125)

3. **Expect ~0-1 for normalized features:** After translation and scaling, most coordinates are in [-1, 1] range

4. **Handle missing hands:** The pipeline automatically pads with zeros. No special handling needed.

5. **Debug preprocessing issues:** Check hand size is > 1e-6 (sanity check for valid poses)

## References

- **MediaPipe Hands:** https://google.github.io/mediapipe/solutions/hands.html
- **Landmark indices:** 0=Wrist, 9=Middle MCP, 4=Thumb tip, 8=Index tip, etc.
- **XGBoost integration:** `src/models/xgboost_classifier.py`
