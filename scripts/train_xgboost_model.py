"""
Train an XGBoost sign language classifier from collected landmark data.

Reads landmark_data.csv produced by collect_training_data.py. The CSV already
contains normalized features (126-element vectors) from landmark_normalizer:
- Landmarks translated to wrist origin (landmark 0)
- Scaled by hand size (wrist-to-middle-finger-MCP distance)
- Left hand (63 features) + right hand (63 features) = 126 total

Trains an XGBoost classifier with 5-fold cross-validation and saves the
trained model bundle to data/models/sign_model.pkl for use during inference.

Note: The same landmark_normalizer is used during real-time prediction,
ensuring training/inference consistency.

Usage:
    python scripts/train_xgboost_model.py
"""

import sys
import os
import csv
import pickle
import pathlib
import logging

import numpy as np

logger = logging.getLogger(__name__)

_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add src to path so we can reuse XGBoostSignClassifier for model structure
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from models.xgboost_classifier import XGBoostSignClassifier  # noqa: E402

DATA_PATH = _PROJECT_ROOT / "data" / "training_data" / "landmark_data.csv"
MODEL_PATH = _PROJECT_ROOT / "data" / "models" / "sign_model.pkl"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_dataset(data_path: pathlib.Path):
    """
    Load landmark CSV with pre-normalized features.
    
    The CSV contains 126-element vectors (left hand 63 + right hand 63).
    Each vector was normalized by collect_training_data.py using:
    1. Translate landmarks to wrist (landmark 0) origin
    2. Scale by hand size (wrist-to-middle-finger-MCP distance)
    3. Flatten to 63-element feature vector per hand
    
    These normalized features are used consistently during:
    - Training (this script)
    - Real-time inference (sign_detector.py via xgboost_classifier.py)
    
    This ensures no train/test skew from preprocessing differences.

    Returns:
        X: numpy array of shape (N, 126) with normalized features
        y: numpy array of string labels (sign names)
    """
    features = []
    labels = []
    skipped = 0

    with open(data_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            label = row.get("label", "").strip()
            if not label:
                skipped += 1
                continue

            # CSV format: label + left hand (63 values) + right hand (63 values)
            # Columns: label, left_x0, left_y0, left_z0, ..., left_z20, right_x0, ..., right_z20
            try:
                feature_values = []
                
                # Extract left hand features (63 values: 21 landmarks × 3 coordinates)
                for i_lm in range(21):
                    for ax in ("x", "y", "z"):
                        col_name = f"left_{ax}{i_lm}"
                        feature_values.append(float(row[col_name]))
                
                # Extract right hand features (63 values)
                for i_lm in range(21):
                    for ax in ("x", "y", "z"):
                        col_name = f"right_{ax}{i_lm}"
                        feature_values.append(float(row[col_name]))
                
                features.append(feature_values)
                labels.append(label)
            except (ValueError, KeyError) as e:
                logger.debug(f"Row {i}: skipping due to {e}")
                skipped += 1

    if skipped:
        print(f"  ⚠ Skipped {skipped} invalid rows")

    return np.array(features, dtype=np.float32), np.array(labels)


def main() -> None:
    print("=== XGBoost Sign Language Classifier Training ===\n")

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    if not DATA_PATH.exists():
        print(f"Error: Training data not found at {DATA_PATH}")
        print("Run this first:  python scripts/collect_training_data.py")
        sys.exit(1)

    print(f"Loading data from {DATA_PATH} ...")
    X, y = load_dataset(DATA_PATH)

    if len(X) == 0:
        print("Error: No valid samples found in the CSV.")
        sys.exit(1)

    expected_features = 126  # left hand (63) + right hand (63)
    if X.shape[1] != expected_features:
        print(f"Error: Expected {expected_features} features per sample, got {X.shape[1]}.")
        print("This usually means the CSV format is incorrect.")
        sys.exit(1)

    classes, counts = np.unique(y, return_counts=True)
    print(f"  {len(X)} samples  |  {len(classes)} classes  |  {X.shape[1]} features per sample\n")

    low_class = False
    for label, count in zip(classes, counts):
        status = "✓" if count >= 30 else "⚠ low"
        print(f"  {status}  {label}: {count} samples")
        if count < 30:
            low_class = True

    if low_class:
        print("\n  Tip: aim for 50+ samples per sign for reliable accuracy.\n")

    if len(X) < 50:
        print("\nError: Dataset too small. Collect at least 50 total samples.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Encode labels
    # ------------------------------------------------------------------ #
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ------------------------------------------------------------------ #
    # 3. Cross-validate
    # ------------------------------------------------------------------ #
    import xgboost as xgb  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: PLC0415

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        print("\nError: Need at least 2 samples per class for cross-validation.")
        sys.exit(1)

    print(f"Running {n_splits}-fold cross-validation (126-feature vectors) ...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    if scores.mean() < 0.70:
        print("\n  ⚠ Accuracy is below 70%. Suggestions:")
        print("    - Collect more samples per sign (aim for 100+)")
        print("    - Ensure hand is clearly visible during recording")
        print("    - Check for visually similar signs that might confuse the model")
    elif scores.mean() >= 0.90:
        print("  Excellent accuracy — model is ready for integration.")
    else:
        print("  Good accuracy — model should work well in production.")

    # ------------------------------------------------------------------ #
    # 4. Train on full dataset
    # ------------------------------------------------------------------ #
    print("\nTraining on full dataset ...")
    model.fit(X, y_enc)

    # ------------------------------------------------------------------ #
    # 5. Training-set report
    # ------------------------------------------------------------------ #
    from sklearn.metrics import classification_report  # noqa: PLC0415

    y_pred = model.predict(X)
    print("\nTraining-set classification report:")
    print(classification_report(y_enc, y_pred, target_names=le.classes_, zero_division=0))

    # ------------------------------------------------------------------ #
    # 6. Save model bundle
    # ------------------------------------------------------------------ #
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "label_encoder": le,
        "feature_count": 126,  # Two hands: 63 each
        "classes": list(le.classes_),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✓ Model saved to {MODEL_PATH}")
    print("  Restart the application — it will automatically use the trained model.")
    print(f"  Expected live accuracy: ~{scores.mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
