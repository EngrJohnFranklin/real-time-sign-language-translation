"""Train the separate static-word classifier from word template frames.

Only ``love``, ``i_love_you``, and ``stop`` are used. Validation groups all
frames from the same recording together to avoid sequence-frame leakage.
"""

import logging
import pathlib
import pickle
import sys

import numpy as np


SCRIPT_DIR = pathlib.Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from translation.word_template_loader import (  # noqa: E402
    EXCLUDED_TEMPLATE_FILES,
    STATIC_WORDS,
)


TEMPLATE_ROOT = PROJECT_ROOT / "data" / "word_templates"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "word_model.pkl"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_static_word_frames():
    """Return normalized frames, labels, and per-recording group identifiers."""
    features = []
    labels = []
    groups = []
    recording_counts = {}

    for word in sorted(STATIC_WORDS):
        word_dir = TEMPLATE_ROOT / word
        for sample_path in sorted(word_dir.glob("sample_*.npy")):
            if (word, sample_path.name) in EXCLUDED_TEMPLATE_FILES:
                logger.info("Skipping excluded template: %s", sample_path)
                continue

            sequence = np.load(sample_path)
            if sequence.ndim != 2 or sequence.shape[1] != 126:
                logger.warning("Skipping invalid template %s: %s", sample_path, sequence.shape)
                continue

            group = f"{word}/{sample_path.name}"
            features.extend(sequence)
            labels.extend([word] * len(sequence))
            groups.extend([group] * len(sequence))
            recording_counts[word] = recording_counts.get(word, 0) + 1

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(groups),
        recording_counts,
    )


def main() -> None:
    """Train and save the independent static-word model."""
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb

    features, labels, groups, recording_counts = load_static_word_frames()
    if not len(features):
        raise SystemExit("No valid static-word template frames were found.")

    classes, class_frame_counts = np.unique(labels, return_counts=True)
    minimum_recordings = min(recording_counts.get(word, 0) for word in STATIC_WORDS)
    if minimum_recordings < 2:
        raise SystemExit("Need at least two recordings per static word for validation.")

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    folds = min(5, minimum_recordings)
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    print("=== Static Word Classifier Training ===")
    print(f"Frames: {len(features)} | features: {features.shape[1]}")
    for word, frame_count in zip(classes, class_frame_counts):
        print(
            f"  {word}: {recording_counts.get(word, 0)} recordings, "
            f"{frame_count} frames"
        )

    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        features,
        encoded_labels,
        groups=groups,
        cv=splitter,
        scoring="accuracy",
    )
    print(f"{folds}-fold group-aware CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    print("Fold scores: " + ", ".join(f"{score:.3f}" for score in scores))

    model.fit(features, encoded_labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(
            {
                "model": model,
                "label_encoder": encoder,
                "feature_count": 126,
                "classes": list(encoder.classes_),
            },
            model_file,
        )

    print(f"Saved separate static-word model: {MODEL_PATH}")
    print("The letter model data/models/sign_model.pkl was not modified.")


if __name__ == "__main__":
    main()