# Dataset Implementation Summary

## Current Data Pipelines

| Pipeline | Labels | Source data | Model output | Runtime use |
|---|---|---|---|---|
| FSL alphabet | `A` through `Z` | `data/training_data/landmark_data.csv` | `data/models/sign_model.pkl` | Live camera letter recognition |
| Static words | `good`, `hello`, `i_love_you`, `love`, `me`, `no`, `quiet`, `sorry`, `thank_you`, `yes` | `data/word_templates/` | `data/models/word_model.pkl` | Optional word recognition |

Both pipelines use normalized dual-hand landmark features: 63 values per hand,
for 126 values per frame. Missing hands are zero-padded.

## Supporting Tools

- `scripts/collect_training_data.py`: records validated alphabet samples to CSV.
- `scripts/analyze_dataset.py`: summarizes the alphabet CSV.
- `scripts/train_xgboost_model.py`: cross-validates and writes the alphabet
  model.
- `scripts/collect_word_templates.py`: records static-word landmark sequences.
- `scripts/train_word_classifier.py`: uses group-aware validation and writes the
  static-word model.

The speech UI is separate from these model pipelines. It uses a constrained Vosk
grammar and displays matching static images from `assets/sign_images/`.