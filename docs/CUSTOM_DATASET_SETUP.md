# Custom Dataset Setup

The project has two independent datasets. Do not combine their files or model
outputs.

## FSL Alphabet Dataset

Use `scripts/collect_training_data.py` to collect one normalized 126-feature
landmark vector per sample for `A` through `Z`. The output file is
`data/training_data/landmark_data.csv`; train it with:

```powershell
python scripts/train_xgboost_model.py
```

This produces `data/models/sign_model.pkl`, used by the live camera sign-input
view.

## Static Word-Template Dataset

The static-word template classes are:

`good`, `hello`, `i_love_you`, `love`, `me`, `no`, `quiet`, `sorry`,
`thank_you`, and `yes`.

Collect recordings with:

```powershell
python scripts/collect_word_templates.py
```

Templates are stored as `sample_*.npy` files under `data/word_templates/<word>/`.
Train the separate classifier with:

```powershell
python scripts/train_word_classifier.py
```

This produces `data/models/word_model.pkl` and does not modify the alphabet
model. The template trainer uses grouped validation so frames from one recording
do not appear in both training and validation data.

## Data Quality

Keep hands fully visible and vary participant, handedness, camera distance,
orientation, and lighting. Verify vocabulary labels against authoritative FSL
references before treating a custom dataset as linguistic ground truth.