# Ten-Word Sign Template Dataset

## Purpose

This dataset provides landmark templates for ten supported words:

`good`, `hello`, `i_love_you`, `love`, `me`, `no`, `quiet`, `sorry`,
`thank_you`, and `yes`.

## Data Shape

Each recording is a NumPy array with shape `(T, 126)`:

- `T`: number of accepted video frames in the recording
- first 63 values: normalized left-hand landmarks
- final 63 values: normalized right-hand landmarks

The collector requires at least 10 frames. Missing hands are represented by
zero-padded features.

## Commands

```powershell
python scripts/collect_word_templates.py
python scripts/train_word_classifier.py
```

The resulting word classifier is saved to `data/models/word_model.pkl`.

## Quality Checklist

- Record at least two separate takes for every word.
- Capture a variety of positions, distances, lighting conditions, and users.
- Keep labels consistent with the directory names above.
- Inspect the model's group-aware validation score before relying on it.

This dataset is limited to the ten labels listed above; it does not provide
unrestricted word or sentence translation.