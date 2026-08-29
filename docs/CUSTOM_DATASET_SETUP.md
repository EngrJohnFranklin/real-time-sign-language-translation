# Ten-Word Static Sign Dataset Setup

This dataset is limited to these labels:

`good`, `hello`, `i_love_you`, `love`, `me`, `no`, `quiet`, `sorry`,
`thank_you`, and `yes`.

## Create the Dataset

```powershell
python scripts/collect_word_templates.py
```

The collector records a normalized 126-feature landmark sequence for each take
and stores it in `data/word_templates/<word>/`.

## Train

```powershell
python scripts/train_word_classifier.py
```

This creates `data/models/word_model.pkl`. The trainer keeps all frames from a
recording in one validation group, preventing frames from the same take from
appearing in both training and validation data.

## Boundaries

- This dataset is for the ten listed words only.
- The Speak Input screen displays images from `assets/sign_images/`; it does
  not use video playback.