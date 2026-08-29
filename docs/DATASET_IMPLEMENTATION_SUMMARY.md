# Ten-Word Dataset Implementation Summary

## Supported Labels

The static-word pipeline supports ten labels:

`good`, `hello`, `i_love_you`, `love`, `me`, `no`, `quiet`, `sorry`,
`thank_you`, and `yes`.

## Pipeline

```text
collect_word_templates.py
  -> data/word_templates/<word>/sample_XX.npy
  -> train_word_classifier.py
  -> data/models/word_model.pkl
```

Each `sample_XX.npy` file contains a `(T, 126)` normalized landmark sequence.
The trainer uses `StratifiedGroupKFold` so all frames from one recording remain
in the same training or validation group.

## Related Runtime Behavior

The live Speak Input interface recognizes its constrained Vosk vocabulary and
shows a matching static PNG from `assets/sign_images/`. It does not play sign
videos.