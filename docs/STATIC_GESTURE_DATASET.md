# Static Word-Template Dataset

This guide covers the static-word template workflow. It is separate from the
FSL alphabet CSV classifier.

## Supported Template Labels

`good`, `hello`, `i_love_you`, `love`, `me`, `no`, `quiet`, `sorry`,
`thank_you`, and `yes`.

## Collect Templates

```powershell
python scripts/collect_word_templates.py
```

Each accepted recording is stored under `data/word_templates/<word>/` as a
126-feature landmark sequence. Review samples for clear framing and label
accuracy; the loader excludes known duration outliers while retaining them on
disk for auditability.

## Train the Word Model

```powershell
python scripts/train_word_classifier.py
```

The trainer validates with `StratifiedGroupKFold`, keeping frames from each
recording in a single validation group. It writes `data/models/word_model.pkl`.
This command does not overwrite `data/models/sign_model.pkl`.

## Scope

This model recognizes only its trained static-word labels. It is not the source
of the speech-view sign image: that view maps recognized Vosk vocabulary to
images stored under `assets/sign_images/`.