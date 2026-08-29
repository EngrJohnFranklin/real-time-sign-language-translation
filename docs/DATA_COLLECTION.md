# Ten-Word Sign Data Collection Guide

## Scope

This workflow collects static sign templates for exactly these ten words:

`good`, `hello`, `i_love_you`, `love`, `me`, `no`, `quiet`, `sorry`,
`thank_you`, and `yes`.

It uses `scripts/collect_word_templates.py` and saves template recordings under
`data/word_templates/`.

## Start Recording

From the project root:

```powershell
python scripts/collect_word_templates.py
```

To record one word first, use its file-system label:

```powershell
python scripts/collect_word_templates.py --word thank_you
```

Use `--dry-run` to test the camera and controls without writing files.

## Controls

| Key | Action |
|---|---|
| `Space` | Start or stop the current recording |
| `Enter` | Save the stopped recording and select the next word |
| `N` | Skip to the next word and discard the unsaved recording |
| `P` | Return to the previous word and discard the unsaved recording |
| `Esc` | Stop recording, then press `Esc` again to exit |

The recorder needs at least 10 valid frames before it saves a recording.

## Output Format

Each saved sample is a NumPy file at:

```text
data/word_templates/<word>/sample_XX.npy
```

Each file is a sequence shaped `(T, 126)`, where `T` is the number of captured
frames. Every frame has 63 normalized landmark values for the left hand and 63
for the right hand. A `metadata.json` file is appended in the same word folder.

## Collection Quality

- Keep the full hand or hands inside the camera frame.
- Use even lighting and a clear background.
- Record multiple takes per word with varied position, distance, orientation,
  handedness, and participants.
- Keep each target sign steady while recording; this is a static-word dataset.
- Check each label against an authoritative FSL source before treating the
  samples as linguistic reference data.

## Train the Word Classifier

After collecting at least two recordings for every word:

```powershell
python scripts/train_word_classifier.py
```

The trainer uses group-aware cross-validation and saves
`data/models/word_model.pkl`.