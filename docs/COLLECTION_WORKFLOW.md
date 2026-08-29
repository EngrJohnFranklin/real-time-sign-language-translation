# Ten-Word Collection Workflow

## Vocabulary


1. `love`
2. `i_love_you`
3. `thank_you`
4. `quiet`
5. `no`
6. `hello`
7. `sorry`
8. `yes`
9. `me`
10. `good`

## Workflow

1. Review the label against an authoritative FSL reference.
2. Run `python scripts/collect_word_templates.py`.
3. Select the displayed word or pass `--word <label>`.
4. Press `Space` to start recording while holding the sign clearly in frame.
5. Press `Space` again to stop the take.
6. Press `Enter` to save the take and advance, or `N`/`P` to discard it and
   change words.
7. Repeat for multiple recordings of each of the ten words.
8. Train with `python scripts/train_word_classifier.py`.

The collector saves only recordings with at least 10 valid landmark frames.
The trainer requires at least two recordings per word so it can perform
group-aware validation.

## Files Created

```text
data/word_templates/<word>/sample_XX.npy
data/word_templates/<word>/metadata.json
data/models/word_model.pkl
```

See [DATA_COLLECTION.md](DATA_COLLECTION.md) for controls and sample format.