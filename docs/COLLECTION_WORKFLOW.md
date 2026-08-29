# FSL Alphabet Collection Workflow

This workflow creates training data for the live FSL alphabet classifier. The
collection script gets its labels from `models.sign_detector.SignType`: `A`
through `Z`.

## Prepare

1. Review `FSL_REFERENCE.md`. The included alphabet handshape notes are
   provisional; verify labels against an authoritative FSL source before using
   them as ground truth.
2. Use a clear, well-lit recording area and keep each hand fully in frame.
3. Run the collection tool:

```powershell
python scripts/collect_training_data.py
```

## Controls

| Key | Action |
|---|---|
| `Space` | Capture one valid sample |
| `R` | Start continuous capture |
| `S` | Stop continuous capture |
| `N` | Select the next alphabet label |
| `P` | Select the previous alphabet label |
| `Esc` | Save and exit |

The tool writes normalized 126-feature rows to
`data/training_data/landmark_data.csv`. It rejects incomplete, low-confidence,
or near-duplicate samples.

## Collect

Collect samples for every intended label, varying hand side, position, rotation,
distance, lighting, and participants. Use at least 50 total samples to train;
collect substantially more and balance labels for reliable evaluation.

## Train and Evaluate

```powershell
python scripts/analyze_dataset.py
python scripts/train_xgboost_model.py
```

Training saves `data/models/sign_model.pkl`. The script reports class counts and
cross-validation results. Test the trained model in the desktop application:

```powershell
python src/main.py
```