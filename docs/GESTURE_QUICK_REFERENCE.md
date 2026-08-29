# FSL Alphabet Quick Reference

The camera classifier labels samples `A` through `Z`. Handshape descriptions in
this project are provisional and must be checked against an authoritative FSL
reference before data collection or user instruction.

## Before Collecting

- Read [FSL_REFERENCE.md](FSL_REFERENCE.md) for the current label table and
  verification status.
- Keep the full hand inside the frame and avoid motion blur.
- Collect each label under varied lighting, distance, rotation, handedness, and
  participant conditions.
- Use `scripts/collect_training_data.py` to ensure labels and feature formatting
  match runtime inference.

## Current Scope

- Alphabet labels: `A` through `Z`
- Feature vector: 126 normalized values, 63 for each hand
- Model file: `data/models/sign_model.pkl`
- Not currently supported: numbers, unrestricted words, or sentence-level sign
  translation

For the separate static-word set, see [CUSTOM_DATASET_SETUP.md](CUSTOM_DATASET_SETUP.md).