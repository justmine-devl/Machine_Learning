# Bioacoustic Species Recognition

Unified research code for the team's BirdCLEF 2025 bioacoustic species-recognition study. This is an academic experiment repository—not a competition submission. It preserves all five members' original work while exposing the report's common pipeline as an installable Python package and runnable scripts.

```text
audio -> 5 s regular/shifted windows -> normalized log-mel spectrogram
      -> EfficientNet baseline or SED + attention pooling
      -> BCE-family training / teacher-student learning
      -> model averaging -> shifted-window blending -> temporal smoothing
      -> macro AUC, mAP, precision, recall, and F1
```

## Methods represented

- Dataset analysis, multi-label metadata, recording-level splits, and rare-class balancing
- 32 kHz audio loading, silence padding, 5-second crops, and 2.5-second shifted inference
- 192-bin log-mel spectrograms (`n_fft=2048`, `hop_length=768`, 50–15,000 Hz)
- EfficientNet-family clip baseline and SED model with class-wise temporal attention
- BCE, weighted BCE, focal BCE, weighted focal BCE, soft-label BCE, and student loss
- Pseudo-label confidence filtering, Noisy Student, and self-distillation utilities
- Checkpoint prediction averaging, boundary-aware blending, temporal smoothing, and power adjustment

## Repository layout

```text
.
├── config.yaml                  # paths and shared experiment hyperparameters
├── src/bioacoustic/             # reusable implementation
├── scripts/                     # data, training, pseudo-label, evaluation, inference CLIs
├── notebooks/                   # lightweight report demonstrations importing src/
├── experiments/                 # method notes and experiment-specific artifacts
├── reports/                     # report mapping, figures, tables, and audit
├── data/                        # local dataset placement (large files ignored)
├── weights/                     # local checkpoints (ignored)
├── legacy/                      # untouched original work from all five members
└── tests/                       # lightweight unit tests
```

The separately requested `legacy_original_snapshot/` is a local, Git-ignored safety copy of the five folders as they existed before the refactor. The canonical preserved originals are tracked under `legacy/`.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows PowerShell
python -m pip install -r requirements.txt
python -m pip install -e .
```

OpenVINO is optional and only needed to revisit the original deployment workflow:

```bash
python -m pip install -e ".[openvino]"
```

## Data preparation

Place `train.csv` and audio as described in [data/README.md](data/README.md), update `config.yaml`, then run:

```bash
python scripts/prepare_data.py --metadata data/metadata.csv
```

The metadata needs `filename` (or `filepath`) and `primary_label`; `secondary_labels` is optional and accepts a stringified list.

## Training

```bash
python scripts/train_baseline.py --config config.yaml
python scripts/train_sed.py --config config.yaml
python scripts/train_student.py --spectrograms data/processed/specs.npy \
  --hard-targets data/processed/hard.npy --soft-targets data/processed/soft.npy
```

Set `model.pretrained: false` for offline smoke tests. Large checkpoints and outputs are ignored by Git.

## Pseudo-labeling, evaluation, and ensemble inference

```bash
python scripts/generate_pseudo_labels.py --checkpoint weights/teacher.pt \
  --audio-dir data/unlabeled_audio --out experiments/pseudo_labeling/soft_labels.csv

python scripts/evaluate.py --targets y_true.npy --predictions y_pred.npy --search-threshold

python scripts/infer_ensemble.py fold0.npy fold1.npy fold2.npy \
  --out experiments/ensemble/predictions.npy
```

The ensemble CLI consumes model-output matrices, whether produced by PyTorch or the preserved OpenVINO workflow. It deliberately keeps model execution separate from post-processing so results are reproducible and easy to audit.

## Reproducibility and report support

- All shared hyperparameters and paths live in `config.yaml`; no unified module contains a Kaggle or local-machine path.
- [reports/experiment_notes.md](reports/experiment_notes.md) maps report sections to code and notebooks.
- [legacy/README.md](legacy/README.md) maps each member's originals to the unified implementation.
- [reports/repository_audit.md](reports/repository_audit.md) records overlaps, hard-coded paths, dependencies, and preserved artifacts found during migration.

Datasets, audio, model weights, predictions, caches, Kaggle outputs, and experiment logs are intentionally not committed. Keep only small, explicitly documented demonstration files in `data/sample/`.
