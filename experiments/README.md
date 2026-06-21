# Experiments

This folder contains a unified experimental codebase for the group BirdCLEF-style bioacoustic species recognition project.
It is designed to replace disconnected member-specific folders with a consistent set of reproducible experiments.

## Folder layout

```text
experiments/
├── common/                 # shared reusable code
├── 01_baseline/            # spectrogram-based EfficientNet baseline
├── 02_sed/                 # SED model + temporal attention pooling
├── 03_pseudo_labeling/     # pseudo-label generation and pseudo-label training
├── 04_noisy_student/       # teacher-student training with stronger noise
├── 05_self_distillation/   # hard-label + soft-label distillation
├── 06_ensemble/            # multi-model / OpenVINO-style ensemble evaluation
└── 07_analysis/            # plots and tables for report
```

## Data assumptions

The code expects a metadata CSV with at least:

```text
filename,primary_label
```

Optional columns:

```text
secondary_labels,fold
```

Audio files should be placed under an audio directory and referenced by `filename`.
For BirdCLEF-style data, `filename` may include subfolders such as `species_id/audio.ogg`.

## Typical commands

Baseline:

```bash
python experiments/01_baseline/train_baseline.py \
  --config experiments/01_baseline/config_baseline.yaml
```

SED:

```bash
python experiments/02_sed/train_sed.py \
  --config experiments/02_sed/config_sed.yaml
```

Pseudo-label generation:

```bash
python experiments/03_pseudo_labeling/generate_pseudo_labels.py \
  --config experiments/03_pseudo_labeling/config_pseudo.yaml
```

Train with pseudo-labels:

```bash
python experiments/03_pseudo_labeling/train_with_pseudo_labels.py \
  --config experiments/03_pseudo_labeling/config_pseudo.yaml
```

Noisy Student:

```bash
python experiments/04_noisy_student/train_noisy_student.py \
  --config experiments/04_noisy_student/config_noisy_student.yaml
```

Self-distillation:

```bash
python experiments/05_self_distillation/train_self_distillation.py \
  --config experiments/05_self_distillation/config_self_distillation.yaml
```

Ensemble evaluation:

```bash
python experiments/06_ensemble/evaluate_ensemble.py \
  --config experiments/06_ensemble/config_ensemble.yaml
```

Report plots:

```bash
python experiments/07_analysis/plot_curves.py \
  --metrics-csv outputs/metrics/history.csv \
  --out-dir reports/figures
```

