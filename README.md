# Bioacoustic Species Recognition

Unified research repository for the team's BirdCLEF 2025 multi-label bioacoustic study. The current codebase covers dataset preparation, EfficientNet classification, sound event detection (SED), pseudo-labeling, Noisy Student, self-distillation, ensemble inference, evaluation, and report visualization.

```text
audio -> 5-second windows -> log-mel spectrogram
      -> EfficientNet classifier or SED + attention pooling
      -> BCE-family / teacher-student training
      -> regular + shifted ensemble -> temporal smoothing
      -> macro AUC, micro AUC, mAP, precision, recall, F1
```

## Structure

```text
.
|-- config.yaml                 # default configuration for scripts/
|-- src/bioacoustic/            # reusable library code
|-- scripts/                    # general command-line entry points
|-- experiments/                # method-specific configs and runnable experiments
|   |-- baseline/
|   |-- sed/
|   |-- pseudo_labeling/
|   |-- noisy_student/
|   |-- self_distillation/
|   |-- ensemble/
|   `-- analysis/
|-- notebooks/                  # visualization and demonstration notebooks
|-- reports/                    # figures, tables, and code-to-report mapping
|-- tests/                      # lightweight unit and integration tests
|-- data/                       # local dataset layout; large files are ignored
`-- weights/                    # local checkpoints; ignored by Git
```

The pre-refactor five-member folders are retained locally in the Git-ignored `legacy_original_snapshot/` safety copy.

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install OpenVINO only when running its inference path:

```bash
python -m pip install -e ".[openvino]"
```

## Data preparation

Follow [data/README.md](data/README.md), adjust `config.yaml`, then create folds and the class list:

```bash
python scripts/prepare_data.py --config config.yaml
```

## General scripts

```bash
python scripts/train_baseline.py --config config.yaml --out-dir outputs/baseline
python scripts/train_sed.py --config config.yaml --out-dir outputs/sed

python scripts/generate_pseudo_labels.py --config config.yaml \
  --checkpoint outputs/sed/best.pt

python scripts/train_student.py --config config.yaml \
  --soft-targets outputs/self_distillation/teacher_soft_targets.npy \
  --out-dir outputs/student

python scripts/evaluate.py --pred outputs/predictions.npy \
  --target outputs/targets.npy --classes outputs/processed/classes.txt \
  --search-threshold

python scripts/infer_ensemble.py --config config.yaml \
  --models-dir weights/openvino --audio-dir data/test_soundscapes
```

## Reproducible experiments

Unlike `scripts/`, each experiment owns a concrete configuration and output directory:

```bash
python experiments/baseline/train_baseline.py \
  --config experiments/baseline/config_baseline.yaml

python experiments/sed/train_sed.py \
  --config experiments/sed/config_sed.yaml

python experiments/pseudo_labeling/generate_pseudo_labels.py \
  --config experiments/pseudo_labeling/config_pseudo.yaml
python experiments/pseudo_labeling/train_with_pseudo_labels.py \
  --config experiments/pseudo_labeling/config_pseudo.yaml

python experiments/noisy_student/train_noisy_student.py \
  --config experiments/noisy_student/config_noisy_student.yaml

python experiments/self_distillation/train_self_distillation.py \
  --config experiments/self_distillation/config_self_distillation.yaml

python experiments/ensemble/infer_ensemble.py \
  --config experiments/ensemble/config_ensemble.yaml
python experiments/ensemble/evaluate_ensemble.py \
  --config experiments/ensemble/config_ensemble.yaml
```

See [experiments/README.md](experiments/README.md) for inputs, outputs, and the distinction between each method. Generated datasets, checkpoints, predictions, and logs live under `outputs/` and are not committed.

## Report support

- [reports/docs/code_report_mapping.md](reports/docs/code_report_mapping.md) maps report sections to current code and experiment runners.
- `reports/tables/` contains small report-ready CSV summaries.
- `reports/figures/` contains generated plots and pipeline diagrams.
- `scripts/make_report_plots.py` and `experiments/analysis/` regenerate analysis artifacts.

Reported CSV values are research records, not automatically reproduced metrics. Exact reproduction still requires the original dataset, fold metadata, and model checkpoints.
