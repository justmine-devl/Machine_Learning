# Bioacoustic Species Recognition

Unified research repository for a BirdCLEF 2025-style multi-label bioacoustic species-recognition project.

It provides a complete pipeline for:

- dataset inspection and stratified cross-validation;
- audio loading, resampling, padding, cropping, and fixed-window segmentation;
- normalized log-mel spectrogram generation;
- EfficientNet-family clip classification;
- sound event detection (SED) with class-wise temporal attention;
- class-balanced BCE and focal objectives;
- pseudo-labeling, Noisy Student, and self-distillation;
- PyTorch and OpenVINO ensemble inference;
- shifted-window blending, temporal smoothing, and probability adjustment;
- macro/micro AUC, tables, and plots.

## End-to-end pipeline

```text
raw audio
  -> mono resampling
  -> first/random 5-second training crop
  -> regular and 2.5-second-shifted inference windows
  -> normalized log-mel spectrogram
  -> EfficientNet classifier or SED + attention pooling
  -> BCE-family supervised or teacher-student training
  -> checkpoint/fold probability averaging
  -> regular/shifted-window blending
  -> temporal smoothing and optional power adjustment
  -> multi-label metrics and report artifacts
```

The default report-aligned acoustic configuration is:

| Parameter | Default |
|---|---:|
| Sample rate | 32,000 Hz |
| Clip duration | 5 seconds |
| Shifted-window offset | 2.5 seconds |
| FFT size | 2,048 |
| Hop length | 768 |
| Mel bins | 192 |
| Frequency range | 50-15,000 Hz |
| Target classes | 206 |

## Repository structure

```text
.
|-- README.md
|-- config.yaml
|-- requirements.txt
|-- pyproject.toml
|-- data/
|   `-- sample/
|-- src/
|   `-- bioacoustic/
|       |-- audio.py
|       |-- spectrogram.py
|       |-- dataset.py
|       |-- models.py
|       |-- losses.py
|       |-- training.py
|       |-- pseudo_labeling.py
|       |-- distillation.py
|       |-- ensemble.py
|       |-- openvino_ensemble.py
|       |-- metrics.py
|       |-- visualization.py
|       `-- utils.py
|-- scripts/
|   |-- prepare_data.py
|   |-- train_baseline.py
|   |-- train_sed.py
|   |-- generate_pseudo_labels.py
|   |-- train_student.py
|   |-- evaluate.py
|   |-- infer_ensemble.py
|   `-- make_report_plots.py
|-- experiments/
|   |-- baseline/
|   |-- sed/
|   |-- pseudo_labeling/
|   |-- noisy_student/
|   |-- self_distillation/
|   |-- ensemble/
|   `-- analysis/
|-- notebooks/
|-- reports/
|   |-- docs/
|   |-- tables/
|   `-- figures/
|-- tests/
`-- weights/
```

`src/bioacoustic/` contains reusable implementation. `scripts/` provides general command-line entry points using the root configuration. `experiments/` contains method-specific runners and configurations intended to reproduce individual report components.

The original five-member folders are retained locally in the Git-ignored `legacy_original_snapshot/` safety copy. It is not part of the tracked research package and should not be edited.

## Installation

Python 3.10 or newer is recommended. The project has also been smoke-tested with Python 3.13.

### Complete environment

The single root `requirements.txt` installs dependencies for training, tests, report generation, notebooks, and OpenVINO inference.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### Lightweight alternatives

Install only the core package dependencies:

```bash
python -m pip install -e .
```

Core package plus the test runner:

```bash
python -m pip install -e ".[dev]"
```

Core package plus OpenVINO:

```bash
python -m pip install -e ".[openvino]"
```

The alternatives use extras declared in `pyproject.toml`; they do not create additional requirements files. OpenVINO is only needed for `.xml`/`.bin` deployment-model inference.

## Dataset placement

The dataset is not included. The default root `config.yaml` expects:

```text
data/
|-- train_metadata.csv
|-- train_audio/
|   |-- <primary_label>/<filename>.ogg
|   `-- <filename>.ogg
|-- unlabeled_soundscapes/
|-- test_soundscapes/
`-- sample/
```

Both flat audio storage and `primary_label/filename` nesting are supported.

Required metadata columns:

| Column | Required | Description |
|---|---|---|
| `filename` | Yes | Audio filename relative to the configured audio directory |
| `primary_label` | Yes | Main species label |
| `secondary_labels` | No | Stringified list such as `['species_a', 'species_b']` |
| `fold` | No | Cross-validation fold; generated when absent |

Example:

```csv
filename,primary_label,secondary_labels
XC12345.ogg,species_a,"['species_b']"
XC67890.ogg,species_c,[]
```

Only small, license-compatible demonstration files should be placed in `data/sample/`. Raw audio, processed arrays, and generated datasets are ignored by Git.

## Configuration

The shared `config.yaml` controls general scripts. Important sections are:

- `data`: metadata, audio, soundscape, test, class-list, and column paths;
- `audio`: sample rate, clip duration, and shifted-window offset;
- `spectrogram`: FFT, hop, mel bins, frequency limits, normalization;
- `model`: model type, backbone, channels, pretrained weights, dropout;
- `training`: folds, batches, epochs, optimizer, loss, workers, device preference;
- `distillation`: hard-label/soft-label loss weighting;
- `pseudo_labeling`: confidence thresholds, batch size, and labeled ratio;
- `ensemble`: OpenVINO prefixes, blending, smoothing, and power adjustment;
- `evaluation`: threshold and threshold-search behavior.

Each experiment has its own `config_*.yaml`. For experiment runners, that method-specific YAML is authoritative. Review every local data path and checkpoint path before launching a long run.

## Recommended workflow

### 1. Prepare metadata and folds

```bash
python scripts/prepare_data.py --config config.yaml
```

Default outputs:

```text
outputs/processed/
|-- metadata_with_folds.csv
|-- classes.txt
`-- dataset_summary.csv
```

The class list determines checkpoint output order. Do not change its ordering between training, pseudo-label generation, evaluation, and inference.

### 2. Train the clip-level baseline

```bash
python scripts/train_baseline.py \
  --config config.yaml \
  --out-dir outputs/baseline
```

The baseline treats each log-mel spectrogram as a single image and produces independent sigmoid logits for all classes.

### 3. Train the SED model

```bash
python scripts/train_sed.py \
  --config config.yaml \
  --out-dir outputs/sed
```

The SED model retains the temporal CNN feature axis, pools frequency, and applies class-wise attention over time. Its output dictionary includes clip logits, frame logits, and attention weights.

### 4. Generate pseudo labels

```bash
python scripts/generate_pseudo_labels.py \
  --config config.yaml \
  --checkpoint outputs/sed/best.pt \
  --audio-dir data/unlabeled_soundscapes \
  --out outputs/pseudo_labeling/pseudo_labels.csv
```

The generated table contains:

- `row_id`;
- `audio_path`;
- `chunk_index`;
- one soft-probability column per class.

`audio_path` and `chunk_index` are required so student training can reload the exact teacher window instead of relying on an ambiguous filename.

### 5. Train a student from aligned soft targets

```bash
python scripts/train_student.py \
  --config config.yaml \
  --soft-targets outputs/self_distillation/teacher_soft_targets.npy \
  --out-dir outputs/student
```

The student objective combines hard-label Weighted Focal BCE with soft-label BCE:

```text
student_loss = hard_weight * hard_label_focal_bce
             + (1 - hard_weight) * soft_label_bce
```

Soft-target rows must remain aligned with the metadata rows used to construct the training split.

### 6. Evaluate saved predictions

```bash
python scripts/evaluate.py \
  --pred outputs/predictions.npy \
  --target outputs/targets.npy \
  --classes outputs/processed/classes.txt \
  --search-threshold \
  --out-dir outputs/evaluation
```

Both `.npy` matrices and numeric CSV matrices are supported. Prediction and target shapes must match `[samples, classes]`.

Evaluation outputs include aggregate metrics, per-class metrics, and an optional best global threshold selected using macro F1.

### 7. Run OpenVINO ensemble inference

```bash
python scripts/infer_ensemble.py \
  --config config.yaml \
  --models-dir weights/openvino \
  --audio-dir data/test_soundscapes \
  --classes outputs/processed/classes.txt \
  --out outputs/inference/predictions.csv
```

Expected OpenVINO naming:

```text
weights/openvino/
|-- model_0.xml
|-- model_0.bin
|-- model_1.xml
|-- model_1.bin
|-- model2_0.xml
`-- model2_0.bin
```

`model_` identifies regular-window models and `model2_` identifies shifted-window models. Prefixes are configurable.

### 8. Generate report plots

```bash
python scripts/make_report_plots.py \
  --history outputs/sed/history.csv \
  --out-dir reports/figures
```

## Reproducible method experiments

General scripts are convenient entry points. The following method-specific experiments provide explicit configurations and dedicated outputs.

### EfficientNet baseline

```bash
python experiments/baseline/train_baseline.py \
  --config experiments/baseline/config_baseline.yaml
```

Purpose: establish the clip-level reference before temporal SED modeling.

### Sound event detection

```bash
python experiments/sed/train_sed.py \
  --config experiments/sed/config_sed.yaml
```

Purpose: preserve temporal evidence and use class-wise attention pooling for sparse calls.

### Pseudo-labeling

```bash
python experiments/pseudo_labeling/generate_pseudo_labels.py \
  --config experiments/pseudo_labeling/config_pseudo.yaml

python experiments/pseudo_labeling/train_with_pseudo_labels.py \
  --config experiments/pseudo_labeling/config_pseudo.yaml
```

The second runner mixes labeled recordings with traceable soft pseudo-labeled windows according to `pseudo_labeling.labeled_ratio`.

### Noisy Student

```bash
python experiments/noisy_student/train_noisy_student.py \
  --config experiments/noisy_student/config_noisy_student.yaml
```

Use `--skip_pseudo_generation` when the configured pseudo-label table already exists:

```bash
python experiments/noisy_student/train_noisy_student.py \
  --config experiments/noisy_student/config_noisy_student.yaml \
  --skip_pseudo_generation
```

### Self-distillation

```bash
python experiments/self_distillation/train_self_distillation.py \
  --config experiments/self_distillation/config_self_distillation.yaml
```

To generate teacher probabilities without training the student:

```bash
python experiments/self_distillation/train_self_distillation.py \
  --config experiments/self_distillation/config_self_distillation.yaml \
  --generate_teacher_targets_only
```

### Ensemble inference and validation

```bash
python experiments/ensemble/infer_ensemble.py \
  --config experiments/ensemble/config_ensemble.yaml

python experiments/ensemble/evaluate_ensemble.py \
  --config experiments/ensemble/config_ensemble.yaml
```

The ensemble experiment supports PyTorch checkpoints and an optional OpenVINO branch. Post-processing includes checkpoint averaging, shifted-window blending, temporal smoothing, and power adjustment.

### Experiment analysis

```bash
python experiments/analysis/make_experiment_tables.py \
  --metrics_dir outputs \
  --out_dir reports/tables

python experiments/analysis/plot_curves.py \
  --history_csv outputs/sed/history.csv \
  --out_dir reports/figures
```

## Experiment summary

| Method | Main runner | Typical output | Role |
|---|---|---|---|
| Baseline | `experiments/baseline/train_baseline.py` | `outputs/baseline/` | Clip classifier reference |
| SED | `experiments/sed/train_sed.py` | `outputs/sed/` | Temporal attention model |
| Pseudo-labeling | `experiments/pseudo_labeling/` | `outputs/pseudo_labeling/` | Soundscape adaptation |
| Noisy Student | `experiments/noisy_student/` | `outputs/noisy_student/` | Mixed labeled/pseudo training |
| Self-distillation | `experiments/self_distillation/` | `outputs/self_distillation/` | Hard and soft supervision |
| Ensemble | `experiments/ensemble/` | `outputs/ensemble/` | Robust inference and validation |
| Analysis | `experiments/analysis/` | `reports/tables/`, `reports/figures/` | Report artifacts |

## Core module guide

| Module | Responsibility |
|---|---|
| `audio.py` | Loading, resampling, crop/pad, regular and shifted segmentation |
| `spectrogram.py` | Log-mel extraction and batched conversion |
| `dataset.py` | Metadata, labels, folds, supervised and pseudo-label datasets |
| `models.py` | Clip classifier, SED model, temporal attention, model factory |
| `losses.py` | BCE, weighted BCE, focal BCE, soft-label and distillation loss |
| `training.py` | Train/validation loops and checkpoint persistence |
| `pseudo_labeling.py` | Confidence selection, storage, and labeled/pseudo mixing |
| `distillation.py` | Teacher inference and soft-target persistence |
| `ensemble.py` | Averaging, shifted blending, smoothing, power adjustment |
| `openvino_ensemble.py` | OpenVINO discovery, compilation, and inference |
| `metrics.py` | AUC, mAP, precision, recall, F1, threshold and class metrics |
| `visualization.py` | Metric curves and report plots |
| `utils.py` | Seeding, configuration, device, directories, JSON output |

## Notebooks

The notebooks are report-oriented demonstrations that import reusable code from `src/bioacoustic/`:

1. `01_dataset_analysis.ipynb`
2. `02_baseline_experiment.ipynb`
3. `03_sed_experiment.ipynb`
4. `04_pseudo_labeling_experiment.ipynb`
5. `05_teacher_student_experiment.ipynb`
6. `06_ensemble_evaluation.ipynb`

Run notebooks from the repository root or the `notebooks/` directory. Update each notebook's configuration cell before executing it against local data.

## Reports

```text
reports/
|-- docs/       # code-to-report mapping
|-- tables/     # small CSV summaries
`-- figures/    # plots and pipeline diagrams
```

The code-to-report map is stored in `reports/docs/code_report_mapping.md`. Tables and figures should identify their source run. A manually entered or transcribed report value is not a reproduced metric unless its config, fold, checkpoint, and prediction artifacts are available.

## Checkpoints and generated artifacts

Store local PyTorch weights under `weights/` and OpenVINO models under `weights/openvino/`. Weights are excluded from Git because they are large generated artifacts and may have separate distribution terms.

When sharing a checkpoint, record:

- model type and backbone;
- fold and random seed;
- class-list checksum or exact class-list file;
- training configuration;
- validation metric and threshold;
- checkpoint checksum;
- download location and license.

Generated outputs such as checkpoints, predictions, caches, `.npy` arrays, experiment logs, and W&B runs are ignored by Git. Small report tables and figures may be committed when they are needed to support the academic report.

## Testing

Run the full automated suite:

```bash
python -m pytest -q
```

Compile-check all Python sources:

```bash
python -m compileall -q src scripts experiments tests
```

The latest synthetic smoke test covered:

- metadata preparation and folds;
- baseline and SED training;
- pseudo-label generation and mixed training;
- Noisy Student and self-distillation;
- PyTorch and OpenVINO ensemble inference;
- evaluation, tables, and plotting;
- all six notebook workflows.

Smoke tests validate that code paths execute and artifacts are produced. They do not validate scientific performance on the full BirdCLEF dataset.

## Reproducibility notes

- Keep the same class ordering across all stages.
- Split at recording level to avoid leakage between windows from one recording.
- Use deterministic validation crops; training may use first/random crops.
- Exclude validation soundscapes from pseudo-label training.
- Tune pseudo-label confidence and final decision thresholds only on validation data.
- Record fold, seed, config, package versions, and checkpoint hashes for every reported run.
- Do not treat the committed report tables as automatically reproduced results.

## Known limitations

- Datasets and trained weights are not distributed with the repository.
- Exact report reproduction requires the original folds and checkpoints.
- OpenVINO inference requires compatible model input/output shapes and class ordering.
- Rare classes may be absent from individual validation folds; macro AUC excludes invalid one-class targets.
- Ensemble inference increases runtime and storage requirements.
- Synthetic smoke-test metrics have no scientific meaning.
