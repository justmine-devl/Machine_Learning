# Experiments

This directory is the reproducibility layer above `src/bioacoustic/`. Reusable algorithms belong in `src/`; each experiment here supplies a concrete config, orchestration script, expected inputs, and output location.

| Experiment | Runner | Main output | Purpose |
|---|---|---|---|
| Baseline | `baseline/train_baseline.py` | `outputs/baseline/` | Clip-level EfficientNet reference |
| SED | `sed/train_sed.py` | `outputs/sed/` | Temporal feature map with class-wise attention |
| Pseudo-labeling | `pseudo_labeling/generate_pseudo_labels.py`, `train_with_pseudo_labels.py` | `outputs/pseudo_labeling/` | Select soft soundscape labels and mix them with labeled recordings |
| Noisy Student | `noisy_student/train_noisy_student.py` | `outputs/noisy_student/` | Generate pseudo labels, then run the mixed student pipeline |
| Self-distillation | `self_distillation/train_self_distillation.py` | `outputs/self_distillation/` | Train a student from hard labels and teacher probabilities |
| Ensemble | `ensemble/infer_ensemble.py`, `evaluate_ensemble.py` | `outputs/ensemble/` | Fold/checkpoint averaging, shifted windows, smoothing, evaluation |
| Analysis | `analysis/*.py` | `reports/tables/`, `reports/figures/` | Aggregate histories and produce report artifacts |

## Required local inputs

Update the selected experiment YAML before running it. The common assumptions are:

```text
data/train_metadata.csv
data/train_audio/<primary_label>/<filename>
data/unlabeled_soundscapes/*.{ogg,wav,mp3,flac}
data/test_soundscapes/*.{ogg,wav,mp3,flac}
weights/...                       # teacher or ensemble checkpoints
```

Pseudo-label CSV files include `row_id`, `audio_path`, `chunk_index`, and one soft-probability column per class. Those traceability fields are required so student training reloads the exact teacher window.

## Typical order

1. Prepare folds with `scripts/prepare_data.py` or point an experiment config at metadata that already contains `fold`.
2. Train baseline and SED folds.
3. Update the teacher checkpoint path in the pseudo/self-distillation configs.
4. Generate pseudo labels or teacher targets.
5. Train pseudo-label, Noisy Student, or self-distilled variants.
6. Update ensemble checkpoints, run inference/evaluation, then aggregate tables and plots.

Large generated outputs are intentionally ignored by Git. Commit only configs, small summary tables, figures needed by the report, and notes required to interpret a run.
