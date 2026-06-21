# Code-to-report mapping

| Report section | Core modules | Runnable support |
|---|---|---|
| Dataset and exploratory analysis | `dataset.py`, `visualization.py` | `scripts/prepare_data.py`, notebook 01, `experiments/analysis/` |
| Audio preprocessing | `audio.py`, `spectrogram.py` | notebooks 01-03 and all training runners |
| EfficientNet baseline | `models.EfficientNetClassifier` | `experiments/baseline/` |
| SED and attention pooling | `models.EfficientNetSED`, `AttentionPooling` | `experiments/sed/` |
| BCE losses and class balancing | `losses.py` | baseline/SED configs and runners |
| Pseudo-labeling | `pseudo_labeling.py`, `PseudoLabelAudioDataset` | `experiments/pseudo_labeling/` |
| Noisy Student | mixed labeled/pseudo sampler and soft pseudo targets | `experiments/noisy_student/` |
| Self-distillation | `distillation.py`, `student_distillation_loss` | `experiments/self_distillation/` |
| Ensemble and post-processing | `ensemble.py`, `openvino_ensemble.py` | `experiments/ensemble/` |
| Evaluation and results | `metrics.py`, `visualization.py` | `scripts/evaluate.py`, `experiments/analysis/`, notebooks 06 |

## Reproduction boundary

The code now exposes every report-level component, but the committed repository intentionally excludes audio, model weights, prediction arrays, and run logs. Tables and figures under `reports/` should therefore be treated as recorded project artifacts unless their producing run is also available locally.
