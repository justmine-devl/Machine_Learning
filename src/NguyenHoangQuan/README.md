# Bioacoustic SED Ensemble

A compact research repo for **multi-label species recognition from audio** using a spectrogram-based Sound Event Detection (SED) pipeline.

This project is designed for experimentation and method deployment, not for competition submission. The core idea is:

```text
raw audio
→ 5-second windows + shifted windows
→ log-mel spectrograms
→ EfficientNet-based SED model
→ temporal attention pooling
→ sigmoid species probabilities
→ model ensemble
→ temporal smoothing
→ evaluation metrics / predictions
```

## Main Features

- Log-mel spectrogram preprocessing
- Regular and shifted-window audio segmentation
- EfficientNet-family SED model with temporal attention pooling
- BCE-based losses:
  - Binary Cross-Entropy
  - Weighted BCE
  - Weighted Focal BCE
  - Soft-label BCE for distillation
- OpenVINO ensemble inference
- Temporal smoothing across audio chunks
- Validation evaluation with AUC, mAP, Precision, Recall, and F1-score

## Repository Structure

```text
bioacoustic-sed-minimal/
├── README.md
├── requirements.txt
├── config.yaml
├── .gitignore
│
├── src/bioacoustic_sed/
│   ├── audio.py              # audio loading, windows, log-mel spectrograms
│   ├── model.py              # EfficientNet SED model + attention pooling
│   ├── losses.py             # BCE, Weighted BCE, Focal BCE, Soft-label BCE
│   ├── openvino_ensemble.py  # OpenVINO model loading + ensemble inference
│   ├── metrics.py            # AUC, mAP, Precision, Recall, F1
│   └── utils.py              # config, seed, path helpers
│
├── scripts/
│   ├── evaluate_ensemble.py  # evaluate ensemble on labeled validation audio
│   ├── infer_audio.py        # run ensemble inference on unlabeled audio
│   └── train_single.py       # compact single-model training script
│
├── data/
│   └── README.md             # where to place metadata/audio
│
├── weights/
│   └── README.md             # where to place .pt/.pth/.xml/.bin models
│
├── experiments/
│   ├── notebooks/
│   └── results/
│
└── tests/
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Format

For validation/evaluation, prepare a metadata CSV with at least:

```text
filename,primary_label
```

Example:

```csv
filename,primary_label
XC12345.ogg,sp1
XC67890.ogg,sp2
```

The script supports both flat audio folders and class-subfolder structure:

```text
data/audio/XC12345.ogg
```

or:

```text
data/audio/sp1/XC12345.ogg
```

## Evaluate OpenVINO Ensemble

Put OpenVINO models in:

```text
weights/openvino/
├── model_0.xml
├── model_0.bin
├── ...
├── model_12.xml
├── model2_0.xml
├── model2_0.bin
└── ...
```

Run:

```bash
python scripts/evaluate_ensemble.py \
  --config config.yaml \
  --metadata data/metadata.csv \
  --audio-dir data/audio \
  --models-dir weights/openvino \
  --out-dir experiments/results
```

Outputs:

```text
experiments/results/
├── metrics_summary.csv
├── predictions.csv
├── y_true.npy
├── y_pred.npy
└── threshold_curve.csv
```

## Inference on Unlabeled Audio

```bash
python scripts/infer_audio.py \
  --config config.yaml \
  --audio-dir data/audio \
  --models-dir weights/openvino \
  --class-list data/classes.txt \
  --out experiments/results/predictions.csv
```

## Train Single SED Model

```bash
python scripts/train_single.py \
  --config config.yaml \
  --metadata data/metadata.csv \
  --audio-dir data/audio \
  --out-dir experiments/results/checkpoints
```

## Loss Design

The method uses BCE-based objectives. For teacher training, the main loss is **Weighted Focal BCE with Logits**, which combines BCE, class weighting, and focal modulation. For student/self-distillation, the objective combines hard-label Weighted Focal BCE and soft-label BCE from teacher predictions.

```text
teacher loss = Weighted Focal BCE
student loss = alpha * hard-label Weighted Focal BCE
             + (1 - alpha) * soft-label BCE
```

## Notes

- Do not commit large audio files or model weights to GitHub.
- Store weights in `weights/` locally, or provide download links in `weights/README.md`.
- Store generated outputs in `experiments/results/`; this folder is ignored by Git.
