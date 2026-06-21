# Common modules

These modules are shared by all group experiments:

- `audio.py`: loading, padding, regular/shifted segmentation, temporal smoothing
- `spectrogram.py`: log-mel spectrogram generation
- `dataset.py`: metadata, class list, multi-hot labels, validation folds
- `models.py`: EfficientNet baseline and SED + attention pooling
- `losses.py`: BCE, Weighted Focal BCE, Soft-label BCE, student distillation loss
- `pseudo_labeling.py`: teacher prediction and pseudo-label selection
- `ensemble.py`: PyTorch model averaging and shifted-window blending
- `openvino_ensemble.py`: OpenVINO model ensemble inference
- `metrics.py`: AUC, mAP, precision, recall, F1, threshold search
- `training.py`: reusable training and validation loops
