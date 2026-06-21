"""OpenVINO ensemble inference utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .audio import load_audio, segment_for_inference
from .ensemble import average_predictions, blend_regular_shifted, temporal_smoothing
from .spectrogram import batch_log_mel

try:
    import openvino as ov
except Exception:  
    ov = None


class OpenVINOEnsemble:
    """Wrapper around multiple OpenVINO classification models."""

    def __init__(self, model_paths: Sequence[str | Path], device: str = "CPU") -> None:
        if ov is None:
            raise ImportError("openvino is required for OpenVINOEnsemble.")
        self.core = ov.Core()
        self.models = []
        for path in model_paths:
            model = self.core.read_model(model=str(path))
            self.models.append(self.core.compile_model(model, device))

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        """Run all models on a batch and average sigmoid probabilities."""
        preds = []
        for compiled in self.models:
            output = compiled(x)[compiled.output(0)]
            probs = 1.0 / (1.0 + np.exp(-output))
            preds.append(probs.astype(np.float32))
        return average_predictions(preds)


def find_openvino_models(models_dir: str | Path, prefix: str = "model") -> List[Path]:
    models_dir = Path(models_dir)
    return sorted(models_dir.glob(f"{prefix}*.xml"))


def infer_audio_file_openvino(
    audio_path: str | Path,
    regular_models: OpenVINOEnsemble,
    shifted_models: Optional[OpenVINOEnsemble] = None,
    sample_rate: int = 32000,
    duration: float = 5.0,
    shift: float = 2.5,
    blend_alpha: float = 0.5,
    smooth: bool = True,
    spectrogram_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Run OpenVINO ensemble on one audio file and return chunk probabilities."""
    spectrogram_kwargs = spectrogram_kwargs or {}
    wav = load_audio(audio_path, sample_rate=sample_rate)
    regular_windows, shifted_windows = segment_for_inference(wav, sample_rate, duration, shift)
    regular_x = batch_log_mel(regular_windows, sample_rate=sample_rate, **spectrogram_kwargs)
    regular_pred = regular_models.predict_batch(regular_x)

    if shifted_models is not None and shifted_windows:
        shifted_x = batch_log_mel(shifted_windows, sample_rate=sample_rate, **spectrogram_kwargs)
        shifted_pred = shifted_models.predict_batch(shifted_x)
        pred = blend_regular_shifted(regular_pred, shifted_pred, alpha=blend_alpha)
    else:
        pred = regular_pred
    if smooth:
        pred = temporal_smoothing(pred)
    return pred.astype(np.float32)
