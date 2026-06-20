from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    from openvino.runtime import Core
except ImportError:
    Core = None


def discover_openvino_models(models_dir: str | Path, prefix: str) -> List[Path]:
    """Find OpenVINO XML model files matching a prefix, sorted by numeric suffix when possible."""
    models_dir = Path(models_dir)
    paths = list(models_dir.glob(f"{prefix}*.xml"))

    def sort_key(path: Path):
        stem = path.stem.replace(prefix, "")
        try:
            return int(stem)
        except ValueError:
            return stem

    return sorted(paths, key=sort_key)


def load_openvino_models(
    models_dir: str | Path,
    prefix: str,
    device: str = "CPU",
    model_limit: int | None = None,
):
    """Load and compile OpenVINO models from XML files."""
    if Core is None:
        raise ImportError("openvino is required. Install with `pip install openvino`.")

    model_paths = discover_openvino_models(models_dir, prefix)
    if model_limit is not None:
        model_paths = model_paths[:model_limit]
    if not model_paths:
        raise FileNotFoundError(f"No OpenVINO models found in {models_dir} with prefix '{prefix}'")

    core = Core()
    compiled = []
    for path in model_paths:
        model = core.read_model(model=str(path))
        compiled_model = core.compile_model(model, device)
        compiled.append(compiled_model)
    return compiled


def _run_compiled_model(compiled_model, batch: np.ndarray) -> np.ndarray:
    """Run one compiled OpenVINO model and return logits."""
    batch = batch.astype(np.float32)
    outputs = compiled_model([batch])
    output_key = compiled_model.output(0)
    return np.asarray(outputs[output_key])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def predict_model_group(compiled_models: Sequence, specs: np.ndarray) -> np.ndarray:
    """
    Average sigmoid probabilities from a group of OpenVINO models.

    Args:
        compiled_models: list of OpenVINO compiled models
        specs: [N, 1, n_mels, time]

    Returns:
        probs: [N, num_classes]
    """
    preds = []
    for model in compiled_models:
        logits = _run_compiled_model(model, specs)
        preds.append(sigmoid(logits))
    return np.mean(np.stack(preds, axis=0), axis=0)


def blend_regular_shifted(
    regular: np.ndarray,
    shifted: np.ndarray,
    regular_weight: float = 0.50,
    shifted_left_weight: float = 0.25,
    shifted_right_weight: float = 0.25,
) -> np.ndarray:
    """
    Blend regular-window predictions with neighboring shifted-window predictions.

    For middle chunks:
        final[t] = 0.50 * regular[t]
                 + 0.25 * shifted[t-1]
                 + 0.25 * shifted[t]

    Boundary chunks use the nearest shifted prediction.
    """
    if shifted is None or len(shifted) == 0:
        return regular.copy()

    output = regular.copy()
    n_regular = len(regular)
    n_shifted = len(shifted)

    if n_regular == 1:
        output[0] = 0.75 * regular[0] + 0.25 * shifted[0]
        return output

    output[0] = (regular_weight + shifted_left_weight) * regular[0] + shifted_right_weight * shifted[0]
    output[-1] = (regular_weight + shifted_right_weight) * regular[-1] + shifted_left_weight * shifted[-1]

    for i in range(1, n_regular - 1):
        left_idx = min(i - 1, n_shifted - 1)
        right_idx = min(i, n_shifted - 1)
        output[i] = (
            regular_weight * regular[i]
            + shifted_left_weight * shifted[left_idx]
            + shifted_right_weight * shifted[right_idx]
        )
    return output


def temporal_smoothing(preds: np.ndarray, weight: float = 0.10) -> np.ndarray:
    """
    Smooth predictions across adjacent chunks.

    middle: 0.1 previous + 0.8 current + 0.1 next
    edge:   0.9 current + 0.1 neighbor
    """
    if len(preds) <= 1 or weight <= 0:
        return preds.copy()

    out = preds.copy()
    out[0] = (1.0 - weight) * preds[0] + weight * preds[1]
    out[-1] = (1.0 - weight) * preds[-1] + weight * preds[-2]

    center_weight = 1.0 - 2.0 * weight
    for i in range(1, len(preds) - 1):
        out[i] = weight * preds[i - 1] + center_weight * preds[i] + weight * preds[i + 1]
    return out


def aggregate_chunks(preds: np.ndarray, method: str = "max") -> np.ndarray:
    """Aggregate chunk-level probabilities into one audio-level prediction."""
    if method == "max":
        return preds.max(axis=0)
    if method == "mean":
        return preds.mean(axis=0)
    if method == "median":
        return np.median(preds, axis=0)
    raise ValueError(f"Unknown aggregate method: {method}")


def predict_audio_ensemble(
    regular_models: Sequence,
    shifted_models: Sequence,
    regular_specs: np.ndarray,
    shifted_specs: np.ndarray,
    regular_weight: float = 0.50,
    shifted_left_weight: float = 0.25,
    shifted_right_weight: float = 0.25,
    smoothing_weight: float = 0.10,
) -> np.ndarray:
    """Run full ensemble prediction for one audio file."""
    regular_pred = predict_model_group(regular_models, regular_specs)
    shifted_pred = predict_model_group(shifted_models, shifted_specs) if shifted_models else None
    blended = blend_regular_shifted(
        regular_pred,
        shifted_pred,
        regular_weight=regular_weight,
        shifted_left_weight=shifted_left_weight,
        shifted_right_weight=shifted_right_weight,
    )
    return temporal_smoothing(blended, weight=smoothing_weight)
