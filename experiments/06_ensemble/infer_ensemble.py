from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.config import get_arg_parser, load_config
from experiments.common.dataset import load_classes
from experiments.common.openvino_ensemble import load_openvino_models, predict_audio_openvino_ensemble
from experiments.common.spectrogram import SpectrogramConfig
from experiments.common.utils import seed_everything


def main() -> None:
    args = get_arg_parser("Run OpenVINO ensemble inference on audio folder").parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    classes = load_classes(cfg["class_file"])
    output_dir = Path(cfg.get("output_dir", "outputs/ensemble")); output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(cfg["audio_dir"])
    files = sorted(list(audio_dir.rglob("*.ogg")) + list(audio_dir.rglob("*.wav")) + list(audio_dir.rglob("*.mp3")))

    spec_cfg = SpectrogramConfig(sample_rate=cfg.get("sample_rate", 32000), n_fft=cfg.get("n_fft", 2048), hop_length=cfg.get("hop_length", 768), win_length=cfg.get("win_length", 2048), n_mels=cfg.get("n_mels", 192), f_min=cfg.get("f_min", 50), f_max=cfg.get("f_max", 15000))
    regular_models = load_openvino_models(cfg["models_dir"], cfg.get("regular_prefix", "model_"), cfg.get("model_limit", None), cfg.get("openvino_device", "CPU"))
    shifted_models = load_openvino_models(cfg["models_dir"], cfg.get("shifted_prefix", "model2_"), cfg.get("model_limit", None), cfg.get("openvino_device", "CPU"))

    rows = []
    for audio_path in files:
        pred = predict_audio_openvino_ensemble(audio_path, regular_models, shifted_models, spec_cfg, cfg.get("clip_duration", 5.0), cfg.get("shift_seconds", 2.5), cfg.get("batch_size", 12), cfg.get("blend_alpha", 0.5), cfg.get("temporal_smoothing", True))
        for i, vec in enumerate(pred):
            row = {"filename": str(audio_path.relative_to(audio_dir)), "chunk_index": i, "end_sec": int((i + 1) * cfg.get("clip_duration", 5.0))}
            row.update({cls: float(vec[j]) for j, cls in enumerate(classes)})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "inference_predictions.csv", index=False)
    print(f"Saved predictions to {output_dir / 'inference_predictions.csv'}")


if __name__ == "__main__":
    main()
