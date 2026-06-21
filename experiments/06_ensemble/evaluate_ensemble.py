from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.config import get_arg_parser, load_config
from experiments.common.dataset import add_stratified_folds, build_class_list, load_classes, make_label_vector
from experiments.common.ensemble import ensemble_predict_audio
from experiments.common.metrics import compute_all_metrics, per_class_metrics, search_best_threshold
from experiments.common.models import ModelConfig, build_model
from experiments.common.openvino_ensemble import load_openvino_models, predict_audio_openvino_ensemble
from experiments.common.spectrogram import SpectrogramConfig
from experiments.common.utils import get_device, load_checkpoint, save_json, seed_everything


def load_torch_models(cfg, num_classes, device):
    models = []
    for ckpt in cfg.get("checkpoints", []):
        model = build_model(cfg.get("model_type", "sed"), ModelConfig(
            backbone=cfg.get("backbone", "tf_efficientnetv2_s.in21k"),
            num_classes=num_classes,
            pretrained=cfg.get("pretrained", False),
            dropout=cfg.get("dropout", 0.2),
        )).to(device)
        load_checkpoint(model, ckpt, map_location=device)
        model.eval()
        models.append(model)
    if not models:
        raise ValueError("No PyTorch checkpoints provided. Fill `checkpoints` in config or set use_openvino=true.")
    return models


def main() -> None:
    args = get_arg_parser("Evaluate ensemble on validation fold").parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    output_dir = Path(cfg.get("output_dir", "outputs/ensemble")); output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.get("device", "auto"))

    df = pd.read_csv(cfg["metadata_csv"])
    if "fold" not in df.columns:
        df = add_stratified_folds(df, cfg.get("n_folds", 5), cfg.get("seed", 42), cfg.get("primary_col", "primary_label"))
    if Path(cfg.get("class_file", "outputs/classes.txt")).exists():
        classes = load_classes(cfg.get("class_file", "outputs/classes.txt"))
    else:
        classes = build_class_list(df, cfg.get("primary_col", "primary_label"), cfg.get("secondary_col", "secondary_labels"))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    valid_df = df[df["fold"] == cfg.get("fold", 0)].reset_index(drop=True)

    spec_cfg = SpectrogramConfig(sample_rate=cfg.get("sample_rate", 32000), n_fft=cfg.get("n_fft", 2048), hop_length=cfg.get("hop_length", 768), win_length=cfg.get("win_length", 2048), n_mels=cfg.get("n_mels", 192), f_min=cfg.get("f_min", 50), f_max=cfg.get("f_max", 15000))

    if cfg.get("use_openvino", False):
        regular_models = load_openvino_models(cfg["models_dir"], prefix=cfg.get("regular_prefix", "model_"), limit=cfg.get("model_limit", None), device=cfg.get("openvino_device", "CPU"))
        shifted_models = load_openvino_models(cfg["models_dir"], prefix=cfg.get("shifted_prefix", "model2_"), limit=cfg.get("model_limit", None), device=cfg.get("openvino_device", "CPU"))
        torch_models = None
    else:
        torch_models = load_torch_models(cfg, len(classes), device)
        regular_models = shifted_models = None

    y_true_rows = []
    y_pred_rows = []
    row_meta = []
    for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="evaluate ensemble"):
        audio_path = Path(cfg["audio_dir"]) / str(row[cfg.get("filename_col", "filename")])
        if cfg.get("use_openvino", False):
            chunk_pred = predict_audio_openvino_ensemble(
                audio_path=audio_path,
                regular_models=regular_models,
                shifted_models=shifted_models,
                spec_cfg=spec_cfg,
                clip_duration=cfg.get("clip_duration", 5.0),
                shift_seconds=cfg.get("shift_seconds", 2.5),
                batch_size=cfg.get("batch_size", 12),
                blend_alpha=cfg.get("blend_alpha", 0.5),
                smooth=cfg.get("temporal_smoothing", True),
            )
        else:
            chunk_pred = ensemble_predict_audio(
                models_regular=torch_models,
                audio_path=audio_path,
                spec_cfg=spec_cfg,
                device=device,
                models_shifted=None,
                clip_duration=cfg.get("clip_duration", 5.0),
                batch_size=cfg.get("batch_size", 12),
                blend_alpha=cfg.get("blend_alpha", 0.5),
                smooth=cfg.get("temporal_smoothing", True),
            )
        # Recording-level weak label is repeated for all chunks. This is acceptable for internal validation analysis.
        target = make_label_vector(row, class_to_idx, cfg.get("primary_col", "primary_label"), cfg.get("secondary_col", "secondary_labels"))
        for chunk_idx, pred in enumerate(chunk_pred):
            y_true_rows.append(target)
            y_pred_rows.append(pred)
            row_meta.append({"filename": str(audio_path), "chunk_index": chunk_idx})

    y_true = np.stack(y_true_rows)
    y_pred = np.stack(y_pred_rows)
    best_t, best_f1 = search_best_threshold(y_true, y_pred)
    metrics = compute_all_metrics(y_true, y_pred, threshold=best_t)
    metrics["best_macro_f1"] = best_f1
    save_json(metrics, output_dir / "metrics_summary.json")
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(per_class_metrics(y_true, y_pred, classes, threshold=best_t)).to_csv(output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(row_meta).to_csv(output_dir / "prediction_rows.csv", index=False)
    np.save(output_dir / "y_true.npy", y_true)
    np.save(output_dir / "y_pred.npy", y_pred)
    print(metrics)


if __name__ == "__main__":
    main()
