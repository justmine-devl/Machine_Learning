from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.config import get_arg_parser, load_config
from experiments.common.dataset import build_class_list, load_classes, save_classes
from experiments.common.models import ModelConfig, build_model
from experiments.common.pseudo_labeling import generate_pseudo_label_csv
from experiments.common.spectrogram import SpectrogramConfig
from experiments.common.utils import get_device, load_checkpoint, seed_everything


def main() -> None:
    args = get_arg_parser("Generate pseudo-labels from unlabeled soundscapes").parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))

    if Path(cfg.get("class_file", "outputs/classes.txt")).exists():
        classes = load_classes(cfg.get("class_file", "outputs/classes.txt"))
    else:
        df = pd.read_csv(cfg["metadata_csv"])
        classes = build_class_list(df, cfg.get("primary_col", "primary_label"), cfg.get("secondary_col", "secondary_labels"))
        save_classes(classes, cfg.get("class_file", "outputs/classes.txt"))

    spec_cfg = SpectrogramConfig(
        sample_rate=cfg.get("sample_rate", 32000),
        n_fft=cfg.get("n_fft", 2048),
        hop_length=cfg.get("hop_length", 768),
        win_length=cfg.get("win_length", 2048),
        n_mels=cfg.get("n_mels", 192),
        f_min=cfg.get("f_min", 50),
        f_max=cfg.get("f_max", 15000),
    )

    model = build_model(cfg.get("model_type", "sed"), ModelConfig(
        backbone=cfg.get("backbone", "tf_efficientnetv2_s.in21k"),
        num_classes=len(classes),
        pretrained=cfg.get("pretrained", False),
        dropout=cfg.get("dropout", 0.2),
    )).to(device)
    load_checkpoint(model, cfg["teacher_checkpoint"], map_location=device)

    unlabeled_dir = Path(cfg["unlabeled_audio_dir"])
    files = sorted(list(unlabeled_dir.rglob("*.ogg")) + list(unlabeled_dir.rglob("*.wav")) + list(unlabeled_dir.rglob("*.mp3")))
    if not files:
        raise FileNotFoundError(f"No audio files found under {unlabeled_dir}")

    out = generate_pseudo_label_csv(
        model=model,
        soundscape_files=files,
        spec_cfg=spec_cfg,
        device=device,
        output_csv=cfg.get("pseudo_csv", "outputs/pseudo_labeling/pseudo_labels.csv"),
        class_names=classes,
        min_max_probability=cfg.get("min_max_probability", 0.5),
        class_probability_floor=cfg.get("class_probability_floor", 0.1),
        clip_duration=cfg.get("clip_duration", 5.0),
        batch_size=cfg.get("batch_size", 12),
    )
    print(f"Saved {len(out)} pseudo-labeled chunks to {cfg.get('pseudo_csv')}")


if __name__ == "__main__":
    main()
