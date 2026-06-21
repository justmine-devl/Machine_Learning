"""Run the inference-compatible BirdCLEF 2025 multi-iterative Noisy Student pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml


def add_project_src_to_path() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        src = parent / "src"
        if src.exists():
            sys.path.insert(0, str(src))
            return parent
    raise RuntimeError("Could not locate the repository src directory")


PROJECT_ROOT = add_project_src_to_path()

from bioacoustic.training import BirdCLEFTrainingConfig


def parse_scalar(value: str) -> Any:
    """Parse CLI values using YAML scalar rules (bool, null, number, or string)."""
    return yaml.safe_load(value)


def load_training_config(path: str, overrides: list[str]) -> BirdCLEFTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw.pop("profile", None)
    valid = {item.name for item in fields(BirdCLEFTrainingConfig)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, received: {item!r}")
        key, value = item.split("=", 1)
        if key not in valid:
            raise ValueError(f"Unknown TrainingConfig field: {key}")
        raw[key] = parse_scalar(value)
    return BirdCLEFTrainingConfig(**raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BirdCLEF 2025 inference-compatible multi-iterative Noisy Student training"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any training configuration field; repeat this option as needed.",
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if args.debug is not None:
        overrides.append(f"debug={json.dumps(args.debug)}")

    config = load_training_config(args.config, overrides)
    # Keep CLI help and config validation usable without importing the full
    # torch/torchaudio/timm training stack.
    from bioacoustic.training import BirdCLEFTrainingPipeline

    print("Resolved BirdCLEF Noisy Student config:", flush=True)
    print(json.dumps(config.to_dict(), indent=2, default=str), flush=True)
    outputs = BirdCLEFTrainingPipeline(config).run()
    print("Pipeline outputs:", flush=True)
    print(json.dumps(outputs, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
