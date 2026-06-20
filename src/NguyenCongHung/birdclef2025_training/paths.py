from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .config import TrainingConfig


@dataclass
class DataPaths:
    official_root: Optional[Path]
    extra_root: Optional[Path]
    train_audio: Optional[Path]
    train_soundscapes: Optional[Path]
    test_soundscapes: Optional[Path]
    train_csv: Optional[Path]
    taxonomy_csv: Optional[Path]
    sample_submission_csv: Optional[Path]
    recording_location_txt: Optional[Path]


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path and path.exists():
            return path
    return None


def find_by_name(roots: Sequence[Path], name: str, is_dir: bool = False) -> Optional[Path]:
    for root in roots:
        if not root or not root.exists():
            continue
        candidate = root / name
        if candidate.exists() and (candidate.is_dir() if is_dir else candidate.is_file()):
            return candidate
    for root in roots:
        if not root or not root.exists():
            continue
        for candidate in root.rglob(name):
            if candidate.exists() and (candidate.is_dir() if is_dir else candidate.is_file()):
                return candidate
    return None


def _path_or_none(value: Optional[str]) -> Optional[Path]:
    return Path(value) if value else None


def discover_paths(config: TrainingConfig) -> DataPaths:
    explicit_root = _path_or_none(config.data_root)
    candidate_roots = [explicit_root]
    if config.auto_discover_paths:
        candidate_roots.extend([
            Path("/kaggle/input/birdclef-2025"),
            _path_or_none(config.extra_data_root),
            Path.cwd(),
            Path("./input/birdclef-2025"),
            Path("./data/birdclef-2025"),
            Path("../input/birdclef-2025"),
        ])
    existing_roots = [p for p in candidate_roots if p is not None and p.exists()]
    official_candidates = [explicit_root]
    if config.auto_discover_paths:
        official_candidates.extend([
            Path("/kaggle/input/birdclef-2025"),
            Path("./input/birdclef-2025"),
            Path("./data/birdclef-2025"),
            Path("../input/birdclef-2025"),
            Path.cwd(),
        ])
    official_root = first_existing(official_candidates)
    explicit_extra_root = _path_or_none(config.extra_data_root)

    paths = DataPaths(
        official_root=official_root,
        extra_root=explicit_extra_root if explicit_extra_root and explicit_extra_root.exists() else None,
        train_audio=_path_or_none(config.train_audio_dir) or find_by_name(existing_roots, "train_audio", is_dir=True),
        train_soundscapes=_path_or_none(config.train_soundscapes_dir) or find_by_name(existing_roots, "train_soundscapes", is_dir=True),
        test_soundscapes=_path_or_none(config.test_soundscapes_dir) or find_by_name(existing_roots, "test_soundscapes", is_dir=True),
        train_csv=_path_or_none(config.train_csv) or find_by_name(existing_roots, "train.csv"),
        taxonomy_csv=_path_or_none(config.taxonomy_csv) or find_by_name(existing_roots, "taxonomy.csv"),
        sample_submission_csv=_path_or_none(config.sample_submission_csv) or find_by_name(existing_roots, "sample_submission.csv"),
        recording_location_txt=_path_or_none(config.recording_location_txt) or find_by_name(existing_roots, "recording_location.txt"),
    )
    print("Discovered paths:")
    for key, value in paths.__dict__.items():
        print(f"  {key}: {value}")
    if paths.extra_root is None:
        print("Warning: optional extra data root not found; continuing with official data only.")
    return paths


