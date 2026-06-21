"""Metadata handling and PyTorch datasets."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import (
    augment_wave,
    crop_first_or_random,
    crop_or_pad_wave,
    load_audio,
    mixup_focal_pseudo,
    pad_or_trim,
)
from .spectrogram import log_mel_spectrogram

try:
    import librosa
except Exception:  
    librosa = None

if TYPE_CHECKING:
    from .training import BirdCLEFTrainingConfig


def read_metadata(path: str | Path) -> pd.DataFrame:
    """Read metadata CSV."""
    return pd.read_csv(path)


def parse_secondary_labels(value) -> List[str]:
    """Parse BirdCLEF-style secondary_labels field safely."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            return [] if value.strip() in {"", "[]"} else [value]
    return []


def build_class_list(df: pd.DataFrame, primary_col: str = "primary_label") -> List[str]:
    """Build sorted class list from the primary label column."""
    return sorted(df[primary_col].dropna().astype(str).unique().tolist())


def make_label_map(classes: Sequence[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(classes)}


def encode_multihot(
    primary_label: str,
    secondary_labels=None,
    label_map: Optional[Dict[str, int]] = None,
    num_classes: Optional[int] = None,
    include_secondary: bool = True,
) -> np.ndarray:
    """Encode primary and optional secondary labels into a multi-hot vector."""
    if label_map is None and num_classes is None:
        raise ValueError("Either label_map or num_classes must be provided.")
    if num_classes is None:
        num_classes = len(label_map)  # type: ignore[arg-type]
    target = np.zeros(num_classes, dtype=np.float32)
    if label_map and str(primary_label) in label_map:
        target[label_map[str(primary_label)]] = 1.0
    if include_secondary and label_map is not None:
        for lab in parse_secondary_labels(secondary_labels):
            if lab in label_map:
                target[label_map[lab]] = 1.0
    return target


def add_stratified_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_col: str = "primary_label",
    seed: int = 42,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Assign stratified folds by primary label."""
    from sklearn.model_selection import StratifiedKFold

    out = df.copy().reset_index(drop=True)
    out[fold_col] = -1
    y = out[label_col].astype(str).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(out, y)):
        out.loc[val_idx, fold_col] = fold
    return out


class BirdAudioDataset(Dataset):
    """Simple audio-to-spectrogram dataset for supervised training."""

    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str | Path,
        classes: Sequence[str],
        filename_col: str = "filename",
        primary_col: str = "primary_label",
        secondary_col: str = "secondary_labels",
        sample_rate: int = 32000,
        duration: float = 5.0,
        train: bool = True,
        include_secondary: bool = True,
        spectrogram_kwargs: Optional[dict] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for BirdAudioDataset.")
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.classes = list(classes)
        self.label_map = make_label_map(classes)
        self.filename_col = filename_col
        self.primary_col = primary_col
        self.secondary_col = secondary_col
        self.sample_rate = sample_rate
        self.duration = duration
        self.train = train
        self.include_secondary = include_secondary
        self.spectrogram_kwargs = spectrogram_kwargs or {}

    def __len__(self) -> int:
        return len(self.df)

    def _audio_path(self, filename: str) -> Path:
        path = self.audio_dir / filename
        if path.exists():
            return path
        # BirdCLEF train_audio may be organized by label/filename.
        primary = str(
            self.df.loc[self.df[self.filename_col] == filename, self.primary_col].iloc[
                0
            ]
        )
        nested = self.audio_dir / primary / filename
        return nested if nested.exists() else path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._audio_path(str(row[self.filename_col]))
        waveform = load_audio(path, sample_rate=self.sample_rate)
        if self.train:
            waveform = crop_first_or_random(waveform, self.sample_rate, self.duration)
        else:
            waveform = pad_or_trim(
                waveform, int(self.sample_rate * self.duration), random_crop=False
            )
        spec = log_mel_spectrogram(
            waveform, sample_rate=self.sample_rate, **self.spectrogram_kwargs
        )
        target = encode_multihot(
            row[self.primary_col],
            row.get(self.secondary_col, None),
            self.label_map,
            include_secondary=self.include_secondary,
        )
        return torch.from_numpy(spec), torch.from_numpy(target)


class PseudoLabelAudioDataset(Dataset):
    """Load fixed soundscape windows with soft pseudo-label targets.

    The pseudo-label table must contain ``audio_path`` and ``chunk_index``
    columns plus one probability column for every class. This keeps the
    generated teacher output traceable to the exact audio window used during
    student training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        classes: Sequence[str],
        sample_rate: int = 32000,
        duration: float = 5.0,
        spectrogram_kwargs: Optional[dict] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for PseudoLabelAudioDataset.")
        required = {"audio_path", "chunk_index", *classes}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Pseudo-label table is missing columns: {missing}")
        self.df = df.reset_index(drop=True)
        self.classes = list(classes)
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.spectrogram_kwargs = spectrogram_kwargs or {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        waveform = load_audio(
            Path(str(row["audio_path"])), sample_rate=self.sample_rate
        )
        start = int(row["chunk_index"]) * self.target_length
        window = pad_or_trim(waveform[start:], self.target_length, random_crop=False)
        spec = log_mel_spectrogram(
            window,
            sample_rate=self.sample_rate,
            **self.spectrogram_kwargs,
        )
        target = row[self.classes].to_numpy(dtype=np.float32)
        return torch.from_numpy(spec), torch.from_numpy(target)


CLASS_ORDER = [
    "grekis",
    "compau",
    "trokin",
    "roahaw",
    "banana",
    "whtdov",
    "socfly1",
    "yeofly1",
    "bobfly1",
    "wbwwre1",
    "soulap1",
    "sobtyr1",
    "trsowl",
    "laufal1",
    "strcuc1",
    "bbwduc",
    "saffin",
    "amekes",
    "tropar",
    "compot1",
    "blbgra1",
    "bubwre1",
    "strfly1",
    "gycwor1",
    "greegr",
    "linwoo1",
    "pirfly1",
    "littin1",
    "bkmtou1",
    "yercac1",
    "butsal1",
    "smbani",
    "bugtan",
    "chbant1",
    "yebela1",
    "rutjac1",
    "cotfly1",
    "whbman1",
    "yehcar1",
    "solsan",
    "rumfly1",
    "yecspi2",
    "blhpar1",
    "creoro1",
    "paltan1",
    "rinkin1",
    "orcpar",
    "stbwoo2",
    "speowl1",
    "yebfly1",
    "plbwoo1",
    "yebsee1",
    "bkcdon",
    "strher",
    "y00678",
    "babwar",
    "strowl1",
    "gybmar",
    "cocwoo1",
    "secfly1",
    "thbeup1",
    "pavpig2",
    "baymac",
    "rtlhum",
    "purgal2",
    "colcha1",
    "crcwoo1",
    "ywcpar",
    "chfmac1",
    "rugdov",
    "gohman1",
    "watjac1",
    "grnkin",
    "greani1",
    "whfant1",
    "cattyr",
    "srwswa1",
    "blbwre1",
    "mastit1",
    "greibi1",
    "snoegr",
    "41663",
    "leagre",
    "blcjay1",
    "grbhaw1",
    "eardov1",
    "blcant4",
    "whbant1",
    "yectyr1",
    "rufmot1",
    "thlsch3",
    "cargra1",
    "bicwre1",
    "anhing",
    "neocor",
    "shtfly1",
    "recwoo1",
    "amakin1",
    "ragmac1",
    "grasal4",
    "gretin1",
    "65448",
    "spepar1",
    "fotfly",
    "ruther1",
    "yehbla2",
    "cregua1",
    "21211",
    "whttro1",
    "brtpar1",
    "rubsee1",
    "blkvul",
    "verfly",
    "cinbec1",
    "labter1",
    "grepot1",
    "palhor2",
    "yelori1",
    "517119",
    "colara1",
    "crbtan1",
    "rebbla1",
    "piepuf1",
    "savhaw1",
    "blchaw1",
    "22973",
    "crebob1",
    "whwswa1",
    "spbwoo1",
    "22333",
    "bucmot3",
    "22976",
    "tbsfin1",
    "cocher1",
    "royfly1",
    "bobher1",
    "olipic1",
    "plukit1",
    "whmtyr1",
    "rosspo1",
    "52884",
    "65373",
    "blctit1",
    "50186",
    "ampkin1",
    "bafibi1",
    "woosto",
    "555086",
    "grysee1",
    "566513",
    "65962",
    "48124",
    "bubcur1",
    "42007",
    "piwtyr1",
    "rutpuf1",
    "715170",
    "65349",
    "65344",
    "41970",
    "shghum1",
    "norscr1",
    "sahpar1",
    "67252",
    "24322",
    "turvul",
    "135045",
    "65547",
    "787625",
    "1462737",
    "plctan1",
    "555142",
    "126247",
    "65336",
    "1564122",
    "24272",
    "548639",
    "46010",
    "1346504",
    "963335",
    "476538",
    "714022",
    "66893",
    "134933",
    "1192948",
    "868458",
    "523060",
    "24292",
    "65419",
    "1194042",
    "1462711",
    "81930",
    "67082",
    "66578",
    "66531",
    "66016",
    "21038",
    "41778",
    "21116",
    "64862",
    "528041",
    "476537",
    "47067",
    "42113",
    "42087",
    "1139490",
]

LABEL2IDX = {label: idx for idx, label in enumerate(CLASS_ORDER)}

IDX2LABEL = {idx: label for label, idx in LABEL2IDX.items()}


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


def find_by_name(
    roots: Sequence[Path], name: str, is_dir: bool = False
) -> Optional[Path]:
    for root in roots:
        if not root or not root.exists():
            continue
        candidate = root / name
        if candidate.exists() and (
            candidate.is_dir() if is_dir else candidate.is_file()
        ):
            return candidate
    for root in roots:
        if not root or not root.exists():
            continue
        for candidate in root.rglob(name):
            if candidate.exists() and (
                candidate.is_dir() if is_dir else candidate.is_file()
            ):
                return candidate
    return None


def _path_or_none(value: Optional[str]) -> Optional[Path]:
    return Path(value) if value else None


def discover_paths(config: BirdCLEFTrainingConfig) -> DataPaths:
    explicit_root = _path_or_none(config.data_root)
    candidate_roots = [explicit_root]
    if config.auto_discover_paths:
        candidate_roots.extend(
            [
                _path_or_none(config.extra_data_root),
                Path.cwd(),
                Path("./input/birdclef-2025"),
                Path("./data/birdclef-2025"),
                Path("../input/birdclef-2025"),
            ]
        )
    existing_roots = [p for p in candidate_roots if p is not None and p.exists()]
    official_candidates = [explicit_root]
    if config.auto_discover_paths:
        official_candidates.extend(
            [
                Path("./input/birdclef-2025"),
                Path("./data/birdclef-2025"),
                Path("../input/birdclef-2025"),
                Path.cwd(),
            ]
        )
    official_root = first_existing(official_candidates)
    explicit_extra_root = _path_or_none(config.extra_data_root)

    paths = DataPaths(
        official_root=official_root,
        extra_root=(
            explicit_extra_root
            if explicit_extra_root and explicit_extra_root.exists()
            else None
        ),
        train_audio=_path_or_none(config.train_audio_dir)
        or find_by_name(existing_roots, "train_audio", is_dir=True),
        train_soundscapes=_path_or_none(config.train_soundscapes_dir)
        or find_by_name(existing_roots, "train_soundscapes", is_dir=True),
        test_soundscapes=_path_or_none(config.test_soundscapes_dir)
        or find_by_name(existing_roots, "test_soundscapes", is_dir=True),
        train_csv=_path_or_none(config.train_csv)
        or find_by_name(existing_roots, "train.csv"),
        taxonomy_csv=_path_or_none(config.taxonomy_csv)
        or find_by_name(existing_roots, "taxonomy.csv"),
        sample_submission_csv=_path_or_none(config.sample_submission_csv)
        or find_by_name(existing_roots, "sample_submission.csv"),
        recording_location_txt=_path_or_none(config.recording_location_txt)
        or find_by_name(existing_roots, "recording_location.txt"),
    )
    print("Discovered paths:")
    for key, value in paths.__dict__.items():
        print(f"  {key}: {value}")
    if paths.extra_root is None:
        print(
            "Warning: optional extra data root not found; continuing with official data only."
        )
    return paths


def infer_filename_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["filename", "filepath", "file", "path", "audio_path"]:
        if col in df.columns:
            return col
    return None


def resolve_audio_path(
    row: pd.Series, train_audio_dir: Optional[Path]
) -> Optional[Path]:
    if train_audio_dir is None:
        return None
    filename_col = infer_filename_column(pd.DataFrame([row]))
    primary = str(row.get("primary_label", ""))
    raw_name = str(row.get(filename_col, "")) if filename_col else ""
    candidates = []
    if raw_name:
        raw = Path(raw_name)
        candidates.extend(
            [raw, train_audio_dir / raw_name, train_audio_dir / primary / raw.name]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if raw_name:
        matches = list(train_audio_dir.rglob(Path(raw_name).name))
        if matches:
            return matches[0]
    return None


def build_focal_dataframe(
    train_df: pd.DataFrame, train_audio_dir: Optional[Path]
) -> pd.DataFrame:
    if train_df.empty:
        return train_df.copy()
    df = train_df.copy()
    df["resolved_path"] = df.apply(
        lambda row: resolve_audio_path(row, train_audio_dir), axis=1
    )
    missing = int(df["resolved_path"].isna().sum())
    if missing:
        print(
            f"Warning: {missing} focal rows could not be resolved to audio files and will be dropped."
        )
    df = df[df["resolved_path"].notna()].reset_index(drop=True)
    if "secondary_labels" not in df.columns:
        df["secondary_labels"] = "[]"
    return df


def target_from_row(row: pd.Series, frames_per_clip: int) -> np.ndarray:
    y = np.zeros(len(CLASS_ORDER), dtype=np.float32)
    primary = str(row.get("primary_label", ""))
    if primary in LABEL2IDX:
        y[LABEL2IDX[primary]] = 1.0
    for label in parse_secondary_labels(row.get("secondary_labels", "[]")):
        if label in LABEL2IDX:
            y[LABEL2IDX[label]] = 1.0
    return np.repeat(y[None, :], frames_per_clip, axis=0)


class LabeledFocalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: BirdCLEFTrainingConfig,
        mode: str = "train",
        augment: bool = False,
        strength: str = "moderate",
        padding_mode: str = "zero_left",
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.mode = mode
        self.augment = augment
        self.strength = strength
        self.padding_mode = padding_mode
        self.target_len = int(config.sample_rate * config.train_duration_sec)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        wave = load_audio(
            Path(row["resolved_path"]), sample_rate=self.config.sample_rate
        )
        wave = crop_or_pad_wave(
            wave, self.target_len, mode=self.mode, pad_mode=self.padding_mode
        )
        if self.augment:
            wave = augment_wave(wave, self.config, strength=self.strength)
        target = target_from_row(row, self.config.frames_per_clip)
        return {
            "wave": torch.tensor(wave, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "source": "focal",
            "filename": Path(row["resolved_path"]).name,
        }


def build_soundscape_windows(
    files: Sequence[Path], config: BirdCLEFTrainingConfig
) -> pd.DataFrame:
    rows = []
    for path in files:
        try:
            duration = librosa.get_duration(path=str(path))
        except Exception:
            duration = config.train_duration_sec
        max_start = max(0.0, duration - config.train_duration_sec)
        starts = np.arange(0.0, max_start + 1e-6, config.soundscape_stride_sec)
        if len(starts) == 0:
            starts = np.array([0.0])
        for start in starts:
            rows.append(
                {
                    "filepath": str(path),
                    "filename": path.stem,
                    "window_start_sec": float(start),
                    "window_end_sec": float(start + config.train_duration_sec),
                }
            )
    return pd.DataFrame(rows)


class UnlabeledSoundscapeDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        config: BirdCLEFTrainingConfig,
        augment: bool = False,
        strength: str = "none",
    ):
        self.index = build_soundscape_windows(files, config)
        self.config = config
        self.augment = augment
        self.strength = strength
        self.target_len = int(config.sample_rate * config.train_duration_sec)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.index.iloc[idx]
        wave = load_audio(
            Path(row["filepath"]),
            sample_rate=self.config.sample_rate,
            offset=float(row["window_start_sec"]),
            duration=self.config.train_duration_sec,
        )
        wave = crop_or_pad_wave(
            wave, self.target_len, mode="valid", pad_mode="zero_right"
        )
        if self.augment:
            wave = augment_wave(wave, self.config, strength=self.strength)
        return {
            "wave": torch.tensor(wave, dtype=torch.float32),
            "filename": row["filename"],
            "filepath": row["filepath"],
            "window_start_sec": float(row["window_start_sec"]),
        }


def read_pseudo_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


class PseudoLabeledSoundscapeDataset(Dataset):
    def __init__(
        self,
        pseudo_path: Path,
        config: BirdCLEFTrainingConfig,
        augment: bool = False,
        strength: str = "strong",
    ):
        self.pseudo_df = read_pseudo_table(pseudo_path)
        self.config = config
        self.augment = augment
        self.strength = strength
        self.groups = list(
            self.pseudo_df.groupby(["filename", "window_start_sec"], sort=False)
        )
        soundscape_to_group_indices: Dict[str, List[int]] = {}
        for group_index, (_, group) in enumerate(self.groups):
            filepath = str(group["filepath"].iloc[0])
            soundscape_to_group_indices.setdefault(filepath, []).append(group_index)
        self.soundscape_group_indices = list(soundscape_to_group_indices.values())
        if "soundscape_weight" in self.pseudo_df.columns:
            weights_by_path = (
                self.pseudo_df.groupby("filepath", sort=False)["soundscape_weight"]
                .first()
                .astype(float)
            )
        else:
            weights_by_path = (
                self.pseudo_df.groupby("filepath", sort=False)[CLASS_ORDER]
                .max()
                .sum(axis=1)
                .astype(float)
            )
        weights_by_path.index = weights_by_path.index.astype(str)
        soundscape_weights = [
            max(float(weights_by_path.get(filepath, 0.0)), 1e-8)
            for filepath in soundscape_to_group_indices
        ]
        self.soundscape_weights = np.asarray(soundscape_weights, dtype=np.float64)
        self.soundscape_probabilities = (
            self.soundscape_weights / self.soundscape_weights.sum()
        )
        self.target_len = int(config.sample_rate * config.train_duration_sec)

    def __len__(self) -> int:
        return len(self.groups)

    def _build_item(self, idx: int) -> Dict[str, Any]:
        (_, window_start_sec), group = self.groups[idx]
        group = group.sort_values("start_sec")
        filepath = Path(group["filepath"].iloc[0])
        wave = load_audio(
            filepath,
            self.config.sample_rate,
            offset=float(window_start_sec),
            duration=self.config.train_duration_sec,
        )
        wave = crop_or_pad_wave(
            wave, self.target_len, mode="train", pad_mode="zero_random"
        )
        if self.augment:
            wave = augment_wave(wave, self.config, strength=self.strength)
        target = np.zeros(
            (self.config.frames_per_clip, len(CLASS_ORDER)), dtype=np.float32
        )
        values = group[CLASS_ORDER].values.astype(np.float32)
        n = min(len(values), self.config.frames_per_clip)
        target[:n] = values[:n]
        if n and n < self.config.frames_per_clip:
            target[n:] = target[n - 1]
        return {
            "wave": torch.tensor(wave, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "source": "pseudo",
            "filename": str(group["filename"].iloc[0]),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._build_item(idx)

    def sample_weighted(self) -> Dict[str, Any]:
        soundscape_index = int(
            np.random.choice(
                len(self.soundscape_group_indices), p=self.soundscape_probabilities
            )
        )
        group_indices = self.soundscape_group_indices[soundscape_index]
        group_index = int(group_indices[np.random.randint(0, len(group_indices))])
        return self._build_item(group_index)


class MixedNoisyStudentDataset(Dataset):
    def __init__(
        self, focal_ds: Dataset, pseudo_ds: Dataset, config: BirdCLEFTrainingConfig
    ):
        self.focal_ds = focal_ds
        self.pseudo_ds = pseudo_ds
        self.config = config

    def __len__(self) -> int:
        return max(len(self.focal_ds), len(self.pseudo_ds))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        focal = self.focal_ds[idx % len(self.focal_ds)]
        if hasattr(self.pseudo_ds, "sample_weighted"):
            pseudo = self.pseudo_ds.sample_weighted()
        else:
            pseudo = self.pseudo_ds[np.random.randint(0, len(self.pseudo_ds))]
        if self.config.mixup_focal_pseudo:
            wave, target, lam = mixup_focal_pseudo(
                focal["wave"],
                focal["target"],
                pseudo["wave"],
                pseudo["target"],
                self.config.pseudo_mixup_lambda,
            )
            return {
                "wave": wave,
                "target": target,
                "source": f"mixup:{lam:.3f}",
                "filename": focal["filename"],
            }
        return focal if np.random.rand() < 0.5 else pseudo


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "wave": torch.stack([item["wave"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "source": [item.get("source", "") for item in batch],
        "filename": [item.get("filename", "") for item in batch],
    }
