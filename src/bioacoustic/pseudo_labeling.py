"""Pseudo-label generation and iterative teacher-student utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import CLASS_ORDER, UnlabeledSoundscapeDataset
from .models import forward_sed_logits, load_model_weights
from .utils import log, save_json

if TYPE_CHECKING:
    from .training import BirdCLEFTrainingConfig


def select_pseudo_labels(
    predictions: np.ndarray,
    row_ids: Optional[List[str]] = None,
    min_max_prob: float = 0.5,
    class_prob_threshold: float = 0.1,
) -> pd.DataFrame:
    """Select pseudo-labeled chunks from model probabilities.
    Keeps chunks whose maximum class probability is at least min_max_prob,
    then zeroes out class probabilities below class_prob_threshold.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    keep = predictions.max(axis=1) >= min_max_prob
    selected = predictions[keep].copy()
    selected[selected < class_prob_threshold] = 0.0
    if row_ids is None:
        row_ids = [f"chunk_{i}" for i in range(len(predictions))]
    selected_ids = np.asarray(row_ids)[keep]
    df = pd.DataFrame(selected)
    df.insert(0, "row_id", selected_ids)
    return df


def save_pseudo_labels(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_pseudo_labels(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def mix_labeled_and_pseudo(
    labeled_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    labeled_ratio: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a mixed training table with labeled and pseudo-labeled rows."""
    rng = np.random.default_rng(seed)
    n_labeled = len(labeled_df)
    n_pseudo = int(n_labeled * (1.0 - labeled_ratio) / max(labeled_ratio, 1e-6))
    pseudo_sample = (
        pseudo_df.sample(n=min(n_pseudo, len(pseudo_df)), random_state=seed)
        if len(pseudo_df)
        else pseudo_df
    )
    a = labeled_df.copy()
    a["source"] = "labeled"
    b = pseudo_sample.copy()
    b["source"] = "pseudo"
    return pd.concat([a, b], ignore_index=True)


def apply_power_scaling(preds: np.ndarray, power: float) -> np.ndarray:
    # power < 1 softens probabilities; power > 1 sharpens probabilities.
    return np.power(np.clip(preds, 0.0, 1.0), power)


def pseudo_power_for_iteration(iteration: int, config: BirdCLEFTrainingConfig) -> float:
    return getattr(config, f"pseudo_power_iter{iteration}", config.pseudo_power)


class PseudoLabeler:
    def __init__(self, config: BirdCLEFTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def generate(
        self,
        model: torch.nn.Module,
        soundscape_files: Sequence[Path],
        output_path: Path,
        checkpoint_path: Path | None = None,
    ) -> Path:
        if checkpoint_path is not None:
            log(f"Pseudo-labeler: loading checkpoint {checkpoint_path}")
            load_model_weights(model, checkpoint_path, strict=True)

        model.to(self.device).eval()
        ds = UnlabeledSoundscapeDataset(soundscape_files, self.config, augment=False)
        loader = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.config.num_workers > 0),
        )

        rows = []
        max_confidences = []
        class_sums = np.zeros(len(CLASS_ORDER), dtype=np.float64)
        total_batches = len(loader)
        log(
            f"Pseudo-labeler: started with files={len(soundscape_files)}, windows={len(ds)}, "
            f"batches={total_batches}, output={output_path}, power={self.config.pseudo_power}"
        )
        for step, batch in enumerate(
            tqdm(loader, desc="pseudo-label soundscapes", leave=True, mininterval=5),
            start=1,
        ):
            waves = batch["wave"].to(self.device)
            logits = forward_sed_logits(model, waves, self.config.frames_per_clip)
            probs = torch.sigmoid(logits).cpu().numpy()
            if self.config.pseudo_power != 1.0:
                probs = apply_power_scaling(probs, self.config.pseudo_power)
            if self.config.pseudo_top_k is not None:
                kth = np.partition(probs, -self.config.pseudo_top_k, axis=-1)[
                    :, :, -self.config.pseudo_top_k
                ][..., None]
                probs = np.where(probs >= kth, probs, 0.0)

            for b in range(probs.shape[0]):
                window_start = float(batch["window_start_sec"][b])
                filename = str(batch["filename"][b])
                filepath = str(batch["filepath"][b])
                for frame in range(self.config.frames_per_clip):
                    frame_probs = probs[b, frame].astype(np.float32)
                    max_conf = float(frame_probs.max())
                    if max_conf < self.config.min_confidence_for_retention:
                        continue
                    class_sums += frame_probs
                    max_confidences.append(max_conf)
                    start_sec = window_start + frame * self.config.label_frame_sec
                    row = {
                        "filename": filename,
                        "filepath": filepath,
                        "window_start_sec": window_start,
                        "start_sec": start_sec,
                        "end_sec": start_sec + self.config.label_frame_sec,
                    }
                    row.update(
                        {
                            label: float(frame_probs[i])
                            for i, label in enumerate(CLASS_ORDER)
                        }
                    )
                    rows.append(row)
            if self.config.log_every_steps and (
                step == 1
                or step % self.config.log_every_steps == 0
                or step == total_batches
            ):
                mean_conf = float(np.mean(max_confidences)) if max_confidences else 0.0
                log(
                    f"Pseudo-labeler: batch {step}/{total_batches}, rows={len(rows)}, mean_max_conf={mean_conf:.5f}"
                )

        pseudo_df = pd.DataFrame(rows)
        soundscape_weights = {}
        if len(pseudo_df):
            soundscape_weights = (
                pseudo_df.groupby("filepath", sort=False)[CLASS_ORDER]
                .max()
                .sum(axis=1)
                .astype(float)
                .to_dict()
            )
            pseudo_df["soundscape_weight"] = (
                pseudo_df["filepath"].map(soundscape_weights).astype(float)
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if output_path.suffix != ".parquet":
                output_path = output_path.with_suffix(".parquet")
            pseudo_df.to_parquet(output_path, index=False)
        except Exception:
            output_path = output_path.with_suffix(".csv")
            pseudo_df.to_csv(output_path, index=False)

        stats = {
            "num_rows": int(len(pseudo_df)),
            "num_windows": int(len(ds)),
            "num_5s_frames": int(len(pseudo_df)),
            "mean_max_confidence": (
                float(np.mean(max_confidences)) if max_confidences else 0.0
            ),
            "max_confidence": (
                float(np.max(max_confidences)) if max_confidences else 0.0
            ),
            "pseudo_threshold": self.config.pseudo_threshold,
            "pseudo_power": self.config.pseudo_power,
            "soundscape_weight_min": (
                float(min(soundscape_weights.values())) if soundscape_weights else 0.0
            ),
            "soundscape_weight_mean": (
                float(np.mean(list(soundscape_weights.values())))
                if soundscape_weights
                else 0.0
            ),
            "soundscape_weight_max": (
                float(max(soundscape_weights.values())) if soundscape_weights else 0.0
            ),
            "top_predicted_classes": {},
            "mean_confidence_per_class": {},
        }
        if len(pseudo_df):
            means = pseudo_df[CLASS_ORDER].mean().sort_values(ascending=False)
            stats["top_predicted_classes"] = {
                k: float(v) for k, v in means.head(30).items()
            }
            stats["mean_confidence_per_class"] = {
                label: float(means.get(label, 0.0)) for label in CLASS_ORDER
            }
        else:
            stats["mean_confidence_per_class"] = {label: 0.0 for label in CLASS_ORDER}

        save_json(stats, output_path.with_suffix(".stats.json"))
        log(f"Saved pseudo labels: {output_path} rows={len(pseudo_df)}")
        return output_path
