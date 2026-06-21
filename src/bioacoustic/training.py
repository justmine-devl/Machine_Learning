"""Generic and iterative teacher-student training orchestration."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .audio import SpecAugment
from .dataset import (
    CLASS_ORDER,
    DataPaths,
    LabeledFocalDataset,
    MixedNoisyStudentDataset,
    PseudoLabeledSoundscapeDataset,
    build_focal_dataframe,
    collate_batch,
    discover_paths,
    parse_secondary_labels,
)
from .losses import compute_loss, make_pos_weight
from .metrics import compute_birdclef_metrics, compute_multilabel_metrics
from .models import (
    ModelsGroupConfig,
    build_birdclef_model,
    forward_sed_logits,
    freeze_backbone_if_needed,
    load_model_weights,
    make_optimizer,
)
from .pseudo_labeling import PseudoLabeler, pseudo_power_for_iteration
from .utils import (
    effective_epochs,
    effective_iterations,
    ensure_output_dirs,
    log,
    print_environment,
    save_json,
    seed_everything,
)
from .visualization import ReportPlotter


@dataclass
class TrainState:
    epoch: int
    train_loss: float
    valid_loss: float
    metrics: Dict[str, float]


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: str = "cuda",
    scheduler: Optional[object] = None,
) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out["clip_logits"], y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Optional[Callable] = None,
    device: str = "cuda",
) -> Dict[str, object]:
    model.eval()
    preds, targets, losses = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        out = model(x)
        logits = out["clip_logits"]
        if loss_fn is not None:
            losses.append(float(loss_fn(logits, y).detach().cpu()))
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0) if preds else np.zeros((0, 0))
    y_true = np.concatenate(targets, axis=0) if targets else np.zeros((0, 0))
    metrics = compute_multilabel_metrics(y_true, y_pred) if y_pred.size else {}
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer=None,
    epoch: int = 0,
    metrics=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer=None, map_location="cpu"
) -> dict:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


@dataclass
class BirdCLEFTrainingConfig:
    # Paths
    data_root: Optional[str] = None
    train_audio_dir: Optional[str] = None
    train_soundscapes_dir: Optional[str] = None
    test_soundscapes_dir: Optional[str] = None
    train_csv: Optional[str] = None
    taxonomy_csv: Optional[str] = None
    sample_submission_csv: Optional[str] = None
    recording_location_txt: Optional[str] = None
    extra_data_root: Optional[str] = None
    output_dir: str = "./outputs"
    resume_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    plots_dir: Optional[str] = None
    pseudo_labels_dir: Optional[str] = None
    stage0_dir: Optional[str] = None
    student_stage_template: str = "stage{iteration}_student_iter{iteration}"
    resume_logs_dir: Optional[str] = None
    resume_pseudo_labels_dir: Optional[str] = None
    resume_stage0_dir: Optional[str] = None
    resume_student_stage_template: str = "stage{iteration}_student_iter{iteration}"
    auto_discover_paths: bool = False

    # Execution
    debug: bool = True
    seed: int = 42
    deterministic: bool = False
    auto_resume: bool = True
    reuse_pseudo: bool = True
    force_retrain: bool = False
    plot_only: bool = False

    # Debug limits
    debug_num_train_files: int = 200
    debug_num_soundscapes: int = 20
    debug_epochs: int = 1
    debug_num_noisy_student_iterations: int = 1

    # Inference-compatible audio/model config
    sample_rate: int = 32000
    train_duration_sec: int = 20
    label_frame_sec: int = 5
    soundscape_stride_sec: int = 5 
    model_key: str = "model_config_6"
    pretrained_backbone: bool = False
    strict_class_order: bool = False

    # Training
    batch_size: int = 2  
    gradient_accumulation_steps: int = 8
    num_workers: int = 0
    epochs_teacher: int = 8  
    epochs_student: int = 5  
    num_noisy_student_iterations: int = 2
    learning_rate: float = 3e-4  
    head_learning_rate: float = 1e-3  
    min_learning_rate: float = 1e-6
    restart_epochs: int = 5
    weight_decay: float = 1e-4  
    gradient_clip_norm: float = 1.0  
    mixed_precision: bool = True
    train_backbone: bool = True
    save_every_epochs: int = 0
    save_best_training_checkpoint: bool = False
    log_every_steps: int = 50

    # Validation
    n_folds: int = 5
    fold: int = 0
    val_threshold: float = 0.5  

    # Loss
    loss_type: str = "cross_entropy"
    focal_gamma: float = 2.0  
    focal_alpha: Optional[float] = None  
    label_smoothing: float = 0.0
    use_pos_weight: bool = False

    # Augmentation
    random_gain_db: float = 6.0  
    gaussian_noise_std: float = 0.003 
    time_shift_sec: float = 1.0  
    time_mask_param: int = 48  
    freq_mask_param: int = 24  
    teacher_aug_strength: str = "moderate"
    student_aug_strength: str = "strong"
    mixup_focal_pseudo: bool = True
    pseudo_mixup_lambda: float = 0.5
    student_drop_path_rate: float = 0.15

    # Pseudo-labeling
    pseudo_threshold: float = 0.0
    pseudo_top_k: Optional[int] = None
    pseudo_power: float = 1.0
    pseudo_power_iter1: float = 1.0
    pseudo_power_iter2: float = 1.5  
    pseudo_power_iter3: float = 2.0  
    pseudo_power_iter4: float = 1.6666666667
    min_confidence_for_retention: float = 0.0

    def __post_init__(self) -> None:
        if self.restart_epochs < 1:
            raise ValueError("restart_epochs must be at least 1")
        if not 0.0 <= self.pseudo_mixup_lambda <= 1.0:
            raise ValueError("pseudo_mixup_lambda must be in [0, 1]")
        if not 0.0 <= self.student_drop_path_rate < 1.0:
            raise ValueError("student_drop_path_rate must be in [0, 1)")

    @property
    def frames_per_clip(self) -> int:
        return int(self.train_duration_sec // self.label_frame_sec)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def resume_path(self) -> Optional[Path]:
        return Path(self.resume_dir) if self.resume_dir else None

    @property
    def logs_path(self) -> Path:
        return Path(self.logs_dir) if self.logs_dir else self.output_path / "logs"

    @property
    def plots_path(self) -> Path:
        return Path(self.plots_dir) if self.plots_dir else self.output_path / "plots"

    @property
    def pseudo_labels_path(self) -> Path:
        return (
            Path(self.pseudo_labels_dir)
            if self.pseudo_labels_dir
            else self.output_path / "pseudo_labels"
        )

    @property
    def stage0_path(self) -> Path:
        return (
            Path(self.stage0_dir)
            if self.stage0_dir
            else self.output_path / "stage0_teacher"
        )

    def student_stage_path(self, iteration: int) -> Path:
        rendered = self.student_stage_template.format(iteration=iteration)
        path = Path(rendered)
        return path if path.is_absolute() else self.output_path / path

    @property
    def resume_logs_path(self) -> Optional[Path]:
        if self.resume_logs_dir:
            return Path(self.resume_logs_dir)
        return self.resume_path / "logs" if self.resume_path else None

    @property
    def resume_pseudo_labels_path(self) -> Optional[Path]:
        if self.resume_pseudo_labels_dir:
            return Path(self.resume_pseudo_labels_dir)
        return self.resume_path / "pseudo_labels" if self.resume_path else None

    @property
    def resume_stage0_path(self) -> Optional[Path]:
        if self.resume_stage0_dir:
            return Path(self.resume_stage0_dir)
        return self.resume_path / "stage0_teacher" if self.resume_path else None

    def resume_student_stage_path(self, iteration: int) -> Optional[Path]:
        if self.resume_path is None and not self.resume_student_stage_template:
            return None
        rendered = self.resume_student_stage_template.format(iteration=iteration)
        path = Path(rendered)
        if path.is_absolute():
            return path
        return self.resume_path / path if self.resume_path else None

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )


EPOCH_CHECKPOINT_PATTERNS = ("epoch_*.training.pt", "epoch_*.pth", "epoch_*.pt")


def extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def checkpoint_epoch_from_path(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def checkpoint_epoch(payload, path: Path) -> int:
    if isinstance(payload, dict) and payload.get("epoch") is not None:
        return int(payload.get("epoch", 0))
    return checkpoint_epoch_from_path(path)


def read_checkpoint_epoch(path: Path) -> int:
    try:
        payload = torch.load(path, map_location="cpu")
        return checkpoint_epoch(payload, path)
    except Exception:
        return checkpoint_epoch_from_path(path)


def find_latest_training_checkpoint(stage_dir: Optional[Path]) -> Optional[Path]:
    if stage_dir is None or not stage_dir.exists():
        return None
    last_path = stage_dir / "last.pth"
    if last_path.exists():
        return last_path
    candidates = []
    for pattern in EPOCH_CHECKPOINT_PATTERNS:
        candidates.extend(stage_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=read_checkpoint_epoch)


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def prune_epoch_checkpoints(stage_dir: Path) -> None:
    for pattern in EPOCH_CHECKPOINT_PATTERNS:
        for path in stage_dir.glob(pattern):
            _remove_if_exists(path)


def save_checkpoint_bundle(
    stage_dir: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler,
    config: BirdCLEFTrainingConfig,
    model_config: Dict,
    epoch: int,
    metrics: Dict,
    stage: str,
    is_best: bool = False,
) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "config": config.to_dict(),
        "class_order": CLASS_ORDER,
        "mel_config": ModelsGroupConfig.spectrogram_params,
        "model_config": model_config,
        "epoch": epoch,
        "metrics": metrics,
        "stage": stage,
    }
    last_path = stage_dir / "last.pth"
    torch.save(payload, last_path)
    if not config.save_best_training_checkpoint:
        _remove_if_exists(stage_dir / "best.pth")
    if is_best:
        best_path = stage_dir / "best.pth"
        if config.save_best_training_checkpoint:
            torch.save(payload, best_path)
            training_checkpoint = str(best_path)
        else:
            _remove_if_exists(best_path)
            training_checkpoint = None
        raw_path = stage_dir / "best_for_inference.pt"
        torch.save(model.state_dict(), raw_path)
        save_json(
            {
                "checkpoint_format": "raw_state_dict",
                "training_checkpoint": training_checkpoint,
                "model_key": config.model_key,
                "class_order": CLASS_ORDER,
                "mel_config": ModelsGroupConfig.spectrogram_params,
                "stage": stage,
                "epoch": epoch,
                "metrics": metrics,
            },
            stage_dir / "best_for_inference.config.json",
        )
    if config.save_every_epochs and epoch % config.save_every_epochs == 0:
        torch.save(payload, stage_dir / f"epoch_{epoch}.pth")
    else:
        prune_epoch_checkpoints(stage_dir)
    return last_path


def load_training_state(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler=None,
) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(payload))
    if not isinstance(payload, dict):
        return checkpoint_epoch_from_path(checkpoint_path)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    return checkpoint_epoch(payload, checkpoint_path)


def make_loader(dataset, config: BirdCLEFTrainingConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.num_workers > 0),
        collate_fn=collate_batch,
        drop_last=False,
    )


class StageTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        model_config: Dict,
        config: BirdCLEFTrainingConfig,
        stage_dir: Path,
        stage_name: str,
        pos_weight: Optional[torch.Tensor] = None,
        spec_aug_strength: str = "moderate",
        resume_stage_dir: Optional[Path] = None,
    ):
        self.model = model
        self.model_config = model_config
        self.config = config
        self.stage_dir = stage_dir
        self.stage_name = stage_name
        self.resume_stage_dir = resume_stage_dir
        self.pos_weight = pos_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        freeze_backbone_if_needed(self.model, config)
        self.optimizer = make_optimizer(self.model, config)
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.mixed_precision and self.device.type == "cuda"
        )
        self.spec_aug = None
        if spec_aug_strength != "none":
            self.spec_aug = SpecAugment(
                config.time_mask_param, config.freq_mask_param
            ).to(self.device)

    @property
    def last_path(self) -> Path:
        return self.stage_dir / "last.pth"

    @property
    def best_path(self) -> Path:
        return self.stage_dir / "best.pth"

    @property
    def best_inference_path(self) -> Path:
        return self.stage_dir / "best_for_inference.pt"

    @property
    def resume_last_path(self) -> Optional[Path]:
        if self.resume_stage_dir is None:
            return None
        return self.resume_stage_dir / "last.pth"

    @property
    def resume_best_inference_path(self) -> Optional[Path]:
        if self.resume_stage_dir is None:
            return None
        return self.resume_stage_dir / "best_for_inference.pt"

    def _existing_last_path(self) -> Optional[Path]:
        candidates = [
            path
            for path in [
                find_latest_training_checkpoint(self.stage_dir),
                find_latest_training_checkpoint(self.resume_stage_dir),
            ]
            if path is not None and path.exists()
        ]
        if not candidates:
            return None
        return max(candidates, key=read_checkpoint_epoch)

    def _existing_best_inference_path(self) -> Optional[Path]:
        for path in [self.best_inference_path, self.resume_best_inference_path]:
            if path is not None and path.exists():
                return path
        return None

    def _copy_minimal_resume_artifacts(self) -> None:
        if self.resume_stage_dir is None or not self.resume_stage_dir.exists():
            return
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        resume_checkpoint = find_latest_training_checkpoint(self.resume_stage_dir)
        if resume_checkpoint is not None and not self.last_path.exists():
            shutil.copy2(resume_checkpoint, self.last_path)
            log(
                f"{self.stage_name}: copied resume checkpoint {resume_checkpoint} -> {self.last_path}"
            )
        for name in [
            "best_for_inference.pt",
            "best_for_inference.config.json",
            "train_log.csv",
        ]:
            src = self.resume_stage_dir / name
            dst = self.stage_dir / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                log(f"{self.stage_name}: copied resume artifact {src} -> {dst}")
        if not self.config.save_best_training_checkpoint:
            best_path = self.stage_dir / "best.pth"
            if best_path.exists():
                best_path.unlink()
        if not self.config.save_every_epochs:
            prune_epoch_checkpoints(self.stage_dir)

    def _append_log(self, row: Dict) -> None:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        stage_log = self.stage_dir / "train_log.csv"
        global_log = self.config.logs_path / "train_log.csv"
        for path in [stage_log, global_log]:
            if path.exists():
                df = pd.read_csv(path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            elif (
                path == stage_log
                and self.resume_stage_dir is not None
                and (self.resume_stage_dir / "train_log.csv").exists()
            ):
                df = pd.read_csv(self.resume_stage_dir / "train_log.csv")
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)

    def _best_loss_from_log(self) -> float:
        log_paths = [self.stage_dir / "train_log.csv"]
        if self.resume_stage_dir is not None:
            log_paths.append(self.resume_stage_dir / "train_log.csv")
        frames = [pd.read_csv(path) for path in log_paths if path.exists()]
        if not frames:
            return float("inf")
        df = pd.concat(frames, ignore_index=True)
        if "valid_loss" not in df.columns or df.empty:
            return float("inf")
        return float(df["valid_loss"].min())

    def _start_epoch(self, total_epochs: int) -> int:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, self.config.restart_epochs),
            T_mult=1,
            eta_min=self.config.min_learning_rate,
        )
        resume_last = self._existing_last_path()
        if (
            self.config.auto_resume
            and not self.config.force_retrain
            and resume_last is not None
        ):
            # Recreate the scheduler because resumed runs may increase the target epoch count.
            last_epoch = load_training_state(
                resume_last, self.model, self.optimizer, None, self.scaler
            )
            log(f"{self.stage_name}: resumed from {resume_last} at epoch {last_epoch}")
            return last_epoch + 1
        if self.config.auto_resume and not self.config.force_retrain:
            if self.resume_stage_dir is not None and not self.resume_stage_dir.exists():
                log(
                    f"{self.stage_name}: resume_stage_dir does not exist: {self.resume_stage_dir}"
                )
            log(f"{self.stage_name}: no resume checkpoint found; starting from epoch 1")
        return 1

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.spec_aug is not None:
            self.spec_aug.train()
        losses = []
        amp_enabled = self.config.mixed_precision and self.device.type == "cuda"
        accumulation_steps = max(1, int(self.config.gradient_accumulation_steps))
        self.optimizer.zero_grad(set_to_none=True)
        total_batches = len(loader)
        log(f"{self.stage_name}: train epoch {epoch} started ({total_batches} batches)")
        pbar = tqdm(
            loader,
            desc=f"{self.stage_name} train epoch {epoch}",
            leave=True,
            mininterval=5,
        )
        for step, batch in enumerate(pbar, start=1):
            waves = batch["wave"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = forward_sed_logits(
                    self.model,
                    waves,
                    self.config.frames_per_clip,
                    spec_augment=self.spec_aug,
                )
                loss = compute_loss(
                    logits, targets, self.config, pos_weight=self.pos_weight
                )
                loss_to_backward = loss / accumulation_steps
            self.scaler.scale(loss_to_backward).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                if self.config.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if step == 1 or step % 10 == 0:
                pbar.set_postfix(loss=f"{loss_value:.4f}")
            if self.config.log_every_steps and (
                step == 1
                or step % self.config.log_every_steps == 0
                or step == total_batches
            ):
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                log(
                    f"{self.stage_name}: train epoch {epoch} batch {step}/{total_batches} loss={loss_value:.5f} mean_loss={mean_loss:.5f}"
                )
        return {"train_loss": float(np.mean(losses)) if losses else float("nan")}

    @torch.no_grad()
    def validate_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        losses = []
        all_targets = []
        all_probs = []
        total_batches = len(loader)
        log(f"{self.stage_name}: validation started ({total_batches} batches)")
        for step, batch in enumerate(
            tqdm(loader, desc=f"{self.stage_name} valid", leave=True, mininterval=5),
            start=1,
        ):
            waves = batch["wave"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            logits = forward_sed_logits(self.model, waves, self.config.frames_per_clip)
            loss = compute_loss(
                logits, targets, self.config, pos_weight=self.pos_weight
            )
            losses.append(float(loss.detach().cpu()))
            all_targets.append(targets.cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            if self.config.log_every_steps and (
                step == 1
                or step % self.config.log_every_steps == 0
                or step == total_batches
            ):
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                log(
                    f"{self.stage_name}: valid batch {step}/{total_batches} mean_loss={mean_loss:.5f}"
                )
        if not all_targets:
            return {
                "valid_loss": float("nan"),
                "macro_auc": float("nan"),
                "macro_f1": float("nan"),
                "valid_auc_classes": 0,
            }
        metrics = compute_birdclef_metrics(
            np.concatenate(all_targets),
            np.concatenate(all_probs),
            threshold=self.config.val_threshold,
        )
        metrics["valid_loss"] = float(np.mean(losses))
        return metrics

    def fit(
        self, train_loader: DataLoader, valid_loader: DataLoader, total_epochs: int
    ) -> Path:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"{self.stage_name}: fit requested total_epochs={total_epochs}, "
            f"train_batches={len(train_loader)}, valid_batches={len(valid_loader)}, "
            f"stage_dir={self.stage_dir}"
        )
        if self.resume_stage_dir is not None:
            log(f"{self.stage_name}: resume_stage_dir={self.resume_stage_dir}")
        if (
            self.config.auto_resume
            and not self.config.force_retrain
            and self._existing_last_path() is not None
            and self._existing_best_inference_path() is not None
        ):
            try:
                existing_checkpoint = self._existing_last_path()
                payload = torch.load(existing_checkpoint, map_location="cpu")
                existing_epoch = checkpoint_epoch(payload, existing_checkpoint)
                if existing_epoch >= total_epochs:
                    self._copy_minimal_resume_artifacts()
                    best_path = (
                        self.best_inference_path
                        if self.best_inference_path.exists()
                        else self._existing_best_inference_path()
                    )
                    log(
                        f"{self.stage_name}: already completed {total_epochs} epochs from {existing_checkpoint}; using {best_path}"
                    )
                    return best_path
            except Exception:
                pass

        start_epoch = self._start_epoch(total_epochs)
        best_loss = self._best_loss_from_log()
        log(
            f"{self.stage_name}: starting loop at epoch {start_epoch}/{total_epochs}; current_best_loss={best_loss:.5f}"
        )
        for epoch in range(start_epoch, total_epochs + 1):
            log(f"{self.stage_name}: epoch {epoch}/{total_epochs} started")
            train_metrics = self.train_one_epoch(train_loader, epoch)
            valid_metrics = self.validate_one_epoch(valid_loader)
            # Absolute epoch keeps the five-epoch restart phase correct after resume.
            self.scheduler.step(epoch)
            row = {
                "stage": self.stage_name,
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **train_metrics,
                **valid_metrics,
            }
            log(f"{self.stage_name}: epoch {epoch}/{total_epochs} metrics {row}")
            valid_loss = valid_metrics.get("valid_loss", float("inf"))
            is_best = bool(np.isfinite(valid_loss) and valid_loss <= best_loss)
            if is_best:
                best_loss = valid_loss
            save_checkpoint_bundle(
                self.stage_dir,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.config,
                self.model_config,
                epoch,
                row,
                self.stage_name,
                is_best=is_best,
            )
            self._append_log(row)
            log(
                f"{self.stage_name}: epoch {epoch}/{total_epochs} saved last={self.last_path} "
                f"best_inference={self.best_inference_path if self.best_inference_path.exists() else 'not_yet'} "
                f"is_best={is_best}"
            )
        if self.best_inference_path.exists():
            log(f"{self.stage_name}: finished; returning {self.best_inference_path}")
            return self.best_inference_path
        log(f"{self.stage_name}: finished; returning {self.last_path}")
        return self.last_path


try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None


def read_csv_optional(path: Optional[Path], name: str) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        log(f"Warning: {name} not found.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    log(f"Loaded {name}: {df.shape} from {path}")
    return df


def validate_class_order(sample_submission: pd.DataFrame, strict: bool = False) -> None:
    if sample_submission.empty:
        log("sample_submission.csv missing; using inference class order.")
        return
    sample_cols = [c for c in sample_submission.columns if c != "row_id"]
    if sample_cols == CLASS_ORDER:
        log("Class order check passed: sample_submission matches inference order.")
        return
    if set(sample_cols) == set(CLASS_ORDER):
        log(
            "Warning: sample_submission has the same classes but different order. Using inference order for checkpoint compatibility."
        )
        return
    missing = sorted(set(CLASS_ORDER) - set(sample_cols))
    extra = sorted(set(sample_cols) - set(CLASS_ORDER))
    msg = f"Class set mismatch. missing_in_sample={missing[:10]}, extra_in_sample={extra[:10]}"
    if strict:
        raise ValueError(msg)
    log(f"Warning: {msg}")
    log("Continuing with inference class order.")


def make_folds(df: pd.DataFrame, config: BirdCLEFTrainingConfig) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if df.empty:
        return df
    y = (
        df["primary_label"].astype(str).values
        if "primary_label" in df.columns
        else np.zeros(len(df))
    )
    if "author" in df.columns:
        groups = df["author"].astype(str).values
    else:
        groups = df["resolved_path"].map(lambda x: Path(x).stem).astype(str).values
    df["fold"] = -1
    n_splits = min(config.n_folds, max(2, len(df)))
    if StratifiedGroupKFold is not None and len(np.unique(y)) > 1:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=config.seed
        )
        split_iter = splitter.split(df, y, groups)
    elif len(np.unique(y)) > 1:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=config.seed
        )
        split_iter = splitter.split(df, y)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(df, y, groups)
    for fold, (_, val_idx) in enumerate(split_iter):
        df.loc[val_idx, "fold"] = fold
    log("Fold primary-label diversity:")
    print(
        (
            df.groupby("fold")["primary_label"].nunique()
            if "primary_label" in df.columns
            else df["fold"].value_counts()
        ),
        flush=True,
    )
    return df


class BirdCLEFTrainingPipeline:
    def __init__(self, config: BirdCLEFTrainingConfig):
        self.config = config
        self.output_dir = config.output_path
        self.paths: Optional[DataPaths] = None
        self.train_df_raw = pd.DataFrame()
        self.taxonomy_df = pd.DataFrame()
        self.sample_submission_df = pd.DataFrame()
        self.soundscape_files: list[Path] = []
        self.plotter = ReportPlotter(
            self.output_dir,
            plots_dir=config.plots_path,
            logs_dir=config.logs_path,
            pseudo_labels_dir=config.pseudo_labels_path,
        )

    @property
    def resume_dir(self) -> Optional[Path]:
        path = self.config.resume_path
        if path is not None and path.exists():
            return path
        return None

    def _import_previous_logs(self) -> None:
        previous_logs_dir = self.config.resume_logs_path
        if previous_logs_dir is None:
            return
        previous_global = previous_logs_dir / "train_log.csv"
        current_global = self.config.logs_path / "train_log.csv"
        if previous_global.exists() and not current_global.exists():
            current_global.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous_global, current_global)
            log(f"Imported previous global log: {previous_global} -> {current_global}")

    def setup(self) -> None:
        log("Pipeline setup started")
        log(f"output_dir={self.output_dir}")
        if self.resume_dir is not None:
            log(f"resume_dir={self.resume_dir}")
        else:
            log("resume_dir=None")
        log(
            "training config: "
            f"debug={self.config.debug}, model_key={self.config.model_key}, "
            f"batch_size={self.config.batch_size}, grad_accum={self.config.gradient_accumulation_steps}, "
            f"teacher_epochs={self.config.epochs_teacher}, student_epochs={self.config.epochs_student}, "
            f"noisy_iters={self.config.num_noisy_student_iterations}"
        )
        seed_everything(self.config.seed, self.config.deterministic)
        print_environment()
        ensure_output_dirs(
            self.output_dir,
            logs_dir=self.config.logs_path,
            plots_dir=self.config.plots_path,
            pseudo_labels_dir=self.config.pseudo_labels_path,
        )
        self._import_previous_logs()
        self.config.save(self.output_dir / "config.json")
        save_json(CLASS_ORDER, self.output_dir / "class_order.json")
        self.paths = discover_paths(self.config)
        self.train_df_raw = read_csv_optional(self.paths.train_csv, "train.csv")
        self.taxonomy_df = read_csv_optional(self.paths.taxonomy_csv, "taxonomy.csv")
        self.sample_submission_df = read_csv_optional(
            self.paths.sample_submission_csv, "sample_submission.csv"
        )
        validate_class_order(
            self.sample_submission_df, strict=self.config.strict_class_order
        )
        if self.config.debug and len(self.train_df_raw):
            self.train_df_raw = self.train_df_raw.head(
                self.config.debug_num_train_files
            ).copy()
            log(f"Debug mode: using {len(self.train_df_raw)} training rows.")
        self.soundscape_files = []
        if self.paths.train_soundscapes is not None:
            self.soundscape_files = sorted(
                Path(self.paths.train_soundscapes).rglob("*.ogg")
            )
            if self.config.debug:
                self.soundscape_files = self.soundscape_files[
                    : self.config.debug_num_soundscapes
                ]
        log(f"train soundscapes: {len(self.soundscape_files)} files")
        self.eda_sanity_checks()

    def eda_sanity_checks(self) -> None:
        df = self.train_df_raw
        log("EDA sanity checks")
        log(f"train.csv rows: {len(df)}")
        log(f"target classes: {len(CLASS_ORDER)}")
        if len(df) and "primary_label" in df.columns:
            counts = df["primary_label"].value_counts()
            log("Top class counts:")
            print(counts.head(20), flush=True)
            log(f"classes with <= 2 samples: {(counts <= 2).sum()}")
        if len(df) and "secondary_labels" in df.columns:
            parsed = df["secondary_labels"].map(parse_secondary_labels)
            log(f"files with secondary labels: {(parsed.map(len) > 0).sum()}")
        if self.paths and self.paths.train_audio is not None:
            log(
                f"train_audio files: {len(list(Path(self.paths.train_audio).rglob('*.ogg')))}"
            )
        log(f"train_soundscape files: {len(self.soundscape_files)}")
        self.plotter.plot_class_distribution(df)

    def build_stage_loaders(
        self,
        focal_df: pd.DataFrame,
        pseudo_path: Optional[Path] = None,
        student: bool = False,
    ):
        log(f"Building dataloaders: student={student}, pseudo_path={pseudo_path}")
        folded = make_folds(focal_df, self.config)
        train_part = folded[folded["fold"] != self.config.fold].reset_index(drop=True)
        valid_part = folded[folded["fold"] == self.config.fold].reset_index(drop=True)
        if valid_part.empty:
            valid_part = train_part.head(max(1, min(32, len(train_part)))).copy()
        log(
            f"Fold split: train_rows={len(train_part)}, valid_rows={len(valid_part)}, fold={self.config.fold}"
        )

        focal_train = LabeledFocalDataset(
            train_part,
            self.config,
            mode="train",
            augment=True,
            strength=(
                self.config.student_aug_strength
                if student
                else self.config.teacher_aug_strength
            ),
            padding_mode="zero_random" if student else "zero_left",
        )
        valid_ds = LabeledFocalDataset(
            valid_part,
            self.config,
            mode="valid",
            augment=False,
            strength="none",
            padding_mode="zero_center",
        )
        if pseudo_path is not None:
            pseudo_ds = PseudoLabeledSoundscapeDataset(
                pseudo_path,
                self.config,
                augment=True,
                strength=self.config.student_aug_strength,
            )
            train_ds = MixedNoisyStudentDataset(focal_train, pseudo_ds, self.config)
        else:
            train_ds = focal_train
        pos_weight = make_pos_weight(train_part, self.config)
        return (
            make_loader(train_ds, self.config, shuffle=True),
            make_loader(valid_ds, self.config, shuffle=False),
            pos_weight,
        )

    def _train_stage(
        self,
        stage_name: str,
        stage_dir: Path,
        focal_df: pd.DataFrame,
        epochs: int,
        pseudo_path: Optional[Path] = None,
        student: bool = False,
        init_checkpoint: Optional[Path] = None,
        resume_stage_dir: Optional[Path] = None,
    ) -> Path:
        log(f"Stage {stage_name}: preparing model and loaders")
        log(
            f"Stage {stage_name}: stage_dir={stage_dir}, resume_stage_dir={resume_stage_dir}, target_epochs={epochs}, student={student}"
        )
        drop_path_rate = self.config.student_drop_path_rate if student else 0.0
        model, model_config = build_birdclef_model(
            self.config, drop_path_rate=drop_path_rate
        )
        resume_last = find_latest_training_checkpoint(resume_stage_dir)
        current_last = find_latest_training_checkpoint(stage_dir)
        if init_checkpoint is not None and not (
            self.config.auto_resume
            and (current_last is not None or resume_last is not None)
        ):
            log(f"Initializing {stage_name} from {init_checkpoint}")
            load_model_weights(model, init_checkpoint)
        train_loader, valid_loader, pos_weight = self.build_stage_loaders(
            focal_df, pseudo_path=pseudo_path, student=student
        )
        trainer = StageTrainer(
            model=model,
            model_config=model_config,
            config=self.config,
            stage_dir=stage_dir,
            stage_name=stage_name,
            pos_weight=pos_weight,
            spec_aug_strength=(
                self.config.student_aug_strength
                if student
                else self.config.teacher_aug_strength
            ),
            resume_stage_dir=resume_stage_dir,
        )
        best_path = trainer.fit(train_loader, valid_loader, total_epochs=epochs)
        log(f"Stage {stage_name}: training returned checkpoint {best_path}")
        self.plotter.render_all(self.train_df_raw)
        return best_path

    def run(self) -> Dict[str, Any]:
        self.setup()
        if self.config.plot_only:
            log("plot_only=True; regenerating plots without training")
            self.plotter.render_all(self.train_df_raw)
            return {"output_dir": str(self.output_dir), "plot_only": True}

        if (
            self.paths is None
            or self.paths.train_audio is None
            or self.train_df_raw.empty
        ):
            log("No official training metadata/audio found. Training is skipped.")
            return {}
        focal_df = build_focal_dataframe(
            self.train_df_raw, Path(self.paths.train_audio)
        )
        if focal_df.empty:
            log("No focal training files were resolved. Training is skipped.")
            return {}
        log(f"Resolved focal training rows: {len(focal_df)}")

        teacher_epochs = effective_epochs(
            self.config.debug, self.config.debug_epochs, self.config.epochs_teacher
        )
        teacher_dir = self.config.stage0_path
        log(f"Starting teacher stage: target_epochs={teacher_epochs}")
        teacher_ckpt = self._train_stage(
            "stage0_teacher",
            teacher_dir,
            focal_df,
            epochs=teacher_epochs,
            pseudo_path=None,
            student=False,
            resume_stage_dir=self.config.resume_stage0_path,
        )

        current_ckpt = teacher_ckpt
        pseudo_paths = []
        n_iters = effective_iterations(
            self.config.debug,
            self.config.debug_num_noisy_student_iterations,
            self.config.num_noisy_student_iterations,
        )
        log(f"Noisy student iterations to run: {n_iters}")
        for iteration in range(1, n_iters + 1):
            if not self.soundscape_files:
                log("No train_soundscapes found; stopping after supervised teacher.")
                break

            log(f"Iteration {iteration}: pseudo-label preparation started")
            pseudo_base = self.config.pseudo_labels_path / f"iter_{iteration}"
            pseudo_path = pseudo_base.with_suffix(".parquet")
            existing_pseudo = (
                pseudo_path if pseudo_path.exists() else pseudo_base.with_suffix(".csv")
            )
            resume_pseudo_dir = self.config.resume_pseudo_labels_path
            if not existing_pseudo.exists() and resume_pseudo_dir is not None:
                resume_base = resume_pseudo_dir / f"iter_{iteration}"
                resume_pseudo = resume_base.with_suffix(".parquet")
                existing_pseudo = (
                    resume_pseudo
                    if resume_pseudo.exists()
                    else resume_base.with_suffix(".csv")
                )
            if self.config.reuse_pseudo and existing_pseudo.exists():
                pseudo_path = existing_pseudo
                log(f"Iteration {iteration}: reusing pseudo labels {pseudo_path}")
            else:
                model, _ = build_birdclef_model(self.config)
                old_power = self.config.pseudo_power
                self.config.pseudo_power = pseudo_power_for_iteration(
                    iteration, self.config
                )
                log(
                    f"Iteration {iteration}: generating pseudo labels with power={self.config.pseudo_power}"
                )
                pseudo_path = PseudoLabeler(self.config).generate(
                    model,
                    self.soundscape_files,
                    pseudo_path,
                    checkpoint_path=current_ckpt,
                )
                self.config.pseudo_power = old_power
                self.plotter.plot_pseudo_stats(pseudo_path.with_suffix(".stats.json"))
            pseudo_paths.append(str(pseudo_path))

            student_epochs = effective_epochs(
                self.config.debug, self.config.debug_epochs, self.config.epochs_student
            )
            student_dir = self.config.student_stage_path(iteration)
            log(
                f"Starting student iteration {iteration}: target_epochs={student_epochs}"
            )
            current_ckpt = self._train_stage(
                f"stage{iteration}_student_iter{iteration}",
                student_dir,
                focal_df,
                epochs=student_epochs,
                pseudo_path=pseudo_path,
                student=True,
                init_checkpoint=current_ckpt,
                resume_stage_dir=self.config.resume_student_stage_path(iteration),
            )

        self.plotter.render_all(self.train_df_raw)
        log(
            f"Pipeline finished. final_checkpoint={current_ckpt}, output_dir={self.output_dir}"
        )
        return {
            "final_checkpoint": str(current_ckpt),
            "pseudo_paths": pseudo_paths,
            "output_dir": str(self.output_dir),
        }
