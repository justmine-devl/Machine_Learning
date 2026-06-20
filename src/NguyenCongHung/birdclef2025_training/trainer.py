from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .audio import SpecAugment
from .checkpoints import checkpoint_epoch, find_latest_training_checkpoint, load_training_state, prune_epoch_checkpoints, read_checkpoint_epoch, save_checkpoint_bundle
from .config import TrainingConfig
from .datasets import collate_batch
from .losses_metrics import compute_loss, compute_metrics
from .modeling import forward_sed_logits, freeze_backbone_if_needed, make_optimizer
from .utils import log


def make_loader(dataset, config: TrainingConfig, shuffle: bool) -> DataLoader:
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
        config: TrainingConfig,
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
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and self.device.type == "cuda")
        self.spec_aug = None
        if spec_aug_strength != "none":
            self.spec_aug = SpecAugment(config.time_mask_param, config.freq_mask_param).to(self.device)

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
            log(f"{self.stage_name}: copied resume checkpoint {resume_checkpoint} -> {self.last_path}")
        for name in ["best_for_inference.pt", "best_for_inference.config.json", "train_log.csv"]:
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
            elif path == stage_log and self.resume_stage_dir is not None and (self.resume_stage_dir / "train_log.csv").exists():
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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, total_epochs))
        resume_last = self._existing_last_path()
        if self.config.auto_resume and not self.config.force_retrain and resume_last is not None:
            # Do not load scheduler state here: Kaggle continuation often increases total epochs.
            # Recreating the scheduler with the new total is safer than reusing an old T_max.
            last_epoch = load_training_state(resume_last, self.model, self.optimizer, None, self.scaler)
            log(f"{self.stage_name}: resumed from {resume_last} at epoch {last_epoch}")
            return last_epoch + 1
        if self.config.auto_resume and not self.config.force_retrain:
            if self.resume_stage_dir is not None and not self.resume_stage_dir.exists():
                log(f"{self.stage_name}: resume_stage_dir does not exist: {self.resume_stage_dir}")
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
        pbar = tqdm(loader, desc=f"{self.stage_name} train epoch {epoch}", leave=True, mininterval=5)
        for step, batch in enumerate(pbar, start=1):
            waves = batch["wave"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = forward_sed_logits(self.model, waves, self.config.frames_per_clip, spec_augment=self.spec_aug)
                loss = compute_loss(logits, targets, self.config, pos_weight=self.pos_weight)
                loss_to_backward = loss / accumulation_steps
            self.scaler.scale(loss_to_backward).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                if self.config.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if step == 1 or step % 10 == 0:
                pbar.set_postfix(loss=f"{loss_value:.4f}")
            if self.config.log_every_steps and (step == 1 or step % self.config.log_every_steps == 0 or step == total_batches):
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                log(f"{self.stage_name}: train epoch {epoch} batch {step}/{total_batches} loss={loss_value:.5f} mean_loss={mean_loss:.5f}")
        return {"train_loss": float(np.mean(losses)) if losses else float("nan")}

    @torch.no_grad()
    def validate_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        losses = []
        all_targets = []
        all_probs = []
        total_batches = len(loader)
        log(f"{self.stage_name}: validation started ({total_batches} batches)")
        for step, batch in enumerate(tqdm(loader, desc=f"{self.stage_name} valid", leave=True, mininterval=5), start=1):
            waves = batch["wave"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            logits = forward_sed_logits(self.model, waves, self.config.frames_per_clip)
            loss = compute_loss(logits, targets, self.config, pos_weight=self.pos_weight)
            losses.append(float(loss.detach().cpu()))
            all_targets.append(targets.cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            if self.config.log_every_steps and (step == 1 or step % self.config.log_every_steps == 0 or step == total_batches):
                mean_loss = float(np.mean(losses)) if losses else float("nan")
                log(f"{self.stage_name}: valid batch {step}/{total_batches} mean_loss={mean_loss:.5f}")
        if not all_targets:
            return {"valid_loss": float("nan"), "macro_auc": float("nan"), "macro_f1": float("nan"), "valid_auc_classes": 0}
        metrics = compute_metrics(np.concatenate(all_targets), np.concatenate(all_probs), threshold=self.config.val_threshold)
        metrics["valid_loss"] = float(np.mean(losses))
        return metrics

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, total_epochs: int) -> Path:
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
                    best_path = self.best_inference_path if self.best_inference_path.exists() else self._existing_best_inference_path()
                    log(f"{self.stage_name}: already completed {total_epochs} epochs from {existing_checkpoint}; using {best_path}")
                    return best_path
            except Exception:
                pass

        start_epoch = self._start_epoch(total_epochs)
        best_loss = self._best_loss_from_log()
        log(f"{self.stage_name}: starting loop at epoch {start_epoch}/{total_epochs}; current_best_loss={best_loss:.5f}")
        for epoch in range(start_epoch, total_epochs + 1):
            log(f"{self.stage_name}: epoch {epoch}/{total_epochs} started")
            train_metrics = self.train_one_epoch(train_loader, epoch)
            valid_metrics = self.validate_one_epoch(valid_loader)
            self.scheduler.step()
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


