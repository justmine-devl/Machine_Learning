from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None

from .checkpoints import find_latest_training_checkpoint, load_model_weights
from .class_order import CLASS_ORDER
from .config import TrainingConfig
from .datasets import (
    LabeledFocalDataset,
    MixedNoisyStudentDataset,
    PseudoLabeledSoundscapeDataset,
    build_focal_dataframe,
    parse_secondary_labels,
)
from .losses_metrics import make_pos_weight
from .modeling import build_model
from .paths import DataPaths, discover_paths
from .plots import ReportPlotter
from .pseudo import PseudoLabeler, pseudo_power_for_iteration
from .trainer import StageTrainer, make_loader
from .utils import effective_epochs, effective_iterations, ensure_output_dirs, log, print_environment, seed_everything, write_json


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
        log("Warning: sample_submission has the same classes but different order. Using inference order for checkpoint compatibility.")
        return
    missing = sorted(set(CLASS_ORDER) - set(sample_cols))
    extra = sorted(set(sample_cols) - set(CLASS_ORDER))
    msg = f"Class set mismatch. missing_in_sample={missing[:10]}, extra_in_sample={extra[:10]}"
    if strict:
        raise ValueError(msg)
    log(f"Warning: {msg}")
    log("Continuing with inference class order.")


def make_folds(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if df.empty:
        return df
    y = df["primary_label"].astype(str).values if "primary_label" in df.columns else np.zeros(len(df))
    if "author" in df.columns:
        groups = df["author"].astype(str).values
    else:
        groups = df["resolved_path"].map(lambda x: Path(x).stem).astype(str).values
    df["fold"] = -1
    n_splits = min(config.n_folds, max(2, len(df)))
    if StratifiedGroupKFold is not None and len(np.unique(y)) > 1:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        split_iter = splitter.split(df, y, groups)
    elif len(np.unique(y)) > 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        split_iter = splitter.split(df, y)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(df, y, groups)
    for fold, (_, val_idx) in enumerate(split_iter):
        df.loc[val_idx, "fold"] = fold
    log("Fold primary-label diversity:")
    print(df.groupby("fold")["primary_label"].nunique() if "primary_label" in df.columns else df["fold"].value_counts(), flush=True)
    return df


class BirdCLEFTrainingPipeline:
    def __init__(self, config: TrainingConfig):
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
        write_json(self.output_dir / "class_order.json", CLASS_ORDER)
        self.paths = discover_paths(self.config)
        self.train_df_raw = read_csv_optional(self.paths.train_csv, "train.csv")
        self.taxonomy_df = read_csv_optional(self.paths.taxonomy_csv, "taxonomy.csv")
        self.sample_submission_df = read_csv_optional(self.paths.sample_submission_csv, "sample_submission.csv")
        validate_class_order(self.sample_submission_df, strict=self.config.strict_class_order)
        if self.config.debug and len(self.train_df_raw):
            self.train_df_raw = self.train_df_raw.head(self.config.debug_num_train_files).copy()
            log(f"Debug mode: using {len(self.train_df_raw)} training rows.")
        self.soundscape_files = []
        if self.paths.train_soundscapes is not None:
            self.soundscape_files = sorted(Path(self.paths.train_soundscapes).rglob("*.ogg"))
            if self.config.debug:
                self.soundscape_files = self.soundscape_files[: self.config.debug_num_soundscapes]
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
            log(f"train_audio files: {len(list(Path(self.paths.train_audio).rglob('*.ogg')))}")
        log(f"train_soundscape files: {len(self.soundscape_files)}")
        self.plotter.plot_class_distribution(df)

    def build_stage_loaders(self, focal_df: pd.DataFrame, pseudo_path: Optional[Path] = None, student: bool = False):
        log(f"Building dataloaders: student={student}, pseudo_path={pseudo_path}")
        folded = make_folds(focal_df, self.config)
        train_part = folded[folded["fold"] != self.config.fold].reset_index(drop=True)
        valid_part = folded[folded["fold"] == self.config.fold].reset_index(drop=True)
        if valid_part.empty:
            valid_part = train_part.head(max(1, min(32, len(train_part)))).copy()
        log(f"Fold split: train_rows={len(train_part)}, valid_rows={len(valid_part)}, fold={self.config.fold}")

        focal_train = LabeledFocalDataset(
            train_part,
            self.config,
            mode="train",
            augment=True,
            strength=self.config.student_aug_strength if student else self.config.teacher_aug_strength,
        )
        valid_ds = LabeledFocalDataset(valid_part, self.config, mode="valid", augment=False, strength="none")
        if pseudo_path is not None:
            pseudo_ds = PseudoLabeledSoundscapeDataset(pseudo_path, self.config, augment=True, strength=self.config.student_aug_strength)
            train_ds = MixedNoisyStudentDataset(focal_train, pseudo_ds, self.config)
        else:
            train_ds = focal_train
        pos_weight = make_pos_weight(train_part, self.config)
        return make_loader(train_ds, self.config, shuffle=True), make_loader(valid_ds, self.config, shuffle=False), pos_weight

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
        log(f"Stage {stage_name}: stage_dir={stage_dir}, resume_stage_dir={resume_stage_dir}, target_epochs={epochs}, student={student}")
        model, model_config = build_model(self.config)
        resume_last = find_latest_training_checkpoint(resume_stage_dir)
        current_last = find_latest_training_checkpoint(stage_dir)
        if init_checkpoint is not None and not (
            self.config.auto_resume and (current_last is not None or resume_last is not None)
        ):
            log(f"Initializing {stage_name} from {init_checkpoint}")
            load_model_weights(model, init_checkpoint)
        train_loader, valid_loader, pos_weight = self.build_stage_loaders(focal_df, pseudo_path=pseudo_path, student=student)
        trainer = StageTrainer(
            model=model,
            model_config=model_config,
            config=self.config,
            stage_dir=stage_dir,
            stage_name=stage_name,
            pos_weight=pos_weight,
            spec_aug_strength=self.config.student_aug_strength if student else self.config.teacher_aug_strength,
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

        if self.paths is None or self.paths.train_audio is None or self.train_df_raw.empty:
            log("No official training metadata/audio found. Training is skipped.")
            return {}
        focal_df = build_focal_dataframe(self.train_df_raw, Path(self.paths.train_audio))
        if focal_df.empty:
            log("No focal training files were resolved. Training is skipped.")
            return {}
        log(f"Resolved focal training rows: {len(focal_df)}")

        teacher_epochs = effective_epochs(self.config.debug, self.config.debug_epochs, self.config.epochs_teacher)
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
            existing_pseudo = pseudo_path if pseudo_path.exists() else pseudo_base.with_suffix(".csv")
            resume_pseudo_dir = self.config.resume_pseudo_labels_path
            if not existing_pseudo.exists() and resume_pseudo_dir is not None:
                resume_base = resume_pseudo_dir / f"iter_{iteration}"
                resume_pseudo = resume_base.with_suffix(".parquet")
                existing_pseudo = resume_pseudo if resume_pseudo.exists() else resume_base.with_suffix(".csv")
            if self.config.reuse_pseudo and existing_pseudo.exists():
                pseudo_path = existing_pseudo
                log(f"Iteration {iteration}: reusing pseudo labels {pseudo_path}")
            else:
                model, _ = build_model(self.config)
                old_power = self.config.pseudo_power
                self.config.pseudo_power = pseudo_power_for_iteration(iteration, self.config)
                log(f"Iteration {iteration}: generating pseudo labels with power={self.config.pseudo_power}")
                pseudo_path = PseudoLabeler(self.config).generate(model, self.soundscape_files, pseudo_path, checkpoint_path=current_ckpt)
                self.config.pseudo_power = old_power
                self.plotter.plot_pseudo_stats(pseudo_path.with_suffix(".stats.json"))
            pseudo_paths.append(str(pseudo_path))

            student_epochs = effective_epochs(self.config.debug, self.config.debug_epochs, self.config.epochs_student)
            student_dir = self.config.student_stage_path(iteration)
            log(f"Starting student iteration {iteration}: target_epochs={student_epochs}")
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
        log(f"Pipeline finished. final_checkpoint={current_ckpt}, output_dir={self.output_dir}")
        return {"final_checkpoint": str(current_ckpt), "pseudo_paths": pseudo_paths, "output_dir": str(self.output_dir)}


