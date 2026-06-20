from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import argparse
import hashlib
import json


@dataclass
class TrainingConfig:
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
    soundscape_stride_sec: int = 5  # ASSUMPTION: not publicly available; tune this
    model_key: str = "model_config_6"
    pretrained_backbone: bool = False
    strict_class_order: bool = False

    # Training
    batch_size: int = 2  # Memory-safe Kaggle default for 20s B4 chunks.
    gradient_accumulation_steps: int = 8
    num_workers: int = 0
    epochs_teacher: int = 8  # ASSUMPTION: not publicly available; tune this
    epochs_student: int = 5  # ASSUMPTION: not publicly available; tune this
    num_noisy_student_iterations: int = 2
    learning_rate: float = 3e-4  # ASSUMPTION: not publicly available; tune this
    head_learning_rate: float = 1e-3  # ASSUMPTION: not publicly available; tune this
    min_learning_rate: float = 1e-6
    restart_epochs: int = 5
    weight_decay: float = 1e-4  # ASSUMPTION: not publicly available; tune this
    gradient_clip_norm: float = 1.0  # ASSUMPTION: not publicly available; tune this
    mixed_precision: bool = True
    train_backbone: bool = True
    save_every_epochs: int = 0
    save_best_training_checkpoint: bool = False
    log_every_steps: int = 50

    # Validation
    n_folds: int = 5
    fold: int = 0
    val_threshold: float = 0.5  # ASSUMPTION: not publicly available; tune this

    # Loss
    loss_type: str = "cross_entropy"
    focal_gamma: float = 2.0  # ASSUMPTION: not publicly available; tune this
    focal_alpha: Optional[float] = None  # ASSUMPTION: not publicly available; tune this
    label_smoothing: float = 0.0
    use_pos_weight: bool = False

    # Augmentation
    random_gain_db: float = 6.0  # ASSUMPTION: not publicly available; tune this
    gaussian_noise_std: float = 0.003  # ASSUMPTION: not publicly available; tune this
    time_shift_sec: float = 1.0  # ASSUMPTION: not publicly available; tune this
    time_mask_param: int = 48  # ASSUMPTION: not publicly available; tune this
    freq_mask_param: int = 24  # ASSUMPTION: not publicly available; tune this
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
    pseudo_power_iter2: float = 1.5  # ASSUMPTION: not publicly available; tune this
    pseudo_power_iter3: float = 2.0  # ASSUMPTION: not publicly available; tune this
    min_confidence_for_retention: float = 0.0

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
        return Path(self.pseudo_labels_dir) if self.pseudo_labels_dir else self.output_path / "pseudo_labels"

    @property
    def stage0_path(self) -> Path:
        return Path(self.stage0_dir) if self.stage0_dir else self.output_path / "stage0_teacher"

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
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstructed BirdCLEF 2025 Noisy Student training")

    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-audio-dir", default=None)
    parser.add_argument("--train-soundscapes-dir", default=None)
    parser.add_argument("--test-soundscapes-dir", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--taxonomy-csv", default=None)
    parser.add_argument("--sample-submission-csv", default=None)
    parser.add_argument("--recording-location-txt", default=None)
    parser.add_argument("--extra-data-root", default=None)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Read-only previous output directory, e.g. /kaggle/input/<saved-session>/outputs. New files still go to --output-dir.",
    )
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--pseudo-labels-dir", default=None)
    parser.add_argument("--stage0-dir", default=None)
    parser.add_argument("--student-stage-template", default="stage{iteration}_student_iter{iteration}")
    parser.add_argument("--resume-logs-dir", default=None)
    parser.add_argument("--resume-pseudo-labels-dir", default=None)
    parser.add_argument("--resume-stage0-dir", default=None)
    parser.add_argument("--resume-student-stage-template", default="stage{iteration}_student_iter{iteration}")
    parser.add_argument("--auto-discover-paths", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-pseudo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--debug-num-train-files", type=int, default=200)
    parser.add_argument("--debug-num-soundscapes", type=int, default=20)
    parser.add_argument("--debug-epochs", type=int, default=1)
    parser.add_argument("--debug-num-noisy-student-iterations", type=int, default=1)

    parser.add_argument("--model-key", default="model_config_6")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs-teacher", type=int, default=8)
    parser.add_argument("--epochs-student", type=int, default=5)
    parser.add_argument("--num-noisy-student-iterations", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--restart-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=0,
        help="Optional periodic full checkpoints. 0 disables epoch_N.pth files to keep Kaggle outputs small.",
    )
    parser.add_argument(
        "--save-best-training-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save best.pth full training checkpoint. Disabled by default; best_for_inference.pt is still saved.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=50,
        help="Print Kaggle-friendly progress every N train/validation/pseudo batches. 0 disables batch heartbeat logs.",
    )

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--val-threshold", type=float, default=0.5)
    parser.add_argument("--loss-type", choices=["cross_entropy", "bce", "focal_bce"], default="cross_entropy")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-pos-weight", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--train-duration-sec", type=int, default=20)
    parser.add_argument("--label-frame-sec", type=int, default=5)
    parser.add_argument("--soundscape-stride-sec", type=int, default=5)

    parser.add_argument("--mixup-focal-pseudo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pseudo-mixup-lambda", type=float, default=0.5)
    parser.add_argument("--student-drop-path-rate", type=float, default=0.15)
    parser.add_argument("--pseudo-threshold", type=float, default=0.0)
    parser.add_argument("--pseudo-top-k", type=int, default=None)
    parser.add_argument("--pseudo-power", type=float, default=1.0)
    parser.add_argument("--pseudo-power-iter1", type=float, default=1.0)
    parser.add_argument("--pseudo-power-iter2", type=float, default=1.5)
    parser.add_argument("--pseudo-power-iter3", type=float, default=2.0)
    parser.add_argument("--min-confidence-for-retention", type=float, default=0.0)

    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    values = vars(args).copy()
    config = TrainingConfig(**values)
    if config.restart_epochs < 1:
        raise ValueError("--restart-epochs must be at least 1")
    if not 0.0 <= config.pseudo_mixup_lambda <= 1.0:
        raise ValueError("--pseudo-mixup-lambda must be in [0, 1]")
    if not 0.0 <= config.student_drop_path_rate < 1.0:
        raise ValueError("--student-drop-path-rate must be in [0, 1)")
    return config
