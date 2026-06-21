from __future__ import annotations

# Noisy Student is implemented as pseudo-label training with stronger augmentation/noise.
# The dataset still uses random crops; the model uses higher dropout and label smoothing.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.common.config import get_arg_parser, load_config

# Reuse the pseudo-label training implementation to avoid duplicated code.
from experiments.common.utils import save_json


def main() -> None:
    args = get_arg_parser("Train Noisy Student model").parse_args()
    cfg = load_config(args.config)
    Path(cfg.get("output_dir", "outputs/noisy_student")).mkdir(parents=True, exist_ok=True)
    save_json({
        "note": "Noisy Student uses the same training pipeline as pseudo-label training, but with stronger augmentation, higher dropout, and larger pseudo sampling ratio.",
        "recommended_command": "python experiments/03_pseudo_labeling/train_with_pseudo_labels.py --config experiments/04_noisy_student/config_noisy_student.yaml"
    }, Path(cfg.get("output_dir", "outputs/noisy_student")) / "noisy_student_notes.json")
    print("Run this command to train the Noisy Student model:")
    print("python experiments/03_pseudo_labeling/train_with_pseudo_labels.py --config", args.config)


if __name__ == "__main__":
    main()
