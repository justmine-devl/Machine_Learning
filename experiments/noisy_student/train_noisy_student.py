# Allow running the experiment script directly from any subfolder.
from __future__ import annotations
import sys
from pathlib import Path


def add_project_src_to_path() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        src = parent / "src"
        if src.exists():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return parent
    # Script is usually experiments/<method>/file.py, so repo root is two levels up.
    fallback = current.parents[2] if len(current.parents) >= 3 else current.parent
    src = fallback / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return fallback


PROJECT_ROOT = add_project_src_to_path()


import argparse
import subprocess
import sys
from pathlib import Path

from bioacoustic.utils import load_config, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--skip_pseudo_generation', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get('seed', 42)))
    here = Path(__file__).resolve().parent
    pseudo_script = here.parent / 'pseudo_labeling' / 'generate_pseudo_labels.py'
    train_script = here.parent / 'pseudo_labeling' / 'train_with_pseudo_labels.py'
    if not args.skip_pseudo_generation:
        subprocess.check_call([sys.executable, str(pseudo_script), '--config', args.config])
    subprocess.check_call([sys.executable, str(train_script), '--config', args.config])
    save_json({'status': 'completed', 'method': 'noisy_student'}, Path(cfg.get('output_dir', 'outputs/noisy_student')) / 'noisy_student_summary.json')


if __name__ == '__main__':
    main()
