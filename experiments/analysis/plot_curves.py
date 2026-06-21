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
from pathlib import Path
import pandas as pd

from bioacoustic.visualization import plot_metric_curve, plot_bar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_csv', required=True)
    parser.add_argument('--out_dir', default='reports/figures')
    args = parser.parse_args()
    df = pd.read_csv(args.history_csv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    metric = 'macro_auc' if 'macro_auc' in df.columns else df.columns[-1]
    plot_metric_curve(df, x='epoch', y=metric, out_path=out / 'experiment_metric_curve.png', group='fold' if 'fold' in df.columns else None, title=f'{metric} curve')
    if 'valid_loss' in df.columns:
        plot_metric_curve(df, x='epoch', y='valid_loss', out_path=out / 'validation_loss_curve.png', group='fold' if 'fold' in df.columns else None, title='Validation loss curve')
    print(f'Saved plots to {out}')


if __name__ == '__main__':
    main()
