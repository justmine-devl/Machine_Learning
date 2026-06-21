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
    fallback = current.parents[2] if len(current.parents) >= 3 else current.parent
    src = fallback / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return fallback


PROJECT_ROOT = add_project_src_to_path()


import argparse
from pathlib import Path
import json
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_dir', default='outputs')
    parser.add_argument('--out_dir', default='reports/tables')
    args = parser.parse_args()
    rows = []
    for path in Path(args.metrics_dir).rglob('*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if any(k in data for k in ['macro_auc', 'best_macro_auc', 'f1_macro']):
            row = {'source_file': str(path)}
            row.update(data)
            rows.append(row)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'experiment_summary.csv', index=False)
    print(f'Saved summary table to {out / "experiment_summary.csv"}')


if __name__ == '__main__':
    main()
