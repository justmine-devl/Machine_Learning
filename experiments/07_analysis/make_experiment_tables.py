from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge experiment metrics into report tables")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of metrics_summary.csv files")
    parser.add_argument("--names", nargs="+", required=True, help="Display names for the experiments")
    parser.add_argument("--output", default="reports/tables/ablation_table.csv")
    args = parser.parse_args()

    if len(args.inputs) != len(args.names):
        raise ValueError("--inputs and --names must have the same length")

    rows = []
    for name, path in zip(args.names, args.inputs):
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict()
        row["method"] = name
        rows.append(row)

    out = pd.DataFrame(rows)
    columns = ["method"] + [c for c in ["macro_auc", "macro_map", "macro_precision", "macro_recall", "macro_f1", "threshold"] if c in out.columns]
    out = out[columns]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"Saved table to {output}")


if __name__ == "__main__":
    main()
