"""Small helper so scripts work before `pip install -e .`.

Each script imports this module first. It adds the repository `src/` folder to
`sys.path` when the package has not been installed yet.
"""
from __future__ import annotations

import sys
from pathlib import Path


def add_project_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
