"""
Filesystem anchors so the app works regardless of the current directory.

Weights, the ``PnLCalib`` checkout and the ``results`` output folder all live
at the repository root (the parent of the ``var_offside`` package).  Resolving
against that root means the CLI, GUI and API behave the same whether launched
from the repo root, a venv, or an installed console script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# .../VAR-Offside/var_offside/config/paths.py -> repo root is parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def resolve_weight(name: Optional[str]) -> Optional[str]:
    """
    Return an absolute path for a weight/asset if it exists at the repo root.

    Non-file references (e.g. torchvision's ``"resnet50"``), already-absolute
    paths, and names not present at the root are returned unchanged, so this is
    always safe to apply.
    """
    if not name:
        return name
    p = Path(name)
    if p.is_absolute():
        return name
    candidate = REPO_ROOT / name
    if candidate.exists():
        return str(candidate)
    return name


def results_dir() -> Path:
    """The repo-root ``results/`` folder, created on demand."""
    d = REPO_ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d
