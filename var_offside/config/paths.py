"""
Where VAR-Offside looks for model weights, third-party code and output files.

Everything is anchored to one *project home* so the CLI, the web server and the
tests behave the same whatever the current working directory is.

Project home
    ``$VAR_OFFSIDE_HOME`` if set, otherwise the source checkout (the folder
    that holds ``pyproject.toml`` next to the ``var_offside`` package), and
    otherwise ``~/.var_offside`` for a non-editable install.

Layout under the project home::

    weights/        model files       (override with $VAR_OFFSIDE_WEIGHTS_DIR)
    third_party/    upstream clones   (PnLCalib, sam-3d-body, Fast-SAM-3D-Body)
    results/        exports           (override with $VAR_OFFSIDE_RESULTS_DIR)

Files placed directly in the project home (the original development layout)
are still found, so older checkouts keep working.

See ``docs/getting-started/models.md`` for what goes where.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_PACKAGE_PARENT = Path(__file__).resolve().parents[2]


def project_home() -> Path:
    """The folder that holds ``weights/``, ``third_party/`` and ``results/``."""
    env = os.environ.get("VAR_OFFSIDE_HOME")
    if env:
        return Path(env).expanduser().resolve()
    if (_PACKAGE_PARENT / "pyproject.toml").is_file():
        return _PACKAGE_PARENT
    return Path.home() / ".var_offside"


# Kept for code that imported the old constant. Prefer ``project_home()``.
REPO_ROOT: Path = project_home()


def weights_dir() -> Path:
    """Folder for model weights."""
    env = os.environ.get("VAR_OFFSIDE_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return project_home() / "weights"


def third_party_dir(name: str, *legacy: str) -> Path:
    """
    Location of an upstream clone such as ``PnLCalib``.

    Looks in ``third_party/<name>``, then ``<home>/<name>``, then each
    ``legacy`` path relative to the home. Returns the ``third_party`` location
    when none exist, so error messages point at the recommended place.
    """
    home = project_home()
    preferred = home / "third_party" / name
    for candidate in (preferred, home / name, *(home / p for p in legacy)):
        if candidate.is_dir():
            return candidate
    return preferred


def resolve_weight(name: Optional[str]) -> Optional[str]:
    """
    Return an absolute path for a weight or asset if it can be found.

    Looks in ``weights/``, ``third_party/`` and the project home, in that
    order. Non-file references (e.g. torchvision's ``"resnet50"``), absolute
    paths and names that are not found are returned unchanged, so this is
    always safe to apply. Ultralytics then downloads its own official weights
    (such as ``yolo11x-pose.pt``) by name.
    """
    if not name:
        return name
    if Path(name).is_absolute():
        return name
    home = project_home()
    for base in (weights_dir(), home / "third_party", home):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return name


def first_existing(*names: str) -> str:
    """Resolve each name in turn and return the first that exists on disk.

    Falls back to the first name under ``weights/``, so a missing file gives
    an error message that points at the recommended location.
    """
    for name in names:
        resolved = resolve_weight(name)
        if resolved and Path(resolved).exists():
            return resolved
    return str(weights_dir() / names[0])


def results_dir() -> Path:
    """Folder for exports, created on demand."""
    env = os.environ.get("VAR_OFFSIDE_RESULTS_DIR")
    d = Path(env).expanduser().resolve() if env else project_home() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d
