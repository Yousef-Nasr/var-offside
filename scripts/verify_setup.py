"""
Environment & asset checker for VAR-Offside.

Run this after installing to confirm the machine is ready:

    python scripts/verify_setup.py

It checks the Python version, required weight files, PyTorch/CUDA, and does a
tiny offscreen pyrender smoke test (the 3D replay needs a working OpenGL
context).  Exit code is non-zero if anything critical is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

# (relative path, critical?, note)
REQUIRED_WEIGHTS = [
    ("yolo11x-pose.pt", True, "player pose detection"),
    ("best_tuned.pt", True, "football object detector"),
    ("model_sam3_3d.ckpt", False, "SAM 3D Body (needed for 3D replay)"),
    ("yolo11l.pt", False, "human detector for SAM 3D Body"),
    ("mobileclip2_b.ts", False, "MobileCLIP2 for SAM 3D Body"),
    ("PnLCalib/weights/SV_kp", False, "PnL keypoints (needed for calibrated 3D)"),
    ("PnLCalib/weights/SV_lines", False, "PnL lines (needed for calibrated 3D)"),
]

REQUIRED_DIRS = [
    ("sam3_body3d", False, "SAM 3D Body wrapper"),
    ("PnLCalib", False, "camera calibration"),
]


def _p(status: str, msg: str):
    print(f"{status} {msg}")


def check_python() -> bool:
    v = sys.version_info
    if v < (3, 10):
        _p(FAIL, f"Python {v.major}.{v.minor} - need 3.10+ (PEP 604 unions used).")
        return False
    _p(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_files() -> bool:
    critical_ok = True
    for rel, critical, note in REQUIRED_WEIGHTS:
        path = REPO_ROOT / rel
        if path.exists():
            _p(OK, f"{rel}  ({note})")
        elif critical:
            _p(FAIL, f"{rel} MISSING - {note}")
            critical_ok = False
        else:
            _p(WARN, f"{rel} missing - {note}")
    for rel, critical, note in REQUIRED_DIRS:
        path = REPO_ROOT / rel
        if path.is_dir():
            _p(OK, f"{rel}/  ({note})")
        elif critical:
            _p(FAIL, f"{rel}/ MISSING - {note}")
            critical_ok = False
        else:
            _p(WARN, f"{rel}/ missing - {note}")
    return critical_ok


def check_imports() -> bool:
    ok = True
    for mod in ("numpy", "cv2", "torch", "ultralytics", "trimesh", "pyrender", "PyQt5"):
        try:
            __import__(mod)
            _p(OK, f"import {mod}")
        except Exception as exc:  # noqa: BLE001
            _p(FAIL, f"import {mod} - {exc}")
            ok = False
    return ok


def check_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            _p(OK, f"CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            _p(WARN, "CUDA not available - 3D reconstruction will be very slow on CPU.")
    except Exception as exc:  # noqa: BLE001
        _p(WARN, f"Could not query CUDA: {exc}")


def check_pyrender_smoke():
    try:
        import numpy as np
        import trimesh
        import pyrender

        mesh = trimesh.creation.box(extents=(1, 1, 1))
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        pose = np.eye(4)
        pose[2, 3] = 3.0
        scene.add(cam, pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
        r = pyrender.OffscreenRenderer(64, 64)
        color, _ = r.render(scene)
        r.delete()
        if color is not None and color.shape == (64, 64, 3):
            _p(OK, "pyrender offscreen render (OpenGL context works)")
            return True
        _p(FAIL, "pyrender returned unexpected output")
        return False
    except Exception as exc:  # noqa: BLE001
        _p(FAIL, f"pyrender smoke test failed - {exc}")
        _p(WARN, "3D replay needs a working OpenGL driver. On a headless box set "
                 "PYOPENGL_PLATFORM=egl.")
        return False


def main() -> int:
    print("=" * 60)
    print("VAR-Offside setup verification")
    print("=" * 60)

    py_ok = check_python()
    print("-" * 60)
    imports_ok = check_imports()
    print("-" * 60)
    files_ok = check_files()
    print("-" * 60)
    check_cuda()
    print("-" * 60)
    check_pyrender_smoke()
    print("=" * 60)

    critical_ok = py_ok and imports_ok and files_ok
    if critical_ok:
        print("Core requirements satisfied. You can run the app.")
        return 0
    print("Some CRITICAL requirements are missing (see [FAIL] above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
