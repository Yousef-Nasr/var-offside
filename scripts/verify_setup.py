"""
Check that this machine is ready to run VAR-Offside, feature by feature.

    python scripts/verify_setup.py

It reads the same settings the app uses (built-in defaults plus config.yaml),
so a path it reports is the path the app will load. Exit code is non-zero
when something the core pipeline needs is missing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def line(status: str, msg: str) -> None:
    print(f"{status} {msg}")


def has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def file_status(path: str, label: str, needed: bool, hint: str) -> bool:
    if Path(path).exists():
        line(OK, f"{label}: {path}")
        return True
    line(FAIL if needed else WARN, f"{label} missing: {path}\n       {hint}")
    return False


def main() -> int:
    print("VAR-Offside setup check")
    print("=" * 60)

    v = sys.version_info
    py_ok = v >= (3, 10)
    line(OK if py_ok else FAIL, f"Python {v.major}.{v.minor}.{v.micro}" + ("" if py_ok else " (need 3.10+)"))

    core_imports = ["numpy", "cv2", "torch", "torchvision", "ultralytics", "sklearn", "scipy", "yaml",
                    "fastapi", "uvicorn", "pydantic", "trimesh", "pyrender",
                    "pytorch_lightning", "timm", "einops", "yacs", "roma", "omegaconf"]
    missing = [m for m in core_imports if not has_module(m)]
    imports_ok = not missing
    line(OK if imports_ok else FAIL,
         "Core packages" if imports_ok else f"Missing packages: {', '.join(missing)} (pip install -e .)")

    from var_offside.config import get_settings
    from var_offside.config.paths import project_home, third_party_dir, weights_dir

    s = get_settings()
    print("-" * 60)
    print(f"Project folder: {project_home()}")
    print(f"Weights folder: {weights_dir()}")
    print(f"Config file:    {s.config_file or 'none (built-in defaults)'}")

    import torch

    if torch.cuda.is_available():
        line(OK, f"CUDA: {torch.cuda.get_device_name(0)} (device = {s.device})")
    else:
        line(WARN, f"No CUDA GPU (device = {s.device}). Detection works on CPU; 3D bodies are very slow.")

    print("-" * 60)
    print("1. Detection (required)")
    det_ok = file_status(s.models.yolo_detection, "Football detector", True,
                         "Run: python scripts/download_weights.py")
    if Path(s.models.yolo_pose).exists():
        line(OK, f"Pose model: {s.models.yolo_pose}")
    else:
        line(OK, f"Pose model: {s.models.yolo_pose} (Ultralytics downloads it on first use)")

    print("-" * 60)
    print("2. Automatic calibration (optional: without it you place the pitch by hand)")
    pnl_dir = third_party_dir("PnLCalib")
    pnl_ok = pnl_dir.is_dir()
    line(OK if pnl_ok else WARN, f"PnLCalib code: {pnl_dir}" + ("" if pnl_ok else
         "\n       git clone https://github.com/mguti97/PnLCalib third_party/PnLCalib"))
    hint = "Run: python scripts/download_weights.py"
    pnl_ok &= file_status(s.models.pnl_keypoints, "PnLCalib keypoints", False, hint)
    pnl_ok &= file_status(s.models.pnl_lines, "PnLCalib lines", False, hint)

    print("-" * 60)
    print("3. 3D view")
    r3d = all(has_module(m) for m in ("trimesh", "pyrender"))
    line(OK if r3d else FAIL, "3D packages (trimesh, pyrender)" if r3d else
         "3D packages missing: pip install -e .")
    if r3d:
        try:
            import numpy as np
            import pyrender
            import trimesh

            scene = pyrender.Scene()
            scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.box(extents=(1, 1, 1))))
            pose = np.eye(4)
            pose[2, 3] = 3.0
            scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=pose)
            renderer = pyrender.OffscreenRenderer(64, 64)
            color, _ = renderer.render(scene)
            renderer.delete()
            line(OK, "OpenGL offscreen render works")
        except Exception as exc:  # noqa: BLE001
            r3d = False
            line(WARN, f"OpenGL render failed: {exc}\n       On a headless Linux box set PYOPENGL_PLATFORM=egl.")

    sam_code = third_party_dir("Fast-SAM-3D-Body").is_dir() or third_party_dir(
        "sam-3d-body", "sam3_body3d/sam-3d-body").is_dir()
    line(OK if sam_code else WARN, "SAM 3D Body code in third_party/" if sam_code else
         "SAM 3D Body code missing: python scripts/download_weights.py")
    hint = "python scripts/download_weights.py (the checkpoint needs Hugging Face access)"
    bodies = sam_code
    bodies &= file_status(s.var_3d.sam3d_checkpoint, "SAM 3D Body checkpoint", False, hint)
    bodies &= file_status(s.var_3d.sam3d_mhr, "SAM 3D Body MHR model", False, hint)

    print("=" * 60)
    print("Summary")
    core_ok = py_ok and imports_ok and det_ok and r3d
    line(OK if (py_ok and imports_ok and det_ok) else FAIL, "Detection and offside decision")
    line(OK if pnl_ok else WARN, "Automatic calibration" + ("" if pnl_ok else " (hand placement only)"))
    line(OK if r3d else FAIL, "3D view with stand-in players")
    line(OK if (r3d and bodies) else WARN, "3D view with real player bodies" + ("" if (r3d and bodies) else " (stand-ins until SAM 3D Body is in place)"))
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
