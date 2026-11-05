# VAR-Offside

Automated football **offside detection** with a broadcast-style **3D VAR replay**.

Given a match frame (or a video clip you scrub to the kick moment), the system
detects players, calibrates the pitch camera, computes the offside line, decides
who is ahead of it, and can render a 3D replay: the reconstructed players stand
on a real 105×68 m pitch with the offside plane drawn through the play.

The package lives in [`var_offside/`](var_offside/) and ships three front-ends:
a desktop GUI, a CLI, and an HTTP API, plus a browser app with a free-moving 3D
view. To run the browser app, see the [web app playbook](var_offside/web/README.md).

---

## Requirements

- **Python 3.10+**.
- A CUDA GPU is strongly recommended — the 3D player reconstruction (SAM 3D
  Body) is slow on CPU.
- A working OpenGL driver (the 3D replay renders with `pyrender`).

### Model weights

These files are **not** in git (they are large). Place them at the repo root:

| File                                     | Used for                                        |
| ---------------------------------------- | ----------------------------------------------- |
| `yolo11x-pose.pt`                      | player pose detection (required)                |
| `best_tuned.pt`                        | football object detector (required)             |
| `model_sam3_3d.ckpt`                   | SAM 3D Body — 3D player meshes                 |
| `yolo11l.pt`, `mobileclip2_b.ts`     | SAM 3D Body dependencies                        |
| `PnLCalib/weights/SV_kp`, `SV_lines` | camera calibration (needed for a real 3D pitch) |

Run `python scripts/verify_setup.py` to see exactly what is present or missing.

---

## Setup

```powershell
# Windows / PowerShell — creates .venv, installs everything, verifies
.\setup_env.ps1                 # default torch
.\setup_env.ps1 -Cuda cu121     # install the CUDA 12.1 torch wheels
```

Or manually:

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -e .
python scripts/verify_setup.py
```

---

## Running

### Desktop app (recommended)

```bash
python -m var_offside.app
```

1. **Open Image** or **Open Video**. For a video, scrub the slider to the kick
   moment.
2. **Run Detection** (Phase 1) — review/edit players, pick the attacking side.
3. **Continue Analysis** (Phase 2) — calibration, offside line, verdict.
4. **Generate 3D Replay** — pick *Cinematic*, *Zoom-out*, or *Debug overlay*,
   then watch it play back in-app.

### CLI

```bash
python -m var_offside.demo --image images/test9.png --attacking-side left
# add 3D:
python -m var_offside.visualization.var_3d --image images/test9.png \
    --output results/offside_3d.mp4 --attacking-side left
```

### HTTP API

```bash
python -m var_offside.api --host 0.0.0.0 --port 8000
```

---

## How the 3D replay works

- Players are reconstructed as 3D meshes by **SAM 3D Body** and planted on the
  pitch at the **same field position the offside decision uses** (the
  direction-aligned ground point projected through the calibrated camera), so
  the 3D scene agrees with the 2D verdict.
- The offside wall is placed at the decision's `field_x` (metres), not
  re-estimated from the image.
- A real 3D pitch needs full camera calibration (**PnL**); without it the
  replay falls back to a lower-fidelity view and the GUI explains why.

The SAM 3D Body checkpoint is loaded once per process and reused across
replays. Call `MeshRecovery.unload_all()` to free GPU memory.

---

## Troubleshooting

- **"3D rendering produced no output"** — the frame likely lacks PnL
  calibration. Pick a frame with clearly visible pitch lines, or drop the PnL
  weights into `PnLCalib/weights/`.
- **`pyrender` / OpenGL errors** — verify the driver with
  `python scripts/verify_setup.py`. On a headless Linux box set
  `PYOPENGL_PLATFORM=egl`.
- **Very slow 3D** — you are probably on CPU; a CUDA GPU is needed for
  interactive speed.

See [`var_offside/README.md`](var_offside/README.md) for the full pipeline and
API reference.
