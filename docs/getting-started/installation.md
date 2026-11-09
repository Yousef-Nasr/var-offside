# Installation

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| OS | Windows, Linux or macOS | Windows or Linux |
| GPU | none (CPU works for detection) | NVIDIA GPU with CUDA, 6 GB+ for 3D bodies |
| OpenGL | A working driver, for the 3D view | `PYOPENGL_PLATFORM=egl` on a Linux server without a display |

Detection, calibration, the verdict and the 3D view with stand-in players run fine on a CPU, just slower. Real 3D player bodies need a CUDA GPU in practice.

## 1. Get the code

```bash
git clone https://github.com/Yousef-Nasr/var-offside.git
cd var-offside
```

## 2. Create an environment

With `venv`:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Or with Conda:

```bash
conda create -n var-offside python=3.11
conda activate var-offside
```

## 3. Install PyTorch

Install PyTorch first so you get the right build for your GPU. Pick your platform at <https://pytorch.org/get-started/locally/>. For example, CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Skip this step for a CPU-only install; the next step pulls in the CPU build.

## 4. Install VAR-Offside

```bash
pip install -e .
```

That installs everything the app does: detection, calibration, the offside decision, the web app, the 3D view and video exports, the SAM 3D Body runtime, the CLI and the Auto-review client.

Optional extras:

| Extra | Adds |
|---|---|
| `dev` | `pytest`, `ruff`, `pre-commit`, `httpx`, for contributors |
| `docs` | `mkdocs-material` to build this site |
| `umap` | UMAP before team clustering (`umap-learn`) |

Install in editable mode (`-e`) from the cloned folder. The app finds `weights/`, `third_party/` and `config.yaml` relative to that folder.

## 5. Download the models

```bash
python scripts/download_weights.py
```

See [Models and weights](models.md) for what each file does and where it goes. The SAM 3D Body checkpoint for real 3D bodies needs Hugging Face access, so request it early.

## 6. Check the setup

```bash
python scripts/verify_setup.py
```

Then go to the [Quickstart](quickstart.md).

## Docker

There's no official image yet. If you build one, the web app needs `PYOPENGL_PLATFORM=egl` and an EGL-capable driver for the 3D view, and `--host 0.0.0.0` inside the container.
