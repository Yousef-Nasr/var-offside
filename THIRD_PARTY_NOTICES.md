# Third-party notices

VAR-Offside is licensed under AGPL-3.0-or-later. It depends on, or can use, the projects below. **None of their code or model weights is included in this repository.** Each is installed by pip or downloaded by the user from its source, under its own licence.

## Models and upstream code

| Component | Licence | Source |
|---|---|---|
| Ultralytics YOLO (library, `yolo11x-pose.pt`, `yolo11l.pt`) | AGPL-3.0 | https://github.com/ultralytics/ultralytics |
| `best_tuned.pt` (fine-tuned from Ultralytics YOLO) | AGPL-3.0 | VAR-Offside releases |
| PnLCalib (code and `SV_kp`, `SV_lines` weights) | GPL-2.0 | https://github.com/mguti97/PnLCalib |
| SAM 3D Body (code and checkpoints, including the MHR model) | SAM License | https://github.com/facebookresearch/sam-3d-body, https://huggingface.co/facebook/sam-3d-body-dinov3 |
| Fast-SAM-3D-Body | MIT | https://github.com/yangtiming/Fast-SAM-3D-Body |
| torchvision ResNet-50 ImageNet weights | BSD-3-Clause | https://github.com/pytorch/vision |

## Python packages

Installed by pip from `pyproject.toml`:

| Package | Licence |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| NumPy, SciPy, scikit-learn | BSD-3-Clause |
| OpenCV (`opencv-python`) | Apache-2.0 |
| Pillow | MIT-CMU (HPND) |
| PyYAML | MIT |
| FastAPI, Starlette, Pydantic | MIT |
| Uvicorn | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| Requests | Apache-2.0 |
| python-dotenv | BSD-3-Clause |
| trimesh | MIT |
| pyrender | MIT |
| imageio-ffmpeg | BSD-2-Clause (its bundled ffmpeg binary is LGPL/GPL) |
| umap-learn (optional) | BSD-3-Clause |

## Web page

The page loads fonts from Google Fonts at runtime: Barlow Condensed, IBM Plex Sans and IBM Plex Mono, all under the SIL Open Font License 1.1. Without internet access it falls back to system fonts. No JavaScript libraries are bundled or loaded from a CDN.
