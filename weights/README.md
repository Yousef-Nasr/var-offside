# weights/

Model files go here. Nothing in this folder except this README is committed.

Quick start, from the project folder:

```bash
python scripts/download_weights.py      # detection + automatic calibration
python scripts/verify_setup.py          # check what was found
```

Expected layout:

```text
weights/
├── best_tuned.pt                  # required  - football detector (ball, goalkeeper, player, referee)
├── yolo11x-pose.pt                # required  - pose model (Ultralytics downloads it if missing)
├── pnlcalib/
│   ├── SV_kp                      # optional  - automatic calibration
│   └── SV_lines                   # optional  - automatic calibration
├── yolo11l.pt                     # optional  - 3D bodies
└── sam-3d-body-dinov3/            # optional  - 3D bodies, from Hugging Face (gated)
    ├── model.ckpt
    ├── model_config.yaml
    └── assets/mhr_model.pt
```

Where each file comes from, its licence, and how to download it by hand:
[docs/getting-started/models.md](../docs/getting-started/models.md)

To keep the models somewhere else, set `VAR_OFFSIDE_WEIGHTS_DIR` or the paths in `config.yaml`.
