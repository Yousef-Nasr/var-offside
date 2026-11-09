# Models and weights

VAR-Offside uses several trained models. None of them are stored in git: they are large, and several have their own licences. You download them once into the `weights/` folder (and some upstream code into `third_party/`), and the app finds them from there.

## What you need

The 3D view is part of every install. The models switch on its parts:

| Feature | Models | Without them |
|---|---|---|
| **Detection** | `best_tuned.pt`, `yolo11x-pose.pt` | Nothing runs |
| **Automatic calibration** | PnLCalib | You place the pitch by hand in the web app, which works fine but takes a minute per frame |
| **3D view with real player bodies** | SAM 3D Body | The 3D view shows simple stand-in players until the checkpoint is in place |

## The fast way

The SAM 3D Body checkpoint is gated, so [request access](#4-real-player-bodies-sam-3d-body) first; approval can take a while. Then, from the project folder:

```bash
hf auth login                               # once, for the SAM 3D Body checkpoint
python scripts/download_weights.py          # everything
python scripts/verify_setup.py              # check what was found
```

If your access isn't approved yet, the script still gets everything else and tells you the checkpoint is missing. Run it again later; files already present are skipped. `--no-3d` skips the 3D body models.

`python scripts/download_weights.py --list` shows every file, its licence and where it goes.

## Where everything goes

After a full download the project folder looks like this:

```text
var-offside/
├── weights/
│   ├── best_tuned.pt                  # 1. football detector        (required)
│   ├── yolo11x-pose.pt                # 1. pose model               (required, auto-downloaded)
│   ├── pnlcalib/
│   │   ├── SV_kp                      # 2. calibration keypoints
│   │   └── SV_lines                   # 2. calibration lines
│   ├── yolo11l.pt                     # 4. person detector for SAM 3D Body
│   └── sam-3d-body-dinov3/            # 4. SAM 3D Body, as downloaded from Hugging Face
│       ├── model.ckpt
│       ├── model_config.yaml
│       └── assets/
│           └── mhr_model.pt
└── third_party/
    ├── PnLCalib/                      # 2. calibration code
    ├── sam-3d-body/                   # 4. SAM 3D Body code
    └── Fast-SAM-3D-Body/              # 4. optional, about 10x faster bodies
```

Both folders are in `.gitignore` except for their README files, so nothing you put there is committed by accident.

## Getting each file by hand

Use this if the script can't reach a server, or you'd rather see each step.

### 1. Detection (required)

| File | Where to get it | Put it at | Licence |
|---|---|---|---|
| `best_tuned.pt` | [VAR-Offside release `weights-v0.1.0`](https://github.com/Yousef-Nasr/var-offside/releases/tag/weights-v0.1.0) | `weights/best_tuned.pt` | AGPL-3.0 |
| `yolo11x-pose.pt` | [Ultralytics assets](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt) | `weights/yolo11x-pose.pt` | AGPL-3.0 |

`best_tuned.pt` is a YOLO detector fine-tuned for football. It finds four classes: `ball`, `goalkeeper`, `player` and `referee`.

If `yolo11x-pose.pt` is missing, Ultralytics downloads it automatically the first time you run a detection (into the folder you started from). Putting it in `weights/` avoids that.

### 2. Automatic calibration (PnLCalib)

[PnLCalib](https://github.com/mguti97/PnLCalib) finds the pitch markings and solves the broadcast camera. It's GPL-2.0, so it isn't bundled: clone the code and download the two single-view weights.

```bash
git clone --depth 1 https://github.com/mguti97/PnLCalib third_party/PnLCalib
```

| File | Where to get it | Put it at |
|---|---|---|
| `SV_kp` | [PnLCalib release v1.0.0](https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp) | `weights/pnlcalib/SV_kp` |
| `SV_lines` | [PnLCalib release v1.0.0](https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines) | `weights/pnlcalib/SV_lines` |

The files have no extension; keep the names exactly as shown.

### 3. 3D view

The 3D packages come with `pip install -e .`, and the view needs no weights of its own. It needs a working OpenGL driver; on a Linux server without a display, set `PYOPENGL_PLATFORM=egl`.

### 4. Real player bodies (SAM 3D Body)

[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) by Meta rebuilds each player as a 3D mesh. Its checkpoints are released under the **SAM License** and are gated on Hugging Face:

1. Open <https://huggingface.co/facebook/sam-3d-body-dinov3>, sign in and request access. Approval can take a while.
2. Log in from the terminal: `hf auth login` (install with `pip install huggingface_hub`).
3. Download the checkpoint folder:

    ```bash
    hf download facebook/sam-3d-body-dinov3 --local-dir weights/sam-3d-body-dinov3
    ```

4. Clone the code. Its runtime dependencies are already installed with VAR-Offside; you don't need upstream's longer training setup, Detectron2 or MoGe.

    ```bash
    git clone --depth 1 https://github.com/facebookresearch/sam-3d-body third_party/sam-3d-body
    ```

5. Optional, about 10x faster: also clone [Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) (MIT). VAR-Offside uses it automatically when it's present.

    ```bash
    git clone --depth 1 https://github.com/yangtiming/Fast-SAM-3D-Body third_party/Fast-SAM-3D-Body
    ```

6. Download [`yolo11l.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) to `weights/yolo11l.pt`.

A CUDA GPU is strongly recommended. The first bodies took about 70 seconds on a laptop RTX 3050; on a CPU it takes many minutes.

## Check your setup

```bash
python scripts/verify_setup.py
```

It reads the same settings as the app, so the paths it prints are the paths the app will load. The summary at the end shows which features are ready:

```text
Summary
[ OK ] Detection and offside decision
[ OK ] Automatic calibration
[ OK ] 3D view with stand-in players
[WARN] 3D view with real player bodies
```

## Keeping weights somewhere else

If your models live on another drive or are shared between projects, point the app at them instead of copying:

| Setting | What it changes |
|---|---|
| `VAR_OFFSIDE_WEIGHTS_DIR=/data/models/var-offside` | The whole `weights/` folder |
| `VAR_OFFSIDE_HOME=/srv/var-offside` | The folder that holds `weights/`, `third_party/`, `results/` and `config.yaml` |
| `models:` and `var_3d:` in [`config.yaml`](configuration.md) | One file at a time, as a name in `weights/` or an absolute path |

The app looks for each file in `weights/`, then `third_party/`, then the project folder itself, so the older layout with everything at the top level keeps working.

## Licences

The models come from different projects with different terms. Check them before you redistribute anything, or use the app commercially. See [Licensing](../about/licensing.md) for the full picture.

| Model | Licence | Redistributable here |
|---|---|---|
| `best_tuned.pt`, `yolo11x-pose.pt`, `yolo11l.pt` | AGPL-3.0 (Ultralytics) | Yes, under AGPL-3.0 |
| PnLCalib `SV_kp`, `SV_lines` | GPL-2.0 | Download from upstream |
| SAM 3D Body | SAM License (Meta) | No, download it yourself from Meta |
