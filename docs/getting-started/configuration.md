# Configuration

VAR-Offside works with no configuration. To change a default, copy the example file and edit what you need:

```bash
cp config.example.yaml config.yaml
```

`config.yaml` is ignored by git, so your local settings never end up in a commit. [`config.example.yaml`](https://github.com/Yousef-Nasr/var-offside/blob/main/config.example.yaml) lists every setting with its default and a comment.

## How settings are found

The app builds its settings once at start-up, in this order. Later steps win.

1. Built-in defaults (`var_offside/config/settings.py`).
2. The config file: `$VAR_OFFSIDE_CONFIG` if set, otherwise `config.yaml` in the project folder.
3. Command-line flags, for the few settings that have them (`--host`, `--port`, `--no-browser`, `--method`).

Every setting is optional. A misspelt name or a value of the wrong type stops the app with an error that names it, for example:

```text
ValueError: config.yaml: unknown setting 'server.prot'. Known: host, open_browser, port
```

To use another file:

```bash
var-offside-web --config ~/var-offside/stadium-cam.yaml
var-offside --config stadium-cam.yaml --image frame.png --attacking-side left --attacking-team 1
```

## Sections

| Section | What it controls |
|---|---|
| `device` | `auto`, `cuda`, `cuda:1` or `cpu`. `auto` uses CUDA when available |
| `output_dir` | Where exports are written. Relative to the project folder |
| `server` | Web app `host`, `port` and `open_browser` |
| `models` | Paths to detection and calibration weights |
| `detection` | Detector confidence and IoU thresholds, minimum player height, maximum players |
| `pose` | Minimum keypoint confidence |
| `teams` | Number of teams and how much of each box is used for kit colour |
| `calibration` | The classical (vanishing point) fallback: grass colour range, Hough and RANSAC parameters |
| `offside` | Pixel tolerance without calibration, pitch size, the 3D mesh anchor |
| `visualization` | Colours (BGR) and line thickness on exported images |
| `var_3d` | SAM 3D Body paths and replay video size and frame rate |

An example that runs on the CPU, serves on another port and writes exports elsewhere:

```yaml
device: cpu
output_dir: D:/var-exports
server:
  port: 9000
  open_browser: false
detection:
  confidence_threshold: 0.35
```

## Model paths

Leave `models` and `var_3d` paths unset if you use the [standard layout](models.md#where-everything-goes). When you do set one:

- A bare name such as `best_tuned.pt` or `pnlcalib/SV_kp` is looked up in `weights/`, then `third_party/`, then the project folder.
- An absolute path is used as given. `~` is expanded.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `VAR_OFFSIDE_HOME` | Project folder: holds `weights/`, `third_party/`, `results/` and `config.yaml` | The cloned repository |
| `VAR_OFFSIDE_WEIGHTS_DIR` | Model folder | `<home>/weights` |
| `VAR_OFFSIDE_RESULTS_DIR` | Export folder (overridden by `output_dir` in the config) | `<home>/results` |
| `VAR_OFFSIDE_CONFIG` | Config file | `<home>/config.yaml` |
| `PYOPENGL_PLATFORM` | Set to `egl` on a Linux machine without a display | |

## Secrets

The OpenRouter key for [Auto-review](../guide/auto-review.md) is a secret, so it isn't in `config.yaml`. It goes in `var_offside/.env` (also ignored by git) or the process environment.

## From Python

```python
from pathlib import Path
from var_offside.config import get_settings, load_settings

settings = get_settings()                 # defaults + config.yaml, cached
print(settings.server.port, settings.models.yolo_detection)

custom = load_settings(Path("other.yaml"))  # a fresh, uncached copy
```
