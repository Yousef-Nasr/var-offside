# Python API

The pipeline can be used directly from Python. The interface below is what the CLI uses; it may still change before 1.0.

## `OffsidePipeline`

```python
from var_offside import OffsidePipeline

pipeline = OffsidePipeline(calibration_method="auto", device=None)
```

| Argument | Meaning |
|---|---|
| `calibration_method` | `"auto"` (PnLCalib, then vanishing point), `"pnl"`, `"vanishing_point"` or `"manual"` |
| `device` | `"cuda"`, `"cpu"` or `None` to use the `device` setting |

Models load on first use and stay loaded, so create one pipeline and reuse it for many frames.

### `analyze()`

```python
result = pipeline.analyze(
    image,                       # path or BGR numpy array
    attacking_side="left",       # "left" or "right" (required)
    attacking_team=1,            # 1 or 2: which kit cluster attacks
    exclude_goalkeeper=True,
    offside_method="geometry",   # "geometry", "mesh_3d" or "auto"
    generate_visualization=True,
    save_result=False,
    output_path=None,
    verbose=False,
    debug=False,                 # save each stage to results/debug/
)
```

| `offside_method` | Meaning |
|---|---|
| `geometry` | Pose keypoints projected through the calibration. The default, and what the web app uses |
| `mesh_3d` | The furthest-forward vertex of each player's SAM 3D Body mesh. Needs the 3D models |
| `auto` | `geometry`, falling back to `mesh_3d` when the line has no reliable pitch projection |

Video options: `generate_3d_video`, `generate_debug_3d_video`, `generate_zoomout_video`.

### `AnalysisResult`

| Field | Type | Meaning |
|---|---|---|
| `is_offside` | `bool` | Any attacker ahead of the line |
| `detections` | `list[PlayerDetection]` | Boxes, pose keypoints, team, keeper flag |
| `calibration` | `CalibrationResult` | Method, quality score, homography, camera |
| `offside_line` | `OffsideLine` | The line, in image and pitch coordinates |
| `offside_result` | `OffsideResult` | Per-player status and margins; `num_offside`, `most_offside_player` |
| `attacking_team`, `defending_team` | `int` | 1 or 2 |
| `visualization` | `numpy.ndarray` | The annotated image |
| `errors` | `list[str]` | Why a stage couldn't finish. Check this before trusting the result |
| `processing_time_ms` | `float` | Total time |

`pipeline.get_summary(result)` returns the text summary the CLI prints.

## Example: a folder of frames

```python
from pathlib import Path
from var_offside import OffsidePipeline

pipeline = OffsidePipeline()
for frame in sorted(Path("frames").glob("*.png")):
    result = pipeline.analyze(frame, attacking_side="right", attacking_team=2,
                              generate_visualization=False)
    if result.errors:
        print(frame.name, "no call:", "; ".join(result.errors))
        continue
    best = result.offside_result.most_offside_player
    print(frame.name, "OFFSIDE" if result.is_offside else "onside",
          f"{best.margin_px:.1f}px" if best else "")
```

## Lower-level pieces

Each stage can be used on its own:

| Module | Main class or function |
|---|---|
| `var_offside.detection` | `PlayerDetector`, `GoalkeeperFilter` |
| `var_offside.teams` | `TeamAssigner` |
| `var_offside.calibration` | `get_calibrator(method)`, `CalibrationResult` |
| `var_offside.calibration.pitch_solver` | Solve a camera from hand-placed points and lines |
| `var_offside.offside` | `OffsideLineCalculator`, `OffsideDecision` |
| `var_offside.visualization` | `OffsideVisualizer`, `PlayerVisualizer`, `FieldOverlay` |
| `var_offside.config` | `get_settings()`, `load_settings(path)` |
