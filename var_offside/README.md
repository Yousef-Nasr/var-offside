# VAR Offside Detection System

Automatic offside detection for football/soccer from TV broadcast freeze-frames.

## Features

- **YOLO Pose-based Detection**: Unified player detection and pose estimation
- **Team Assignment**: ResNet embeddings + K-Means clustering for team separation
- **Multiple Calibration Methods**:
  - PnLCalib (deep learning-based)
  - Vanishing Point (classical CV)
  - Manual point annotation
- **Automatic Side Detection**: Determines attacking direction from goalkeeper positions
- **FIFA-compliant Offside Line**: Uses second-to-last defender's most defensive body part
- **VAR-style Visualization**: Clear offside/onside indicators

## Installation

```bash
cd var_offside
pip install -r requirements.txt
```

## Quick Start

```python
from var_offside import OffsidePipeline

# Initialize pipeline
pipeline = OffsidePipeline(calibration_method="vanishing_point")

# Analyze image
result = pipeline.analyze(
    "images/test9.png",
    attacking_team=1,
    attacking_side="left",
    save_result=True
)

# Print summary
print(pipeline.get_summary(result))

# Check result
if result.is_offside:
    print(f"OFFSIDE! {result.offside_result.num_offside} player(s) in offside position")
else:
    print("All players are onside")
```

## Desktop App

```bash
python -m var_offside.app
```

The app now requires you to choose the attacking team before the offside stage
continues, because automatic attacking-team detection is not implemented safely.

## HTTP API

```bash
python -m var_offside.api --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/analyze-image" ^
  -F "file=@images/test9.png" ^
  -F "attacking_team=1" ^
  -F "attacking_side=left" ^
  -F "calibration_method=auto"
```

The API returns JSON with:

- overall offside verdict
- offside players
- all attacking-player statuses
- the defender defining the line
- calibration status and errors

## Command Line Usage

```bash
# Basic usage
python -m var_offside.demo --image images/test9.png

# With manual overrides
python -m var_offside.demo --image images/test9.png --attacking-team 1 --attacking-side left

# Specify calibration method
python -m var_offside.demo --image images/test9.png --method vanishing_point

# Save to custom path
python -m var_offside.demo --image images/test9.png --output results/my_analysis.jpg
```

## Project Structure

```
var_offside/
├── __init__.py
├── config/                 # Configuration
│   ├── __init__.py
│   └── settings.py
├── calibration/           # Field calibration
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── manual_calibration.py
│   ├── pnl_calibration.py
│   └── vanishing_point.py
├── detection/             # Player detection
│   ├── __init__.py
│   ├── body_parts.py
│   ├── goalkeeper_filter.py
│   └── player_detector.py
├── offside/               # Offside logic
│   ├── __init__.py
│   ├── decision.py
│   ├── line_calculator.py
│   └── side_detector.py
├── teams/                 # Team assignment
│   ├── __init__.py
│   ├── clustering.py
│   ├── embeddings.py
│   └── team_assigner.py
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── geometry.py
│   ├── homography.py
│   └── image_utils.py
├── visualization/         # Visualization
│   ├── __init__.py
│   ├── field_overlay.py
│   ├── offside_viz.py
│   └── player_viz.py
├── demo.py               # Demo script
├── pipeline.py           # Main pipeline
└── requirements.txt
```

## Pipeline Stages

1. **Image Loading**: Load and validate input image
2. **Field Calibration**: Detect field lines and compute homography
3. **Player Detection**: YOLO Pose for bounding boxes + 17 keypoints
4. **Team Assignment**: ResNet embeddings → UMAP → K-Means clustering
5. **Goalkeeper Identification**: Position and color-based detection
6. **Side Detection**: Determine attacking direction from GK positions
7. **Offside Line Calculation**: Find second-to-last defender
8. **Offside Evaluation**: Compare attackers to offside line
9. **Visualization**: Generate VAR-style output

## Calibration Methods

### Auto (default)

Tries PnLCalib first, falls back to vanishing point.

### PnLCalib

Deep learning-based keypoint and line detection for accurate camera calibration.
Requires PnLCalib weights to be available.

### Vanishing Point

Classical CV approach:

- Green field segmentation
- Hough line detection
- RANSAC-based vanishing point estimation

### Manual

Interactive point-and-click calibration for maximum control.

## Models Required

- `yolo11x-pose.pt` - YOLO Pose model for player detection
- PnLCalib weights (optional) - For deep learning calibration

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## License

MIT License
