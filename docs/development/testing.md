# Testing

The test suite needs no model weights and no GPU. Models are replaced with small fakes, so it runs anywhere in about a minute.

```bash
pip install -e ".[dev]"
pytest
```

| Folder | Covers |
|---|---|
| `tests/test_config.py` | `config.yaml` loading and model-path lookup |
| `tests/web/test_offside_rules.py` | The offside decision: line choice, body parts, keeper, halfway line, tolerances |
| `tests/web/test_pitch_solver.py` | Solving the camera from hand-placed points and lines |
| `tests/web/test_server.py`, `test_calibration_api.py` | HTTP routes and session flow, with fake pipeline phases |
| `tests/web/test_auto_review_*.py` | Auto-review routes and the data they rely on |
| `tests/agent/` | Auto-review rule checks, vision checks, run policy, OpenRouter client |

Run one area, or one test:

```bash
pytest tests/web/test_offside_rules.py
pytest -k keeper
```

## Markers

| Marker | Meaning | In CI |
|---|---|---|
| `weights` | Needs real model files in `weights/` | Skipped |
| `gpu` | Needs a CUDA GPU | Skipped |

Mark a test that loads a real model so CI can skip it:

```python
import pytest

@pytest.mark.weights
def test_detector_finds_players(): ...
```

## Smoke tests with real models

These run the real models on a frame you provide. They're slow and not part of `pytest`.

| Script | Checks |
|---|---|
| `scripts/verify_setup.py` | Packages, weights, CUDA and OpenGL |
| `scripts/smoke_live3d.py --image frame.png` | The live 3D scene and every camera preset |
| `scripts/check_pnl_homography.py frame.png` | PnLCalib's homography agrees with its camera |
| `scripts/smoke_openrouter.py frame.png` | One real (paid) Auto-review call |

## Linting

```bash
ruff check .
```

CI runs the same checks. `pre-commit install` runs them before each commit.
