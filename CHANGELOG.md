# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - not yet released

First public release.

### Added

- Web app: load a frame or clip, review players, calibrate, verdict with the margin in metres, live 3D view, exports (image, replay, bird's-eye and debug videos, JSON report).
- Calibration: PnLCalib, hand placement with a camera solver and named warnings, vanishing-point fallback.
- Offside decision from pose keypoints: second-last defender counting the keeper, arms excluded, own-half rule.
- 3D view with SAM 3D Body meshes placed at the decision's pitch positions.
- Auto-review (experimental): rule checks plus vision checks through OpenRouter, with undo.
- `var-offside` CLI, now with a required `--attacking-team`, and the `OffsidePipeline` Python API.
- `config.yaml` for all settings, with validation of unknown keys.
- `weights/` and `third_party/` folders, `scripts/download_weights.py` and a per-feature `scripts/verify_setup.py`.
- 3D view, replay and the SAM 3D Body runtime are part of the core install, not an extra.
- Documentation site.

### Fixed

- The web server accepted uploads of any size and type. It now refuses files over `server.max_upload_mb` (500 MB by default) with HTTP 413, checking the Content-Length before the body is read, and anything that isn't an image or video with HTTP 415.

- Auto-review no longer clears a call with the wrong attacking team when the keeper was grouped with the attackers. Likely keepers get no team letter in the image sent to the model, the team swap is tested as a doubt, and the run asks when it would change the call. `flips_call` on the team decision is now measured once the keeper is known.
- The replay caption showed a different margin from the verdict panel (1.17 m against 1.26 m) because it measured the arm-free body mesh again. The caption and the live 3D label now show the panel's number, and the label no longer shows a negative distance on reverse-angle cameras.

### Removed

- The PyQt desktop app and the separate HTTP API. The web app replaces both.
