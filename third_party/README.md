# third_party/

Upstream code that VAR-Offside can use but does not bundle, because each project has its own licence. Nothing in this folder except this README is committed.

| Folder | Needed for | Get it | Licence |
|---|---|---|---|
| `PnLCalib/` | Automatic pitch calibration | `git clone --depth 1 https://github.com/mguti97/PnLCalib third_party/PnLCalib` | GPL-2.0 |
| `sam-3d-body/` | Real 3D player bodies | `git clone --depth 1 https://github.com/facebookresearch/sam-3d-body third_party/sam-3d-body` | SAM License |
| `Fast-SAM-3D-Body/` | Faster 3D bodies (optional) | `git clone --depth 1 https://github.com/yangtiming/Fast-SAM-3D-Body third_party/Fast-SAM-3D-Body` | MIT |

`python scripts/download_weights.py` clones the calibration code, and `--3d` also clones the 3D code.
Their weights go in `weights/`. See [docs/getting-started/models.md](../docs/getting-started/models.md).
