# Troubleshooting

Start with the setup check. It prints exactly what the app will load and what's missing:

```bash
python scripts/verify_setup.py
```

The server window logs every error, so look there next.

## Install and start-up

| What you see | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: var_offside` | Not installed, or another environment is active | Activate your environment and run `pip install -e .` in the project folder |
| `ValueError: config.yaml: unknown setting ...` | A typo in `config.yaml` | Fix the name it shows; compare with `config.example.yaml` |
| `error while attempting to bind` | The port is in use, often by an old server | Stop it, or start with `--port 8770` |
| `torch.cuda.is_available()` is `False` with a GPU | CPU build of PyTorch | Reinstall PyTorch for your CUDA version, see [Installation](getting-started/installation.md#3-install-pytorch) |

## Models

| What you see | Likely cause | Fix |
|---|---|---|
| Detect fails, `best_tuned.pt` not found | The detector isn't in `weights/` | `python scripts/download_weights.py`, see [Models](getting-started/models.md) |
| Calibration always says NOT FOUND | PnLCalib code or weights missing | Check `third_party/PnLCalib` and `weights/pnlcalib/` |
| Stand-in players never become bodies | SAM 3D Body code or checkpoint missing, or the GPU ran out of memory | Read the status chip, then run `verify_setup.py` |
| `401` or `403` downloading SAM 3D Body | Access not approved yet, or not logged in | Request access on Hugging Face and run `hf auth login`. The 3D view uses stand-ins meanwhile |

## Web app

| What you see | Likely cause | Fix |
|---|---|---|
| The page looks old or a button is missing | Cached files | Hard refresh (Ctrl+F5, or Cmd+Shift+R) |
| A **Mock data** badge in the top bar | The page can't reach the server, or the address has `?mock=1` | Check the server is up, remove `?mock=1`, reload |
| Black 3D view, or the view disconnects | The renderer hit a problem | Click **Reconnect**; if that fails, restart the server |
| Video won't play in the browser | ffmpeg wasn't found, so the file stayed in an older format | Install ffmpeg on PATH, or `pip install imageio-ffmpeg` |
| Video exports are very slow | Running on the CPU | Check `/api/health` says `cuda` |

## 3D and OpenGL

| What you see | Likely cause | Fix |
|---|---|---|
| `pyrender` or OpenGL errors | No usable OpenGL context | Update the GPU driver. On a Linux server without a display, set `PYOPENGL_PLATFORM=egl` |
| The process exits silently while loading bodies | GPU driver conflict on mixed AMD + NVIDIA laptops | Run the analysis before loading SAM 3D Body; see [Architecture](development/architecture.md#rules-that-are-easy-to-break) |

## Wrong calls

| What you see | Likely cause | Fix |
|---|---|---|
| The line is on the wrong player | Wrong team, keeper or direction | Go back to **Edit players** and check all three |
| The line is at a strange angle | Bad calibration | **Edit calibration** and drag the lines onto the markings |
| Every call says **Estimate** | No full camera | Install PnLCalib, or place the pitch by hand |

Still stuck? [Open an issue](https://github.com/Yousef-Nasr/var-offside/issues/new/choose) with the output of `verify_setup.py`.
