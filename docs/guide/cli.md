# Command line

`var-offside` runs the full pipeline on one image and saves an annotated copy. It's useful for scripting and batch runs. For anything you need to review or correct, use the [web app](web-app.md).

```bash
var-offside --image frame.png --attacking-side left --attacking-team 1
```

`python -m var_offside.demo` is the same command.

## Options

| Option | Meaning |
|---|---|
| `--image`, `-i` | Input image (required) |
| `--attacking-side` | `left` or `right`: the direction the attacking team is playing towards in the image (required) |
| `--attacking-team` | `1` or `2`: which kit cluster is attacking (required) |
| `--method`, `-m` | Calibration: `auto` (default), `pnl`, `vanishing_point` or `manual` |
| `--output`, `-o` | Output image path. Default: `results/<name>_offside.jpg` |
| `--config` | A config file other than `config.yaml` |
| `--no-display` | Don't open a window with the result. Use this on servers and in scripts |
| `--verbose`, `-v` | Print each attacker's distance from the line |
| `--debug`, `-d` | Save an image for each stage to `results/debug/` |

The exit code is `0` when the analysis finished without errors, and `1` otherwise.

## Which team is team 1?

Players are split into two teams by kit colour, and the numbering comes from clustering, not from the kits. Run once with `--debug` and open `results/debug/4_teams.jpg` to see which team got which number. The number usually stays the same for frames from the same match and camera, but check when it matters.

## Calibration methods

| Method | When to use it |
|---|---|
| `auto` | Tries PnLCalib, falls back to the vanishing-point method |
| `pnl` | PnLCalib only. Fails if its weights or code are missing |
| `vanishing_point` | Classical lines-and-grass method. No weights needed, but no metric margin |
| `manual` | Opens a window to click pitch points yourself |

See [Calibration](../concepts/calibration.md) for how they differ.

## Batch example

```bash
for f in frames/*.png; do
  var-offside --image "$f" --attacking-side left --attacking-team 1 --no-display \
    || echo "check $f"
done
```
