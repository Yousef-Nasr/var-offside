# Quickstart

This takes you from an installed package to an offside decision on one frame. You need [the package installed](installation.md) and at least the detection weights from [Models](models.md).

## Try the interface without models

Mock mode runs the whole interface with made-up data. Nothing is detected or calibrated, so the verdicts mean nothing, but it shows you every screen:

```bash
var-offside-web
```

Then open <http://127.0.0.1:8765/?mock=1>.

## Make a real call in the browser

1. Start the server, which opens your browser at <http://127.0.0.1:8765>:

    ```bash
    var-offside-web
    ```

2. Drop a broadcast frame (PNG or JPG) or a clip (MP4, MOV) on the page. For a clip, move the timeline to the moment the ball is played.
3. Press **Detect**. The first run is slower while the models load.
4. Review the players: click a player to fix the team or mark the keeper, then choose the **attack direction**, the **attacking team** and the **keeper** (or **Not in frame**).
5. Press **Calibrate**. Check that the yellow pitch lines sit on the real markings; drag any that miss.
6. Press **F5** to continue. The verdict shows the call and the margin in metres, with the photo view and the 3D view side by side.
7. Use **Export** for an image, a replay video or a ZIP report. Files are saved in `results/`.

The [Web app guide](../guide/web-app.md) explains every step and control.

## Make a call from the command line

```bash
var-offside --image frame.png --attacking-side left --attacking-team 1 --no-display
```

The annotated image is saved to `results/frame_offside.jpg`:

```text
==================================================
VAR OFFSIDE ANALYSIS RESULT
==================================================
Players detected: 19
Team 1: 12 players | Team 2: 7 players
Calibration: pnl (quality: 0.80)
Attack side: left
Attacking team: 1 | Defending team: 2

Players ahead of the line detected
   3 player(s) ahead of the line
==================================================
```

Teams are numbered by kit-colour clustering, so team 1 isn't always the same kit. Run once with `--debug` to see which team got which number. See [Command line](../guide/cli.md).

## From Python

```python
from var_offside import OffsidePipeline

pipeline = OffsidePipeline(calibration_method="auto")
result = pipeline.analyze(
    "frame.png",
    attacking_side="left",
    attacking_team=1,
    save_result=True,
)

print(pipeline.get_summary(result))
if result.offside_result is not None:
    print("Players ahead of the line:", result.offside_result.num_offside)
```

See [Python API](../reference/python-api.md).
