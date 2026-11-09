# Auto-review

Auto-review is an optional switch next to the main button in the [web app](web-app.md). Turn it on, open a frame and press **Auto-review**. It runs the review checks in a fixed order, makes each correction itself when a vision language model gives a clear answer, and plays each correction on the photo. Turn it off and the app works as described in the web app guide.

!!! warning "Experimental"
    Accuracy on real frames is largely unmeasured. Check every result.

## Set it up

Auto-review calls a vision model through [OpenRouter](https://openrouter.ai). Copy the example settings and fill in your key and a model:

```bash
cp var_offside/.env.example var_offside/.env
```

| Setting | Meaning | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | Your key from <https://openrouter.ai/keys> | none |
| `OPENROUTER_MODEL_ID` | Any OpenRouter model that accepts images and JSON output | none |
| `OPENROUTER_MAX_COST_USD` | Stop a run once it has spent this much | `0.50` |
| `OPENROUTER_MAX_CALLS` | Stop a run after this many model calls | `20` |
| `AUTO_REVIEW_STAGE` | Rollout stage, 1 to 4. At 1, every result is marked "Check this result" | `1` |

`var_offside/.env` is ignored by git. Values in the process environment win over the file. Restart the server after changing it.

Each run makes up to about 10 model calls.

!!! danger "Your frames leave your machine"
    The frame and player crops go to OpenRouter and the model provider you chose. Check that against any rights or agreements that cover your footage.

## How a run ends

- **Cleared:** a call, with every correction listed. **Show** points to it, and **Undo** reverts it and reruns the call.
- **A question, or Review:** it asks only when it doesn't know. The model can't tell the direction, team or keeper, or it sees a possible missed player with no box who could change the call. A check that fails, or a problem it can't fix such as a bad camera, takes you to Review.
- **Too close:** the margin is smaller than the camera's error at the line, or the call depends on a measurement it can't settle, so there's no firm call.
- **Not usable:** the shot can't be judged, for example a close-up.

## What it does

**When the line appears.** The line and the call show only at the very end, after the camera fit and every check. If the camera fit doesn't finish, the run goes to Review with no line and no call.

**How it finds the attack.** The model gets one marked-up frame with each team in its own colour and letter, plus LEFT, RIGHT and the ball, and answers the attacking team and direction together. The letters vary by frame, so a habit of picking one letter can't favour one team.

**When the team decides the call.** The model's team is tested like any other doubt: the call is worked out again with the other team attacking. If that changes the call, the run asks you which team is attacking, unless a rule that doesn't use the model (the back-line shape or who is nearest the ball) names the same team. Likely keepers are drawn without a team letter in the image the model sees, so a keeper grouped with the wrong team can't decide the answer.

**Without a working key,** for example when its spending limit is reached, the rule checks still run. A frame never clears without vision: you get questions for the direction, team and keeper, then Review.

## Limits

A wrong clear answer from the model is applied too, even when it changes the call, so **Undo** is your safeguard. The attacking team is the exception: when the model's team would change the call and no rule backs it, the run asks. The model comparison covered only 4 to 6 frames for most checks, and side, attacking team, officials and whole calls have no labels yet. A missed player that the model doesn't notice either can still clear with a wrong call.

`AUTO_REVIEW_STAGE` stays at 1, which marks every result "Check this result", until measurements on labelled frames support more.

## Test your connection

This makes one real paid call with one image:

```bash
python scripts/smoke_openrouter.py path/to/frame.jpg
```

## Code

The agent lives in `var_offside/agent/`. Rule checks are in `agent/checks/`, the vision checks in `vision_checks.py`, the run order and decisions in `runner.py` and `policy.py`. The HTTP routes are in `var_offside/web/auto_review_routes.py` and documented in [Web API](../reference/web-api.md#auto-review).
