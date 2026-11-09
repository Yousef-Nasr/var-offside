# Web API

The web app is a FastAPI server. It serves the page from `var_offside/web/static/` at `/` and this API under `/api`. The live 3D view streams over a WebSocket.

Start it with `var-offside-web` (see [Web app](../guide/web-app.md)). FastAPI's interactive docs are at `http://127.0.0.1:8765/docs` while the server runs.

This page is the contract between the server (`var_offside/web/server.py`), the live 3D renderer (`var_offside/web/live3d.py`) and the page (`var_offside/web/static/`). Change it in the same pull request as the code, and keep `tests/web/test_server.py` passing.

!!! warning "No authentication"
    The server has no login. It binds to `127.0.0.1` by default. Only expose it on a network you trust.

## User flow (always pause for review)

1. **Open**: drop or pick an image or video. A video shows a timeline; the user scrubs to the pass frame.
2. **Detect** (phase 1): players and teams. No keeper is marked and nothing is suggested. The app *always*
   stops here for review.
3. **Review**: the user may click a player to switch team / mark as not a player / mark as goalkeeper, and
   drag a box on the image to add a player. The user **must** choose three things before moving on: the attack
   direction (left/right arrows), the attacking team (Team 1 / Team 2), and the keeper (one marked player, or
   "Not in frame"). Marking a keeper replaces any other keeper. Player edits never change the attacking team.
4. **Calibrate**: the pitch is drawn over the photo. One key accepts it. Otherwise the user drags points or
   lines onto the real markings, or places points by hand. Automatic calibration starts in the background
   when Detect finishes. No verdict is shown here.
5. **Analyze** (phase 2): offside line and verdict, using the calibration from step 4. Verdict + checks
   appear; the 3D view loads. An **Edit calibration** action returns to step 4.
6. **3D view**: free orbit, presets, replay inside the view.
7. **Export**: image, replay video, bird's-eye video, debug video, report.

## Common shapes

```jsonc
// Player
{"id": 3, "bbox": [x1, y1, x2, y2], "team": 1 | 2 | null,
 "role": "player" | "goalkeeper" | "referee", "confidence": 0.91}

// Check (one line in the panel's Checks list)
{"key": "players" | "teams" | "side" | "calibration" | "defender" | "verdict",
 "label": "Players", "status": "ok" | "check" | "fail" | "pending", "detail": "19"}
```

Errors: any 4xx/5xx returns `{"detail": "human readable message"}`.
Image coordinates are pixels of the original frame.

## HTTP API (prefix `/api`)

| Method + path | Body | Returns |
|---|---|---|
| `GET /health` | – | `{"status":"ok","device":"cuda"\|"cpu"}` |
| `POST /sessions` | multipart `file` (image or video, up to `server.max_upload_mb`; 413 if larger, 415 if not an image or video) | `{"id","kind":"image"\|"video","width","height","frame_count"?, "fps"?, "filename"}` |
| `GET /sessions/{id}/frame.jpg?index=N` | – | JPEG of frame N (images: N ignored) |
| `DELETE /sessions/{id}` | – | `{"status":"deleted"}`. Stops its runs and exports, and deletes the uploaded file, results, body meshes and export files. 404 if unknown |
| `POST /sessions/{id}/select-frame` | `{"index": N}` | `{"index": N}` — the frame later phases analyse (default 0) |
| `POST /sessions/{id}/detect` | `{}` | **Phase1** (below) |
| `PUT /sessions/{id}/players` | `{"players": [Player…]}` (ids of new players may be `null`) | **Phase1** with redrawn stages |
| `GET /sessions/{id}/calibration` | – | **Calibration** (below); starts automatic calibration if needed |
| `POST /sessions/{id}/calibration/solve` | **Pins** | `{"mode","field_to_image","camera","fit"}`; a trial, changes nothing |
| `PUT /sessions/{id}/calibration` | **Pins** | **Calibration**; applies the pins and clears the verdict (400 if the fit is refused) |
| `DELETE /sessions/{id}/calibration` | – | **Calibration**; back to the automatic result, clears the verdict |
| `POST /sessions/{id}/analyze` | `{"attacking_side":"left"\|"right","attacking_team":1\|2,"keeper":"player"\|"not_in_frame"}` | **Verdict** (below). 400 for an invalid side. A missing or conflicting team or keeper choice returns 200 with verdict `"unknown"`, a `message`, and no earlier result, without waiting for calibration |
| `POST /sessions/{id}/auto-review` | – | `{"job_id","phase1":Phase1,"vision":bool}`. 400 before Detect. Replaces any earlier run on this frame |
| `GET /sessions/{id}/auto-review?since=N` | – | **AutoReviewStatus** (below). 404 if no run |
| `POST /sessions/{id}/auto-review/answers/{qid}` | `{"choice": string}` | `{"status":"running"}`. 409 if the scene changed, or that question isn't waiting |
| `POST /sessions/{id}/auto-review/decisions/{did}/undo` | – | `{"events":[Event…],"result":AutoReviewResult}`. 409 if the scene changed, or the run hasn't ended |
| `GET /sessions/{id}/stages/{key}.png` | – | PNG of a debug stage: `original, detection, teams, calibration, offside, birdseye, result` (404 if not produced yet) |
| `POST /sessions/{id}/export` | `{"kind":"image"\|"replay"\|"birdseye"\|"debug"\|"report", "options"?: Options}` | `{"job_id"}` |
| `GET /jobs/{job_id}` | – | `{"status":"running"\|"done"\|"error"\|"cancelled","progress":0-100,"message","file_url": "/api/files/<name>" \| null}` |
| `DELETE /jobs/{job_id}` | – | `{"status":"cancelled"}` |
| `GET /files/{name}` | – | the exported file (download) |

```jsonc
// Phase1
{"players": [Player…], "team_colors": {"1": "#ff3347", "2": "#2e7dff"},  // fixed: Team 1 red, Team 2 blue
 "checks": [Check…], "stages": ["original","detection","teams"], "time_ms": 812}

// Pins
{"mode": "edit" | "hand",           // edit moves the automatic camera; hand solves from the pins alone
 "pins": [{"kind": "point", "ref": "r_spot", "x": 812.0, "y": 410.0},  // a landmark at this pixel
          {"kind": "line", "ref": "halfway", "x": 300.0, "y": 500.0}],  // this line passes through this pixel
 "start": Camera3x?,                // optional: the camera from the last solve, keeps a drag quick
 "attacking_side": "right"}         // optional: warns about the wrong end of the pitch

// Calibration
{"status": "running" | "ready", "mode": "edit" | "hand", "source": "auto" | "edited" | "hand",
 "method": "pnl" | "vanishing_point" | "failed" | "manual", "full_3d": true, "auto_3d": true,
 "field_to_image": [[...],[...],[...]] | null,  // field metres to image pixels
 "camera": {...} | null,            // opaque: send back as "start"
 "fit": {"tone": "ok"|"warn"|"bad"|"idle", "label": "GOOD FIT", "message": "...", "can_continue": true,
         "miss_px": 1.2, "line_error_m": 0.11, "worst": 2, "needed": 0},
 "pins": [...], "pitch": {"landmarks": [...], "lines": [...], "arcs": [...], "length": 105, "width": 68}}

// Verdict
{"verdict": "offside" | "onside" | "unknown",
 "estimate": false,                  // true: a call from the drawn line without a trusted full 3D calibration;
                                     // margin_m and line_field_x are then null
 "message": "..." | null,            // why the verdict is unknown, or what the estimate means;
                                     // says so when a NOT BELIEVABLE camera was rejected and not used
 "margin_m": 0.84 | null,            // + = attacker ahead of the line, metres; null without full 3D calibration
 "attacking_side": "right",
 "last_defender": {"id": 7, "part": "left heel" | null},
 "offside_player_ids": [2],
 "line_image_points": [[x, y], [x, y]],   // offside line in image pixels, ends clipped to the frame
 "line_field_x": 87.31 | null,            // the line's position along the pitch, metres; null on an estimate
 "calibration": {"method": "pnl" | "vanishing_point" | "failed" | "manual", "full_3d": true, "edited": false},
 "checks": [Check…], "stages": [...], "time_ms": 4210, "view3d_available": true}
```

## Auto-review

An optional run that checks and corrects the scene. Its events are defined in
`var_offside/agent/schemas.py`. With the switch off, nothing here is called.

```jsonc
// AutoReviewStatus
{"status": "running" | "needs_answer" | "done" | "error" | "cancelled",
 "current_check": "keeper" | null, "message": null, "events": [Event…],
 "result": AutoReviewResult | null}

// AutoReviewResult
{"ending": "cleared" | "needs_answer" | "too_close" | "not_usable",
 "reason": string | null, "summary": string | null,
 "players": [Player…], "attacking_side": "left" | "right" | null, "attacking_team": 1 | 2 | null,
 "keeper": "player" | "not_in_frame" | null, "verdict": Verdict | null,
 "stability": {"alternatives_tested": 13, "changed_the_call": 0} | null,
 "uncertainty_m": 0.3 | null, "stage": 1}
```

- `status: "needs_answer"` pauses the run on one `question_asked` event, until it's answered.
- `run_ended` with `needs_answer` means the run has finished and Review must settle the frame.
- The final `verdict` comes from the normal Analyze. A too-close result has verdict `"unknown"`, keeps the line, and has no margin.

## WebSocket `/api/sessions/{id}/view` (live 3D)

Opens after Analyze. Text frames are JSON; binary frames are JPEG images of the current view.

Client → server:

```jsonc
{"type": "camera", "camera": Camera, "width": 1280, "height": 720,
 "quality": "draft" | "full", "options": Options}
{"type": "preset", "name": "broadcast" | "birdseye" | "line" | "free", "width": 1280, "height": 720, "options": Options}
{"type": "replay", "width": 1280, "height": 720, "options": Options}
{"type": "stop"}          // stops a running replay; the camera glides back
{"type": "visible", "visible": false}   // the 3D view is hidden: pause the idle pulse frames
```

Server → client:

```jsonc
{"type": "status", "stage": "loading" | "proxy" | "meshes" | "ready" | "error", "message": "Body meshes 2/4"}
{"type": "camera", "camera": Camera}    // sent after every preset and at the end of a replay, so the client syncs its orbit state
{"type": "replay_end"}
{"type": "defender_part", "name": "RIGHT HEEL"}   // the line-setting body part, once bodies are ready
<binary JPEG>                            // one per render; replay streams ~24 per second
```

```jsonc
// Camera: orbit camera in scene metres (Y up). The server owns the convention; the client only
// adds deltas to what the server last sent and sends it back.
{"target": [x, y, z], "yaw": 35.0, "pitch": 22.0, "distance": 18.0, "fov": 40.0}
//   yaw   degrees around the vertical axis, any value (client wraps)
//   pitch degrees above the pitch plane, clamp 3..89 (never under the grass)
//   distance metres from target, clamp 2..300

// Options
{"line_plane": true, "photo": false, "labels": true, "pulse": true, "key_only": false}
// key_only: show only the last defender and the attacker, in the view, the replay and video exports
```

Rendering rules: `quality:"draft"` renders at half size and JPEG quality 70 for smooth dragging; `"full"`
renders at the requested size, quality 90. The server drops stale camera messages and renders only the newest.
Until body meshes are ready the scene shows stand-in player shapes (`status: "proxy"`), then upgrades
(`status: "ready"`) and re-sends a frame. Input works during that time. A preset glides there in about 0.75 s
and sends draft frames on the way; a `camera` message cancels the glide. While the slow pulse is on and nothing
else is happening, the server sends about 8 frames per second so the pulse keeps moving. A replay holds real
speed by skipping frames, is capped at 1280 wide, and waits for body meshes before it starts. Clients must not
send a `camera` message until they have received their first `camera` message.

## `LiveScene` (Python, `var_offside/web/live3d.py`)

```python
class LiveScene:
    def __init__(self, result: "AnalysisResult", image_bgr: np.ndarray, device: str = "cuda",
                 on_status: Callable[[str, str], None] | None = None): ...
    def build_proxy(self) -> None            # fast: pitch, line, stand-in players
    def recover_bodies(self, should_stop=None) -> bodies | None   # slow: SAM 3D Body, no OpenGL, any thread
    def plant_bodies(self, bodies, should_stop=None) -> None      # swap stand-ins for bodies
    def upgrade_meshes(self, should_stop=None) -> None            # both of the above, one thread
    has_pulse: bool; part_name: str | None
    def preset(self, name: str) -> dict      # returns a Camera dict
    def render(self, camera: dict, width: int, height: int, options: dict, t: float = 0.0) -> np.ndarray  # BGR uint8
    def replay_steps(self, fps: int = 24, kind: str = "replay") -> list[dict]   # the cinematic, one step per frame
    def render_step(self, step: dict, width: int, height: int, options=None) -> np.ndarray
    def replay(self, width: int, height: int, options: dict, fps: int = 24) -> Iterator[np.ndarray]  # every step, BGR frames
    def close(self) -> None
```

All `LiveScene` calls except `recover_bodies` happen on **the one process-wide render thread**
(`sessions.RENDER_THREAD`). On Windows, opening an OpenGL window from any other thread fails once one exists,
so anything else that renders, video exports included, must run there too. `t` is seconds since the view opened, used for the slow pulse
on the defender's line-setting body part.
