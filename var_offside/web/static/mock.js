/**
 * Mock data source for the VAR Offside web UI.
 *
 * Exports `createMockApi()`, which implements exactly the subset of the HTTP and
 * WebSocket API that `app.js` uses. Everything is generated in the browser: the
 * frame is a painted pitch, debug stages are painted canvases and the 3D view is
 * a small perspective renderer. Used for `?mock=1`, or when `/api/health` is
 * unreachable, so the UI can be reviewed without the Python backend.
 */

const IMAGE_W = 1280;
const IMAGE_H = 720;

const TEAM_COLORS = { 1: "#e9e2d0", 2: "#3f6ad8" };
const RED = "#ff4a3d";
const GREEN = "#3ddc84";
const INK = "#eaf2ec";
const MUTED = "#8a9a90";

/** Resolve after `ms` milliseconds. */
function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Deep-ish copy for API-shaped objects. */
function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

/* ==========================================================================
   Mock players
   ========================================================================== */

/** Build the 19 detected people: nine light defenders, nine blue attackers, one referee. */
function buildPlayers() {
  return [
    // Team 1 (defending, light). #1 stands where a keeper would; the user marks them in Review.
    { id: 1, bbox: [1152, 372, 1192, 548], team: 1, role: "player", confidence: 0.97 },
    { id: 7, bbox: [776, 402, 812, 560], team: 1, role: "player", confidence: 0.93 },
    { id: 8, bbox: [712, 438, 746, 580], team: 1, role: "player", confidence: 0.91 },
    { id: 9, bbox: [648, 468, 680, 602], team: 1, role: "player", confidence: 0.89 },
    { id: 10, bbox: [902, 432, 938, 582], team: 1, role: "player", confidence: 0.94 },
    { id: 11, bbox: [966, 458, 1002, 600], team: 1, role: "player", confidence: 0.9 },
    { id: 12, bbox: [690, 338, 718, 438], team: 1, role: "player", confidence: 0.82 },
    { id: 13, bbox: [858, 348, 886, 452], team: 1, role: "player", confidence: 0.85 },
    { id: 14, bbox: [1022, 382, 1054, 494], team: 1, role: "player", confidence: 0.87 },

    // Team 2 (attacking right, blue). #2 is past the line.
    { id: 2, bbox: [850, 392, 888, 556], team: 2, role: "player", confidence: 0.95 },
    { id: 3, bbox: [778, 448, 812, 584], team: 2, role: "player", confidence: 0.92 },
    { id: 4, bbox: [722, 494, 756, 622], team: 2, role: "player", confidence: 0.9 },
    { id: 5, bbox: [642, 516, 674, 642], team: 2, role: "player", confidence: 0.88 },
    { id: 6, bbox: [560, 532, 592, 650], team: 2, role: "player", confidence: 0.91 },
    { id: 16, bbox: [98, 402, 136, 572], team: 2, role: "player", confidence: 0.96 },
    { id: 17, bbox: [492, 562, 524, 678], team: 2, role: "player", confidence: 0.86 },
    { id: 18, bbox: [408, 438, 438, 556], team: 2, role: "player", confidence: 0.84 },
    { id: 19, bbox: [352, 512, 382, 632], team: 2, role: "player", confidence: 0.83 },

    // Not a player: keep it in the mock to exercise the grey role.
    { id: 15, bbox: [600, 356, 628, 470], team: null, role: "referee", confidence: 0.78 },
  ];
}

const OFFSIDE_LINE = [
  [818, 236],
  [842, 720],
];

/** Latest reviewed player list; stage renders follow the user's edits. */
let activePlayers = buildPlayers();

/** What the user must still choose, with the same wording as the server. */
function choiceError(players, attackingTeam, keeper) {
  if (attackingTeam !== 1 && attackingTeam !== 2) return "Choose the attacking team.";
  if (keeper !== "player" && keeper !== "not_in_frame") {
    return 'Choose the keeper or "not in frame".';
  }
  const keepers = players.filter((player) => player.role === "goalkeeper").length;
  if (keeper === "player" && keepers !== 1) return "Choose exactly one player as the keeper.";
  if (keeper === "not_in_frame" && keepers) {
    return 'A keeper is marked, so "not in frame" conflicts with the player list. Choose the keeper or unmark them.';
  }
  return null;
}

/** Verdict body for a request the server would answer with "unknown". */
function unknownVerdict(side, message) {
  return {
    verdict: "unknown",
    estimate: false,
    message,
    margin_m: null,
    attacking_side: side,
    last_defender: { id: null, part: null },
    offside_player_ids: [],
    line_image_points: [],
    line_field_x: null,
    calibration: { method: "failed", full_3d: false, edited: false },
    checks: [
      { key: "players", label: "Players", status: "ok", detail: String(activePlayers.length) },
      { key: "teams", label: "Teams", status: "ok", detail: "2" },
      { key: "side", label: "Attack direction", status: "ok", detail: side === "left" ? "Left" : "Right" },
      { key: "calibration", label: "Calibration", status: "pending", detail: "" },
      { key: "defender", label: "Last defender", status: "check", detail: "not found" },
      { key: "verdict", label: "Verdict", status: "pending", detail: "Unknown" },
    ],
    stages: ["original", "detection", "teams"],
    time_ms: 0,
    view3d_available: false,
  };
}

/** Phase-1 response body. */
function phase1(players) {
  return {
    players: clone(players),
    team_colors: { 1: "#ff3347", 2: "#2e7dff" },
    checks: [
      { key: "players", label: "Players", status: "ok", detail: String(players.length) },
      { key: "teams", label: "Teams", status: "ok", detail: "2" },
      { key: "side", label: "Attack direction", status: "check", detail: "choose left or right" },
      { key: "calibration", label: "Calibration", status: "pending", detail: "" },
      { key: "defender", label: "Last defender", status: "pending", detail: "" },
      { key: "verdict", label: "Verdict", status: "pending", detail: "" },
    ],
    stages: ["original", "detection", "teams"],
    time_ms: 812,
  };
}

/* ==========================================================================
   Canvas painting helpers
   ========================================================================== */

/** Edge of the perspective pitch at vertical position `y`. */
function pitchEdges(y) {
  const t = (y - 205) / (IMAGE_H - 205);
  const left = 118 + (-80 - 118) * t;
  const right = 1162 + (1360 - 1162) * t;
  return [left, right];
}

/** Draw the generated broadcast-style pitch picture. */
function paintPitch() {
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_W;
  canvas.height = IMAGE_H;
  const g = canvas.getContext("2d");

  g.fillStyle = "#070c09";
  g.fillRect(0, 0, IMAGE_W, IMAGE_H);

  // Stands and advertising band.
  g.fillStyle = "#0e1712";
  g.fillRect(0, 0, IMAGE_W, 205);
  g.fillStyle = "#15231b";
  for (let x = -40; x < IMAGE_W; x += 46) {
    g.fillRect(x, 48, 22, 120);
  }
  g.fillStyle = "#1c2c22";
  g.fillRect(0, 182, IMAGE_W, 23);
  g.fillStyle = "#31473a";
  g.fillRect(0, 204, IMAGE_W, 3);

  // Mown wedges, wider near the camera.
  const wedges = 16;
  for (let i = 0; i < wedges; i += 1) {
    const xTop0 = 118 + ((1162 - 118) * i) / wedges;
    const xTop1 = 118 + ((1162 - 118) * (i + 1)) / wedges;
    const xBot0 = -80 + ((1360 + 80) * i) / wedges;
    const xBot1 = -80 + ((1360 + 80) * (i + 1)) / wedges;
    g.beginPath();
    g.moveTo(xTop0, 205);
    g.lineTo(xTop1, 205);
    g.lineTo(xBot1, IMAGE_H);
    g.lineTo(xBot0, IMAGE_H);
    g.closePath();
    g.fillStyle = i % 2 === 0 ? "#1d4a2c" : "#215230";
    g.fill();
  }

  // Subtle light falloff near the top.
  g.fillStyle = "rgba(4, 9, 6, 0.28)";
  g.fillRect(0, 205, IMAGE_W, 120);

  // Paint markings.
  g.strokeStyle = "rgba(226, 238, 229, 0.52)";
  g.lineWidth = 3;
  g.beginPath();
  g.moveTo(118, 205);
  g.lineTo(1162, 205);
  g.stroke();
  g.beginPath();
  g.moveTo(118, 205);
  g.lineTo(-80, IMAGE_H);
  g.stroke();
  g.beginPath();
  g.moveTo(1162, 205);
  g.lineTo(1360, IMAGE_H);
  g.stroke();

  // Halfway line.
  g.beginPath();
  g.moveTo(640, 205);
  g.lineTo(640, IMAGE_H);
  g.stroke();

  // Right penalty box (attacking goal) and six-yard box.
  g.beginPath();
  g.moveTo(1080, 264);
  g.lineTo(1162, 264);
  g.lineTo(1246, 612);
  g.lineTo(1100, 612);
  g.closePath();
  g.stroke();
  g.beginPath();
  g.moveTo(1120, 346);
  g.lineTo(1162, 346);
  g.lineTo(1200, 512);
  g.lineTo(1138, 512);
  g.closePath();
  g.stroke();

  // Centre circle, flattened by perspective.
  g.beginPath();
  for (let i = 0; i <= 48; i += 1) {
    const a = (i / 48) * Math.PI * 2;
    const x = 640 + Math.cos(a) * 118;
    const y = 470 + Math.sin(a) * 52;
    if (i === 0) g.moveTo(x, y);
    else g.lineTo(x, y);
  }
  g.stroke();

  // Goal frame at the right, mostly cropped by the touchline.
  g.strokeStyle = "rgba(236, 244, 238, 0.68)";
  g.lineWidth = 4;
  g.beginPath();
  g.moveTo(1204, 250);
  g.lineTo(1204, 430);
  g.lineTo(1248, 452);
  g.lineTo(1248, 286);
  g.closePath();
  g.stroke();

  // Players painted into the photo, so the overlay boxes visibly land on bodies.
  const painted = buildPlayers().sort((a, b) => a.bbox[3] - b.bbox[3]);
  for (const player of painted) {
    const [x1, y1, x2, y2] = player.bbox;
    const cx = (x1 + x2) / 2;
    const h = y2 - y1;
    const w = x2 - x1;
    const color =
      player.role === "referee"
        ? "#3a4740"
        : player.team === 1
          ? TEAM_COLORS[1]
          : TEAM_COLORS[2];

    // Shadow.
    g.fillStyle = "rgba(0, 0, 0, 0.3)";
    g.beginPath();
    g.ellipse(cx, y2 - 2, w * 0.7, Math.max(3, w * 0.18), 0, 0, Math.PI * 2);
    g.fill();

    // Legs.
    g.strokeStyle = player.role === "referee" ? "#202a24" : "#15211a";
    g.lineWidth = Math.max(3, w * 0.16);
    g.lineCap = "round";
    g.beginPath();
    g.moveTo(cx - w * 0.2, y2 - h * 0.38);
    g.lineTo(cx - w * 0.26, y2 - 2);
    g.moveTo(cx + w * 0.2, y2 - h * 0.38);
    g.lineTo(cx + w * 0.26, y2 - 2);
    g.stroke();

    // Torso.
    g.fillStyle = color;
    const torsoTop = y2 - h * 0.88;
    const torsoH = h * 0.5;
    const torsoW = w * 0.62;
    g.beginPath();
    if (typeof g.roundRect === "function") {
      g.roundRect(cx - torsoW / 2, torsoTop, torsoW, torsoH, torsoW * 0.28);
    } else {
      g.rect(cx - torsoW / 2, torsoTop, torsoW, torsoH);
    }
    g.fill();

    // Arms.
    g.strokeStyle = color;
    g.lineWidth = Math.max(2.5, w * 0.13);
    g.beginPath();
    g.moveTo(cx - torsoW * 0.42, torsoTop + torsoH * 0.14);
    g.lineTo(cx - torsoW * 0.66, torsoTop + torsoH * 0.85);
    g.moveTo(cx + torsoW * 0.42, torsoTop + torsoH * 0.14);
    g.lineTo(cx + torsoW * 0.66, torsoTop + torsoH * 0.85);
    g.stroke();

    // Head.
    g.fillStyle = "#c69a74";
    g.beginPath();
    g.arc(cx, y1 + h * 0.1, Math.max(3, w * 0.2), 0, Math.PI * 2);
    g.fill();

  }

  // Ball near the passer.
  g.fillStyle = "#f2f5ef";
  g.beginPath();
  g.arc(610, 664, 5.5, 0, Math.PI * 2);
  g.fill();
  g.strokeStyle = "#20291f";
  g.lineWidth = 1;
  g.stroke();

  return canvas;
}

let pitchCanvas = null;
let pitchDataUrl = null;

/** Cached painted pitch canvas. */
function getPitchCanvas() {
  if (!pitchCanvas) pitchCanvas = paintPitch();
  return pitchCanvas;
}

/** Cached data URL for the mock frame. */
function getPitchDataUrl() {
  if (!pitchDataUrl) pitchDataUrl = getPitchCanvas().toDataURL("image/jpeg", 0.86);
  return pitchDataUrl;
}

/** Draw one player box onto a stage canvas. */
function drawPlayerBox(g, player, label) {
  const [x1, y1, x2, y2] = player.bbox;
  const color =
    player.role === "referee"
      ? MUTED
      : player.team === 1
        ? TEAM_COLORS[1]
        : player.team === 2
          ? TEAM_COLORS[2]
          : MUTED;
  g.save();
  g.lineWidth = 2;
  g.strokeStyle = color;
  g.fillStyle = "rgba(0, 0, 0, 0.12)";
  if (player.role === "goalkeeper") g.setLineDash([6, 4]);
  g.fillRect(x1, y1, x2 - x1, y2 - y1);
  g.strokeRect(x1, y1, x2 - x1, y2 - y1);
  g.setLineDash([]);
  if (label) {
    const text = `${player.id ?? "new"}${player.role === "goalkeeper" ? " GK" : ""}`;
    g.font = "600 15px 'IBM Plex Mono', Consolas, monospace";
    const w = g.measureText(text).width + 8;
    g.fillStyle = "rgba(7, 12, 9, 0.82)";
    g.fillRect(x1, Math.max(0, y1 - 19), w, 17);
    g.fillStyle = color;
    g.fillText(text, x1 + 4, Math.max(12, y1 - 6));
  }
  g.restore();
}

/** Draw the offside line and the offside players' rings. */
function drawOffside(g, lineWidth) {
  g.save();
  g.strokeStyle = RED;
  g.lineWidth = lineWidth;
  g.shadowColor = "rgba(255, 90, 78, 0.8)";
  g.shadowBlur = 9;
  g.beginPath();
  g.moveTo(OFFSIDE_LINE[0][0], OFFSIDE_LINE[0][1]);
  g.lineTo(OFFSIDE_LINE[1][0], OFFSIDE_LINE[1][1]);
  g.stroke();
  g.restore();

  for (const player of activePlayers) {
    if (player.id !== 2) continue;
    const [x1, y1, x2, y2] = player.bbox;
    g.save();
    g.strokeStyle = RED;
    g.lineWidth = lineWidth;
    g.setLineDash([7, 5]);
    g.strokeRect(x1 - 7, y1 - 7, x2 - x1 + 14, y2 - y1 + 14);
    g.restore();
  }
}

/** Top-down pitch used by the bird's-eye stage. */
function drawTopDown(g, width, height) {
  g.fillStyle = "#0d1811";
  g.fillRect(0, 0, width, height);
  const margin = Math.min(width, height) * 0.07;
  const w = width - margin * 2;
  const h = height - margin * 2;
  g.fillStyle = "#1b4529";
  g.fillRect(margin, margin, w, h);
  g.strokeStyle = "rgba(230, 240, 232, 0.55)";
  g.lineWidth = 2;
  g.strokeRect(margin, margin, w, h);
  g.beginPath();
  g.moveTo(margin + w / 2, margin);
  g.lineTo(margin + w / 2, margin + h);
  g.stroke();
  g.beginPath();
  g.arc(margin + w / 2, margin + h / 2, h * 0.13, 0, Math.PI * 2);
  g.stroke();

  // Penalty boxes.
  g.strokeRect(margin, margin + h * 0.22, w * 0.14, h * 0.56);
  g.strokeRect(margin + w * 0.86, margin + h * 0.22, w * 0.14, h * 0.56);

  // Project the mock image players onto the pitch.
  for (const player of activePlayers) {
    const x = margin + (player.bbox[0] / IMAGE_W) * w;
    const y = margin + (player.bbox[1] / IMAGE_H) * h * 0.9 + h * 0.05;
    g.fillStyle =
      player.role === "referee"
        ? MUTED
        : player.team === 1
          ? TEAM_COLORS[1]
          : TEAM_COLORS[2];
    g.beginPath();
    g.arc(x, y, Math.max(3, width * 0.006), 0, Math.PI * 2);
    g.fill();
  }

  // Offside line, top-down.
  const lx = margin + (OFFSIDE_LINE[0][0] / IMAGE_W) * w;
  g.save();
  g.strokeStyle = RED;
  g.lineWidth = 3;
  g.shadowColor = "rgba(255, 90, 78, 0.75)";
  g.shadowBlur = 10;
  g.beginPath();
  g.moveTo(lx, margin);
  g.lineTo(lx, margin + h);
  g.stroke();
  g.restore();

  g.fillStyle = INK;
  g.font = "600 20px 'Barlow Condensed', 'Arial Narrow', sans-serif";
  g.fillText("BIRD'S EYE", margin + 10, margin + 26);
}

/* ==========================================================================
   Stage images
   ========================================================================== */

const stageCache = new Map();

/** Render (and cache) one debug stage as a data URL. */
function stageDataUrl(key) {
  if (stageCache.has(key)) return stageCache.get(key);

  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_W;
  canvas.height = IMAGE_H;
  const g = canvas.getContext("2d");
  g.drawImage(getPitchCanvas(), 0, 0);

  if (key === "detection" || key === "teams" || key === "result") {
    for (const player of activePlayers) drawPlayerBox(g, player, true);
  }

  if (key === "calibration") {
    g.save();
    g.strokeStyle = "rgba(79, 194, 122, 0.55)";
    g.lineWidth = 1;
    for (let y = 220; y < IMAGE_H; y += 48) {
      const [l, r] = pitchEdges(y);
      g.beginPath();
      g.moveTo(l, y);
      g.lineTo(r, y);
      g.stroke();
    }
    for (let x = 0; x < IMAGE_W; x += 80) {
      g.beginPath();
      g.moveTo(x, 205);
      g.lineTo(x - 80, IMAGE_H);
      g.stroke();
    }
    g.restore();
    g.fillStyle = GREEN;
    g.font = "600 20px 'IBM Plex Mono', Consolas, monospace";
    g.fillText("PnL · full 3D · residual 0.8 px", 28, 42);
  }

  if (key === "offside" || key === "result") {
    drawOffside(g, 3);
  }

  if (key === "result") {
    g.fillStyle = "rgba(7, 12, 9, 0.82)";
    g.fillRect(28, 28, 330, 74);
    g.fillStyle = RED;
    g.font = "700 40px 'Barlow Condensed', 'Arial Narrow', sans-serif";
    g.fillText("OFFSIDE", 44, 66);
    g.fillStyle = INK;
    g.font = "400 16px 'IBM Plex Mono', Consolas, monospace";
    g.fillText("attacker ahead by 0.84 m", 44, 92);
  }

  if (key === "birdseye") {
    drawTopDown(g, IMAGE_W, IMAGE_H);
  }

  if (key === "original") {
    // The pitch itself.
  }

  const url = canvas.toDataURL("image/png");
  stageCache.set(key, url);
  return url;
}

/* ==========================================================================
   Jobs
   ========================================================================== */

const jobs = new Map();
const JOB_DURATION = {
  image: 1500,
  replay: 3800,
  birdseye: 3400,
  debug: 2800,
  report: 2000,
};
let jobSeq = 0;

/** Small downloadable file for a finished mock export. */
function exportFileUrl(kind) {
  if (kind === "report") {
    const body = [
      "VAR OFFSIDE — MOCK REPORT",
      "",
      "Verdict: OFFSIDE",
      "Margin: +0.84 m",
      "Attacking side: right",
      "Last defender: #7 (left heel)",
      "Offside player: #2",
      "",
      "This file is generated by mock.js; no backend was contacted.",
    ].join("\n");
    return `data:text/plain;charset=utf-8,${encodeURIComponent(body)}`;
  }
  return stageDataUrl(kind === "birdseye" ? "birdseye" : "result");
}

/** Create a fake export job. */
function startJob(kind) {
  jobSeq += 1;
  const id = `mock-job-${jobSeq}`;
  jobs.set(id, {
    kind,
    start: performance.now(),
    duration: JOB_DURATION[kind] || 2000,
    status: "running",
  });
  return id;
}

/** Read a fake export job, advancing it from elapsed time. */
function readJob(id) {
  const job = jobs.get(id);
  if (!job) {
    return { status: "error", progress: 0, message: "Unknown job", file_url: null };
  }
  if (job.status === "cancelled") {
    return { status: "cancelled", progress: 0, message: "Cancelled", file_url: null };
  }
  const elapsed = performance.now() - job.start;
  const progress = Math.min(100, Math.round((elapsed / job.duration) * 100));
  if (progress >= 100 && job.status === "running") job.status = "done";
  const stageMessages = {
    image: "Drawing overlay",
    replay: "Encoding replay",
    birdseye: "Rendering bird's-eye",
    debug: "Encoding debug video",
    report: "Packing report",
  };
  return {
    status: job.status,
    progress,
    message: job.status === "done" ? "Ready" : stageMessages[job.kind] || "Working",
    file_url: job.status === "done" ? exportFileUrl(job.kind) : null,
  };
}

/* ==========================================================================
   Fake live 3D view
   ========================================================================== */

const PRESET_CAMERAS = {
  broadcast: { target: [0, 0, 0], yaw: 35, pitch: 22, distance: 18, fov: 40 },
  birdseye: { target: [0, 0, 0], yaw: 0, pitch: 87, distance: 42, fov: 40 },
  line: { target: [1.2, 0, 0], yaw: 90, pitch: 8, distance: 14, fov: 40 },
  free: { target: [0, 0, 0], yaw: 55, pitch: 30, distance: 24, fov: 40 },
};

/* Scene layout (metres): X runs along the pitch towards the goal being
   attacked, Z across it, Y up. The halfway line sits at X = PITCH_MID so the
   offside incident lands near the scene origin, where the cameras look. */
const PITCH_HALF_L = 52.5;
const PITCH_HALF_W = 34;
const PITCH_MID = -20;
const GOAL_X = PITCH_MID + PITCH_HALF_L;
const SCENE_LINE_X = 1.2;

/** Stand-in player positions in scene metres (ids match the mock verdict). */
function scenePlayers() {
  const rows = [
    // Team 1, light kit, defending the goal at +X. #7 sets the line.
    [GOAL_X - 4, 1.2, 1, "goalkeeper", 1],
    [SCENE_LINE_X, -3.4, 1, "player", 7],
    [0.4, 4.6, 1, "player", 8],
    [-0.8, 11.2, 1, "player", 9],
    [0.1, -11.4, 1, "player", 10],
    [-6.4, 1.8, 1, "player", 11],
    [-5.2, -7.2, 1, "player", 12],
    [-7.5, 8.4, 1, "player", 13],
    [-10.4, -2.4, 1, "player", 14],
    // Team 2, blue kit, attacking towards +X. #2 is beyond the line.
    [2.04, -1.2, 2, "player", 2],
    [-2.2, 6.8, 2, "player", 3],
    [-3.6, -9.4, 2, "player", 4],
    [-8.4, 4.2, 2, "player", 5],
    [-12.2, -5.6, 2, "player", 6],
    [PITCH_MID - PITCH_HALF_L + 5, 0, 2, "goalkeeper", 16],
    [-14.6, 9.6, 2, "player", 17],
    [-5.6, 14.2, 2, "player", 18],
    [-16.4, -11.2, 2, "player", 19],
    // Referee.
    [-9.2, -15.4, 0, "referee", 15],
  ];
  return rows.map(([x, z, team, role, id]) => ({
    x,
    z,
    team: team === 0 ? null : team,
    role,
    id,
  }));
}

const OFFSIDE_ID = 2;
const DEFENDER_ID = 7;

const KIT = {
  1: { shirt: "#ece6d6", shade: "#b9b19c", shorts: "#d9d2bf", socks: "#ece6d6" },
  2: { shirt: "#3f6ad8", shade: "#26459b", shorts: "#15255a", socks: "#3f6ad8" },
  gk1: { shirt: "#d7e04c", shade: "#9ea533", shorts: "#2a2f1a", socks: "#d7e04c" },
  gk2: { shirt: "#ff8a3d", shade: "#bb5e22", shorts: "#2b1a10", socks: "#ff8a3d" },
  ref: { shirt: "#1d2622", shade: "#0e1411", shorts: "#0e1411", socks: "#1d2622" },
};
const SKIN = "#c89a77";

/** A tiny orbit-camera perspective renderer, driven by camera messages. */
class MockView {
  constructor(options) {
    this.opts = options || {};
    this.canvas = this.opts.canvas || null;
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.camera = clone(PRESET_CAMERAS.broadcast);
    this.width = 1280;
    this.height = 720;
    this.options = { line_plane: true, photo: false, labels: true, pulse: true };
    this.closed = false;
    this.replayTimer = null;
    this.replayStart = 0;
    this.timers = [];
    this.people = scenePlayers();

    this.timers.push(
      setTimeout(() => {
        if (this.closed) return;
        if (this.opts.onOpen) this.opts.onOpen();
        this.status("loading", "Loading 3D…");
        this.timers.push(
          setTimeout(() => this.status("proxy", "Body meshes 1/4"), 420),
        );
        this.timers.push(
          setTimeout(() => this.status("meshes", "Body meshes 3/4"), 1000),
        );
        this.timers.push(
          setTimeout(() => this.status("ready", "Body meshes 4/4"), 1700),
        );
        this.draw();
      }, 30),
    );
  }

  /** Emit a status message to app.js. */
  status(stage, message) {
    if (!this.closed && this.opts.onStatus) this.opts.onStatus(stage, message);
  }

  /** Accept a camera update from the app and redraw. */
  sendCamera(camera, width, height, quality, options) {
    if (this.closed) return;
    if (camera) this.camera = clone(camera);
    if (width) this.width = width;
    if (height) this.height = height;
    if (options) this.options = options;
    this.draw();
  }

  /** Jump to a named preset, then report the new camera to the app. */
  sendPreset(name, width, height, options) {
    if (this.closed) return;
    if (width) this.width = width;
    if (height) this.height = height;
    if (options) this.options = options;
    const preset = PRESET_CAMERAS[name] || PRESET_CAMERAS.free;
    this.camera = clone(preset);
    this.draw();
    if (this.opts.onCamera) this.opts.onCamera(clone(this.camera));
  }

  /** Stream ~24 mock replay frames, then report the final camera. */
  sendReplay(width, height, options) {
    if (this.closed) return;
    this.stopReplay();
    if (width) this.width = width;
    if (height) this.height = height;
    if (options) this.options = options;
    const base = clone(this.camera);
    this.replayStart = performance.now();
    this.replayTimer = setInterval(() => {
      if (this.closed) {
        this.stopReplay();
        return;
      }
      const t = (performance.now() - this.replayStart) / 1000;
      if (t >= 3.8) {
        this.stopReplay();
        this.camera = {
          target: clone(base.target),
          yaw: base.yaw + 18,
          pitch: 24,
          distance: 20,
          fov: base.fov,
        };
        this.draw();
        if (this.opts.onCamera) this.opts.onCamera(clone(this.camera));
        if (this.opts.onReplayEnd) this.opts.onReplayEnd();
        return;
      }
      this.camera = {
        target: clone(base.target),
        yaw: base.yaw + t * 20,
        pitch: Math.max(3, Math.min(89, 22 + Math.sin(t * 0.9) * 9)),
        distance: 18 + Math.sin(t * 0.7) * 4,
        fov: base.fov,
      };
      this.draw();
    }, 42);
  }

  /** Stop a running mock replay. */
  sendStop() {
    this.stopReplay();
  }

  /** Clear the replay interval. */
  stopReplay() {
    if (this.replayTimer) {
      clearInterval(this.replayTimer);
      this.replayTimer = null;
    }
  }

  /** Tear down timers; explicit closes do not emit `onClose`. */
  close() {
    this.closed = true;
    this.stopReplay();
    for (const timer of this.timers) clearTimeout(timer);
    this.timers = [];
  }

  /** Build an orbit camera basis from the contract's yaw/pitch/distance/fov. */
  basis() {
    const yaw = (this.camera.yaw * Math.PI) / 180;
    const pitch = (this.camera.pitch * Math.PI) / 180;
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    const target = this.camera.target || [0, 0, 0];
    const pos = [
      target[0] + this.camera.distance * cp * Math.sin(yaw),
      target[1] + this.camera.distance * sp,
      target[2] + this.camera.distance * cp * Math.cos(yaw),
    ];
    const fwdRaw = [target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]];
    const fwdLen = Math.hypot(...fwdRaw) || 1;
    const fwd = fwdRaw.map((v) => v / fwdLen);
    const rightRaw = [-fwd[2], 0, fwd[0]];
    const rightLen = Math.hypot(...rightRaw) || 1;
    const right = rightRaw.map((v) => v / rightLen);
    const up = [
      right[1] * fwd[2] - right[2] * fwd[1],
      right[2] * fwd[0] - right[0] * fwd[2],
      right[0] * fwd[1] - right[1] * fwd[0],
    ];
    return { pos, fwd, right, up };
  }

  /** World point to camera space [x right, y up, z forward]. */
  toCam(point) {
    const b = this.frame.basis;
    const rx = point[0] - b.pos[0];
    const ry = point[1] - b.pos[1];
    const rz = point[2] - b.pos[2];
    return [
      rx * b.right[0] + ry * b.right[1] + rz * b.right[2],
      rx * b.up[0] + ry * b.up[1] + rz * b.up[2],
      rx * b.fwd[0] + ry * b.fwd[1] + rz * b.fwd[2],
    ];
  }

  /** Camera-space point to screen pixels. */
  camToScreen(c) {
    const { cx, cy, focal } = this.frame;
    return { x: cx + (focal * c[0]) / c[2], y: cy - (focal * c[1]) / c[2], z: c[2] };
  }

  /** Project a world point; null when it is behind the near plane. */
  p(point) {
    const c = this.toCam(point);
    if (c[2] < 0.3) return null;
    return this.camToScreen(c);
  }

  /** Pixels per metre at a given camera depth. */
  ppm(depth) {
    return this.frame.focal / Math.max(depth, 0.3);
  }

  /** Trace a world polygon, clipped against the near plane. Returns false if empty. */
  tracePoly(points) {
    const near = 0.3;
    const cam = points.map((pt) => this.toCam(pt));
    const out = [];
    for (let i = 0; i < cam.length; i += 1) {
      const a = cam[i];
      const b = cam[(i + 1) % cam.length];
      const ain = a[2] >= near;
      const bin = b[2] >= near;
      if (ain) out.push(a);
      if (ain !== bin) {
        const t = (near - a[2]) / (b[2] - a[2]);
        out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near]);
      }
    }
    if (out.length < 3) return false;
    const g = this.ctx;
    g.beginPath();
    out.forEach((c, i) => {
      const s = this.camToScreen(c);
      if (i === 0) g.moveTo(s.x, s.y);
      else g.lineTo(s.x, s.y);
    });
    g.closePath();
    return true;
  }

  /** Stroke a world polyline, clipping each segment to the near plane. */
  strokePath(points, closed = false) {
    const near = 0.3;
    const g = this.ctx;
    const count = closed ? points.length : points.length - 1;
    g.beginPath();
    for (let i = 0; i < count; i += 1) {
      let a = this.toCam(points[i]);
      let b = this.toCam(points[(i + 1) % points.length]);
      if (a[2] < near && b[2] < near) continue;
      if (a[2] < near || b[2] < near) {
        const t = (near - a[2]) / (b[2] - a[2]);
        const m = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near];
        if (a[2] < near) a = m;
        else b = m;
      }
      const sa = this.camToScreen(a);
      const sb = this.camToScreen(b);
      g.moveTo(sa.x, sa.y);
      g.lineTo(sb.x, sb.y);
    }
    g.stroke();
  }

  /** Ground circle or arc as world points. */
  arc(cx, cz, r, a0 = 0, a1 = Math.PI * 2, steps = 48) {
    const pts = [];
    for (let i = 0; i <= steps; i += 1) {
      const a = a0 + ((a1 - a0) * i) / steps;
      pts.push([cx + Math.cos(a) * r, 0, cz + Math.sin(a) * r]);
    }
    return pts;
  }

  /** Background, grass, markings and goals. */
  drawPitch(w, h) {
    const g = this.ctx;
    const b = this.frame.basis;

    // Sky / stands, with two soft floodlight blooms near the horizon.
    const flat = Math.hypot(b.fwd[0], b.fwd[2]) || 1;
    const far = [
      b.pos[0] + (b.fwd[0] / flat) * 4000,
      0,
      b.pos[2] + (b.fwd[2] / flat) * 4000,
    ];
    const horizon = this.p(far);
    const hy = horizon ? horizon.y : -h;
    const sky = g.createLinearGradient(0, Math.min(hy, h) - h * 0.5, 0, Math.max(hy, 0));
    sky.addColorStop(0, "#020403");
    sky.addColorStop(1, "#0c1712");
    g.fillStyle = sky;
    g.fillRect(0, 0, w, h);
    if (hy > 0) {
      for (const fx of [0.22, 0.78]) {
        const bloom = g.createRadialGradient(w * fx, hy - h * 0.05, 0, w * fx, hy - h * 0.05, w * 0.32);
        bloom.addColorStop(0, "rgba(214, 255, 232, 0.10)");
        bloom.addColorStop(1, "rgba(214, 255, 232, 0)");
        g.fillStyle = bloom;
        g.fillRect(0, 0, w, h);
      }
      // Stand silhouette just above the horizon.
      const stand = g.createLinearGradient(0, hy - h * 0.07, 0, hy);
      stand.addColorStop(0, "rgba(10, 16, 13, 0)");
      stand.addColorStop(1, "rgba(14, 22, 18, 0.95)");
      g.fillStyle = stand;
      g.fillRect(0, hy - h * 0.07, w, h * 0.07);
    }

    // Surround.
    if (this.tracePoly([[-400, 0, -400], [400, 0, -400], [400, 0, 400], [-400, 0, 400]])) {
      g.fillStyle = "#0d2619";
      g.fill();
    }

    // Mown stripes.
    const x0 = PITCH_MID - PITCH_HALF_L;
    const x1 = PITCH_MID + PITCH_HALF_L;
    const z0 = -PITCH_HALF_W;
    const z1 = PITCH_HALF_W;
    const strips = 20;
    for (let i = 0; i < strips; i += 1) {
      const ax = x0 + ((x1 - x0) * i) / strips;
      const bx = x0 + ((x1 - x0) * (i + 1)) / strips;
      if (this.tracePoly([[ax, 0, z0], [bx, 0, z0], [bx, 0, z1], [ax, 0, z1]])) {
        g.fillStyle = i % 2 === 0 ? "#1a5534" : "#1f6240";
        g.fill();
      }
    }

    // Floodlit centre, darker edges.
    const glow = g.createRadialGradient(w * 0.5, h * 0.62, 0, w * 0.5, h * 0.62, Math.max(w, h) * 0.7);
    glow.addColorStop(0, "rgba(200, 255, 220, 0.07)");
    glow.addColorStop(1, "rgba(0, 0, 0, 0.28)");
    g.fillStyle = glow;
    g.fillRect(0, 0, w, h);

    // Markings.
    g.save();
    g.strokeStyle = "rgba(240, 248, 242, 0.78)";
    g.lineWidth = Math.max(1.2, w * 0.0014);
    g.lineCap = "round";
    const box = (xa, xb, half) =>
      this.strokePath([[xa, 0, -half], [xb, 0, -half], [xb, 0, half], [xa, 0, half]], false);
    this.strokePath([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], true);
    this.strokePath([[PITCH_MID, 0, z0], [PITCH_MID, 0, z1]]);
    this.strokePath(this.arc(PITCH_MID, 0, 9.15));
    for (const dir of [1, -1]) {
      const goal = PITCH_MID + dir * PITCH_HALF_L;
      box(goal, goal - dir * 16.5, 20.16);
      box(goal, goal - dir * 5.5, 9.16);
      const spot = goal - dir * 11;
      const reach = Math.acos(5.5 / 9.15);
      const base = dir === 1 ? Math.PI : 0;
      this.strokePath(this.arc(spot, 0, 9.15, base - reach, base + reach, 24));
      g.fillStyle = "rgba(240, 248, 242, 0.85)";
      if (this.tracePoly(this.arc(spot, 0, 0.2, 0, Math.PI * 2, 12))) g.fill();
      // Corner arcs, each turning into the field of play.
      for (const side of [z0, z1]) {
        let a0;
        if (dir === 1) a0 = side === z0 ? Math.PI / 2 : Math.PI;
        else a0 = side === z0 ? 0 : -Math.PI / 2;
        this.strokePath(this.arc(goal, side, 1, a0, a0 + Math.PI / 2, 8));
      }
    }
    if (this.tracePoly(this.arc(PITCH_MID, 0, 0.25, 0, Math.PI * 2, 12))) g.fill();
    g.restore();

    // Goal frames.
    g.save();
    g.strokeStyle = "rgba(250, 252, 250, 0.92)";
    g.lineWidth = Math.max(1.5, w * 0.0022);
    g.lineCap = "round";
    for (const dir of [1, -1]) {
      const gx = PITCH_MID + dir * PITCH_HALF_L;
      const back = gx + dir * 2;
      this.strokePath([[gx, 0, -3.66], [gx, 2.44, -3.66], [gx, 2.44, 3.66], [gx, 0, 3.66]]);
      g.strokeStyle = "rgba(250, 252, 250, 0.3)";
      g.lineWidth = Math.max(1, w * 0.001);
      this.strokePath([[gx, 2.44, -3.66], [back, 0, -3.66], [back, 0, 3.66], [gx, 2.44, 3.66]]);
      g.strokeStyle = "rgba(250, 252, 250, 0.92)";
      g.lineWidth = Math.max(1.5, w * 0.0022);
    }
    g.restore();

    // Distance haze toward the horizon.
    if (hy > -h * 0.2) {
      const haze = g.createLinearGradient(0, hy, 0, hy + h * 0.32);
      haze.addColorStop(0, "rgba(9, 16, 12, 0.92)");
      haze.addColorStop(1, "rgba(9, 16, 12, 0)");
      g.fillStyle = haze;
      g.fillRect(0, Math.max(0, hy - 2), w, h * 0.34);
    }
  }

  /** Ground-level offside line, margin wedge and (optionally) the vertical plane. */
  drawLineGround(w) {
    const g = this.ctx;
    const z0 = -PITCH_HALF_W;
    const z1 = PITCH_HALF_W;
    const attacker = this.people.find((person) => person.id === OFFSIDE_ID);

    // Margin wedge: the gap the attacker is beyond the line.
    if (attacker) {
      const za = attacker.z - 2.4;
      const zb = attacker.z + 2.4;
      if (this.tracePoly([[SCENE_LINE_X, 0.01, za], [attacker.x, 0.01, za], [attacker.x, 0.01, zb], [SCENE_LINE_X, 0.01, zb]])) {
        g.fillStyle = "rgba(255, 74, 61, 0.28)";
        g.fill();
      }
      g.save();
      g.strokeStyle = "rgba(255, 210, 205, 0.7)";
      g.lineWidth = Math.max(1, w * 0.001);
      g.setLineDash([w * 0.004, w * 0.004]);
      this.strokePath([[attacker.x, 0.01, za], [attacker.x, 0.01, zb]]);
      g.restore();
    }

    g.save();
    g.lineCap = "round";
    g.strokeStyle = "rgba(255, 74, 61, 0.28)";
    g.lineWidth = Math.max(4, w * 0.008);
    this.strokePath([[SCENE_LINE_X, 0.01, z0], [SCENE_LINE_X, 0.01, z1]]);
    g.strokeStyle = "#ff4a3d";
    g.lineWidth = Math.max(1.5, w * 0.0022);
    g.shadowColor = "rgba(255, 74, 61, 0.9)";
    g.shadowBlur = w * 0.008;
    this.strokePath([[SCENE_LINE_X, 0.01, z0], [SCENE_LINE_X, 0.01, z1]]);
    g.restore();
  }

  /** Translucent vertical offside plane. */
  drawPlane(w) {
    const g = this.ctx;
    const z0 = -PITCH_HALF_W;
    const z1 = PITCH_HALF_W;
    const top = 2.3;
    if (!this.tracePoly([[SCENE_LINE_X, 0, z0], [SCENE_LINE_X, 0, z1], [SCENE_LINE_X, top, z1], [SCENE_LINE_X, top, z0]])) return;
    const lo = this.p([SCENE_LINE_X, 0, 0]);
    const hi = this.p([SCENE_LINE_X, top, 0]);
    let fill = "rgba(255, 74, 61, 0.08)";
    if (lo && hi && Math.abs(lo.y - hi.y) > 2) {
      const grad = g.createLinearGradient(0, lo.y, 0, hi.y);
      grad.addColorStop(0, "rgba(255, 74, 61, 0.15)");
      grad.addColorStop(1, "rgba(255, 74, 61, 0.02)");
      fill = grad;
    }
    g.fillStyle = fill;
    g.fill();
    g.save();
    g.strokeStyle = "rgba(255, 120, 110, 0.35)";
    g.lineWidth = Math.max(1, w * 0.001);
    this.strokePath([[SCENE_LINE_X, top, z0], [SCENE_LINE_X, top, z1]]);
    g.restore();
  }

  /** One stand-in player: perspective shadow, legs, shorts, shirt, arms, head. */
  drawPerson(person, w) {
    const g = this.ctx;
    const foot = this.toCam([person.x, 0, person.z]);
    if (foot[2] < 0.6) return;
    const r = this.frame.basis.right;
    const at = (lateral, y) => [person.x + r[0] * lateral, y, person.z + r[2] * lateral];
    const kit =
      person.role === "referee"
        ? KIT.ref
        : person.role === "goalkeeper"
          ? KIT[person.team === 1 ? "gk1" : "gk2"]
          : KIT[person.team] || KIT.ref;
    const scale = this.ppm(foot[2]);
    const stroke = (a, b, width, color) => {
      const pa = this.p(a);
      const pb = this.p(b);
      if (!pa || !pb) return;
      g.strokeStyle = color;
      g.lineWidth = Math.max(1, width * scale);
      g.beginPath();
      g.moveTo(pa.x, pa.y);
      g.lineTo(pb.x, pb.y);
      g.stroke();
    };

    const highlight =
      person.id === OFFSIDE_ID ? "#ff4a3d" : person.id === DEFENDER_ID ? "#f2f6f3" : null;

    // Ground ring for the two players that decide the call.
    if (highlight) {
      g.save();
      if (this.tracePoly(this.arc(person.x, person.z, 0.95, 0, Math.PI * 2, 32))) {
        g.fillStyle = person.id === OFFSIDE_ID ? "rgba(255, 74, 61, 0.22)" : "rgba(242, 246, 243, 0.12)";
        g.fill();
      }
      g.strokeStyle = highlight;
      g.lineWidth = Math.max(1.5, w * 0.002);
      g.shadowColor = highlight;
      g.shadowBlur = w * 0.006;
      this.strokePath(this.arc(person.x, person.z, 0.95, 0, Math.PI * 2, 32), true);
      g.restore();
    }

    // Soft contact shadow, offset away from the floodlights.
    g.save();
    if (this.tracePoly(this.arc(person.x + 0.25, person.z + 0.2, 0.55, 0, Math.PI * 2, 20))) {
      g.fillStyle = "rgba(0, 0, 0, 0.42)";
      g.shadowColor = "rgba(0, 0, 0, 0.5)";
      g.shadowBlur = Math.max(2, 0.25 * scale);
      g.fill();
    }
    g.restore();

    g.save();
    g.lineCap = "round";
    g.lineJoin = "round";
    // Legs + socks.
    stroke(at(-0.11, 0.05), at(-0.12, 0.5), 0.13, kit.socks);
    stroke(at(0.11, 0.05), at(0.13, 0.5), 0.13, kit.socks);
    stroke(at(-0.12, 0.45), at(-0.11, 0.86), 0.15, SKIN);
    stroke(at(0.13, 0.45), at(0.11, 0.86), 0.15, SKIN);
    // Shorts.
    stroke(at(-0.1, 0.78), at(-0.1, 0.98), 0.24, kit.shorts);
    stroke(at(0.1, 0.78), at(0.1, 0.98), 0.24, kit.shorts);
    // Arms.
    stroke(at(-0.27, 1.42), at(-0.34, 0.98), 0.11, kit.shade);
    stroke(at(0.27, 1.42), at(0.34, 0.98), 0.11, kit.shade);
    // Torso with a shaded flank.
    stroke(at(0, 1.02), at(0, 1.42), 0.46, kit.shirt);
    stroke(at(0.15, 1.04), at(0.15, 1.4), 0.12, kit.shade);
    g.restore();

    // Head.
    const head = this.p([person.x, 1.62, person.z]);
    if (head) {
      g.fillStyle = SKIN;
      g.beginPath();
      g.arc(head.x, head.y, Math.max(1.2, 0.12 * scale), 0, Math.PI * 2);
      g.fill();
      g.fillStyle = "#2a1d14";
      g.beginPath();
      g.arc(head.x, head.y - 0.03 * scale, Math.max(1, 0.12 * scale), Math.PI, Math.PI * 2);
      g.fill();
    }

    // Label plate.
    if (this.options.labels !== false) {
      const tag = this.p([person.x, 2.02, person.z]);
      if (!tag) return;
      const fontPx = Math.max(10, Math.round(w * 0.0085));
      let text = `#${person.id}`;
      if (person.id === OFFSIDE_ID) text += "  OFFSIDE";
      else if (person.id === DEFENDER_ID) text += "  LAST DEFENDER";
      g.save();
      g.font = `600 ${fontPx}px 'IBM Plex Mono', Consolas, monospace`;
      const tw = g.measureText(text).width;
      const padX = fontPx * 0.55;
      const bw = tw + padX * 2 + 3;
      const bh = fontPx * 1.6;
      const bx = tag.x - bw / 2;
      const by = tag.y - bh;
      g.fillStyle = "rgba(5, 9, 7, 0.82)";
      g.beginPath();
      if (g.roundRect) g.roundRect(bx, by, bw, bh, 3);
      else g.rect(bx, by, bw, bh);
      g.fill();
      g.fillStyle = highlight || (person.role === "referee" ? "#84978c" : kit.shirt);
      g.fillRect(bx, by, 3, bh);
      g.fillStyle = person.id === OFFSIDE_ID ? "#ffb3ad" : "rgba(234, 242, 236, 0.95)";
      g.textBaseline = "middle";
      g.fillText(text, bx + 3 + padX, by + bh / 2 + 0.5);
      g.restore();
    }
  }

  /** Paint the whole mock 3D frame. */
  draw() {
    if (this.closed || !this.ctx) return;
    const g = this.ctx;
    const w = Math.max(1, Math.round(this.width));
    const h = Math.max(1, Math.round(this.height));
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;

    const basis = this.basis();
    const focal = h / (2 * Math.tan((this.camera.fov * Math.PI) / 180 / 2));
    this.frame = { basis, focal, cx: w / 2, cy: h / 2 };

    g.setLineDash([]);
    this.drawPitch(w, h);
    this.drawLineGround(w);

    // Players far to near, split around the plane so it sorts correctly.
    const camSide = Math.sign(basis.pos[0] - SCENE_LINE_X) || 1;
    const sorted = this.people
      .map((person) => ({ person, depth: this.toCam([person.x, 0, person.z])[2] }))
      .sort((a, b) => b.depth - a.depth);
    const behind = sorted.filter(({ person }) => Math.sign(person.x - SCENE_LINE_X) !== camSide);
    const front = sorted.filter(({ person }) => Math.sign(person.x - SCENE_LINE_X) === camSide);

    for (const { person } of behind) this.drawPerson(person, w);
    if (this.options.line_plane !== false) this.drawPlane(w);
    for (const { person } of front) this.drawPerson(person, w);
  }
}

/* ==========================================================================
   Mock calibration
   ========================================================================== */

const MOCK_TRUE_POINTS = [
  [[0, 0], [118, 205]],
  [[105, 0], [1162, 205]],
  [[105, 68], [1360, 720]],
  [[0, 68], [-80, 720]],
];
const MOCK_AUTO_NUDGE = [[12, -7], [-14, 8], [9, -11], [-10, 12]];

/** Weighted least-squares homography, field metres to image pixels. */
function mockHomography(rows) {
  const n = 8;
  const A = Array.from({ length: n }, () => new Array(n).fill(0));
  const b = new Array(n).fill(0);
  const add = (row, target, weight) => {
    for (let p = 0; p < n; p += 1) {
      b[p] += weight * weight * row[p] * target;
      for (let q = 0; q < n; q += 1) A[p][q] += weight * weight * row[p] * row[q];
    }
  };
  for (const { x, y, u, v, w } of rows) {
    const fx = (x - 52.5) / 52.5;
    const fy = (y - 34) / 34;
    const iu = (u - IMAGE_W / 2) / (IMAGE_W / 2);
    const iv = (v - IMAGE_H / 2) / (IMAGE_H / 2);
    add([fx, fy, 1, 0, 0, 0, -iu * fx, -iu * fy], iu, w);
    add([0, 0, 0, fx, fy, 1, -iv * fx, -iv * fy], iv, w);
  }
  for (let i = 0; i < n; i += 1) {
    let pivot = i;
    for (let j = i + 1; j < n; j += 1) if (Math.abs(A[j][i]) > Math.abs(A[pivot][i])) pivot = j;
    if (Math.abs(A[pivot][i]) < 1e-11) return null;
    [A[i], A[pivot]] = [A[pivot], A[i]];
    [b[i], b[pivot]] = [b[pivot], b[i]];
    for (let j = i + 1; j < n; j += 1) {
      const f = A[j][i] / A[i][i];
      for (let k = i; k < n; k += 1) A[j][k] -= f * A[i][k];
      b[j] -= f * b[i];
    }
  }
  const h = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i -= 1) {
    let sum = b[i];
    for (let k = i + 1; k < n; k += 1) sum -= A[i][k] * h[k];
    h[i] = sum / A[i][i];
  }
  const sx = IMAGE_W / 2;
  const sy = IMAGE_H / 2;
  const fx = 1 / 52.5;
  const fy = 1 / 34;
  const H = [
    [sx * h[0] + sx * h[6], sx * h[1] + sx * h[7], sx * h[2] + sx],
    [sy * h[3] + sy * h[6], sy * h[4] + sy * h[7], sy * h[5] + sy],
    [h[6], h[7], 1],
  ];
  return H.map((row) => [row[0] * fx, row[1] * fy, row[2] - row[0] - row[1]]);
}

const MOCK_AUTO_H = mockHomography(
  MOCK_TRUE_POINTS.map(([field, pixel], index) => ({
    x: field[0],
    y: field[1],
    u: pixel[0] + MOCK_AUTO_NUDGE[index][0],
    v: pixel[1] + MOCK_AUTO_NUDGE[index][1],
    w: 1,
  })),
);

/** A small pitch model for the mock: one end plus the halfway line. */
function mockPitch() {
  const mark = (id, name, x, y) => ({ id, name, x, y });
  const line = (id, name, a, b) => ({ id, name, a, b });
  const landmarks = [];
  for (const [code, side, goal, sign] of [["r", "Right", 105, -1], ["l", "Left", 0, 1]]) {
    landmarks.push(
      mark(`${code}_box_front_far`, `${side} penalty box, far front corner`, goal + sign * 16.5, 13.84),
      mark(`${code}_box_front_near`, `${side} penalty box, near front corner`, goal + sign * 16.5, 54.16),
      mark(`${code}_box_goal_far`, `${side} penalty box meets goal line, far side`, goal, 13.84),
      mark(`${code}_box_goal_near`, `${side} penalty box meets goal line, near side`, goal, 54.16),
      mark(`${code}_six_front_far`, `${side} 6-yard box, far front corner`, goal + sign * 5.5, 24.84),
      mark(`${code}_six_front_near`, `${side} 6-yard box, near front corner`, goal + sign * 5.5, 43.16),
      mark(`${code}_spot`, `${side} penalty spot`, goal + sign * 11, 34),
      mark(`${code}_corner_far`, `${side} far corner of the pitch`, goal, 0),
      mark(`${code}_corner_near`, `${side} near corner of the pitch`, goal, 68),
    );
  }
  landmarks.push(mark("c_spot", "Centre spot", 52.5, 34));
  const lines = [
    line("far_touchline", "Far touchline", [0, 0], [105, 0]),
    line("near_touchline", "Near touchline", [0, 68], [105, 68]),
    line("halfway", "Halfway line", [52.5, 0], [52.5, 68]),
  ];
  for (const [code, side, goal, sign] of [["r", "Right", 105, -1], ["l", "Left", 0, 1]]) {
    const box = goal + sign * 16.5;
    lines.push(
      line(`${code}_goal`, `${side} goal line`, [goal, 0], [goal, 68]),
      line(`${code}_box_front`, `${side} penalty box, front line`, [box, 13.84], [box, 54.16]),
      line(`${code}_box_far`, `${side} penalty box, far side`, [goal, 13.84], [box, 13.84]),
      line(`${code}_box_near`, `${side} penalty box, near side`, [goal, 54.16], [box, 54.16]),
    );
  }
  const circle = [];
  for (let i = 0; i <= 64; i += 1) {
    const t = (i / 64) * Math.PI * 2;
    circle.push([52.5 + 9.15 * Math.cos(t), 34 + 9.15 * Math.sin(t)]);
  }
  return { length: 105, width: 68, landmarks, lines, arcs: [circle] };
}

const MOCK_PITCH = mockPitch();

/** The mock's solve: points only, four to fit, softly anchored when editing. */
function mockSolve(body) {
  const pins = (body && body.pins) || [];
  const hand = body && body.mode === "hand";
  const points = pins
    .filter((pin) => pin.kind === "point")
    .map((pin) => {
      const row = MOCK_PITCH.landmarks.find((item) => item.id === pin.ref);
      return row ? { x: row.x, y: row.y, u: pin.x, v: pin.y, w: 1 } : null;
    })
    .filter(Boolean);

  if (hand && points.length < 4) {
    const needed = 4 - points.length;
    return {
      mode: "hand",
      field_to_image: null,
      camera: null,
      fit: {
        tone: "warn",
        label: `PLACE ${needed} MORE`,
        message: "Choose points far apart. (Mock mode counts points only.)",
        can_continue: false,
        needed,
      },
    };
  }
  if (!hand && !pins.length) {
    return { mode: "edit", field_to_image: MOCK_AUTO_H, camera: null, fit: mockAutoFit() };
  }
  const anchors = hand
    ? []
    : MOCK_PITCH.landmarks.map((row) => {
        const w = MOCK_AUTO_H[2][0] * row.x + MOCK_AUTO_H[2][1] * row.y + MOCK_AUTO_H[2][2];
        return {
          x: row.x,
          y: row.y,
          u: (MOCK_AUTO_H[0][0] * row.x + MOCK_AUTO_H[0][1] * row.y + MOCK_AUTO_H[0][2]) / w,
          v: (MOCK_AUTO_H[1][0] * row.x + MOCK_AUTO_H[1][1] * row.y + MOCK_AUTO_H[1][2]) / w,
          w: 0.03,
        };
      });
  const H = mockHomography([...anchors, ...points]) || MOCK_AUTO_H;
  return {
    mode: hand ? "hand" : "edit",
    field_to_image: H,
    camera: null,
    fit: {
      tone: "ok",
      label: hand ? "GOOD FIT" : "ADJUSTED",
      message: `${pins.length} placed. Mock mode draws a flat fit only.`,
      can_continue: true,
      miss_px: 0,
      line_error_m: null,
      worst: null,
      needed: 0,
    },
  };
}

function mockAutoFit() {
  return {
    tone: "ok",
    label: "AUTO FIT",
    message: "Automatic result. Check that the yellow lines sit on the pitch markings.",
    can_continue: true,
    miss_px: null,
    line_error_m: null,
    worst: null,
    needed: 0,
  };
}

/* ==========================================================================
   Public mock API
   ========================================================================== */

/**
 * Build a mock API object with the same surface as the real one in app.js.
 * @returns {object} API implementation for mock mode.
 */
/* ==========================================================================
   Auto-review: scripted runs, one for each ending
   ========================================================================== */

const AR_RUN_GAP_MS = 320;
const AR_TOO_CLOSE = "Too close to call: the margin of +0.12 m is smaller than the ±0.30 m uncertainty.";
const AR_MISSED_BOX = [1022, 382, 1054, 494];
const AR_FALSE_BOX = [1206, 262, 1236, 352];

/** The offside line moved so it passes through an image point. */
function lineThrough(x, y) {
  const [a, b] = OFFSIDE_LINE;
  const along = a[0] + ((y - a[1]) / (b[1] - a[1])) * (b[0] - a[0]);
  return OFFSIDE_LINE.map(([px, py]) => [px + x - along, py]);
}

const AR_SAFE_LINE = lineThrough(AR_MISSED_BOX[2], AR_MISSED_BOX[3]);

/** Pitch markings as image polylines, for the camera correction. */
function arPitchLines(dx = 0, dy = 0) {
  const project = ([x, y]) => {
    const H = MOCK_AUTO_H;
    const w = H[2][0] * x + H[2][1] * y + H[2][2];
    return [
      (H[0][0] * x + H[0][1] * y + H[0][2]) / w + dx,
      (H[1][0] * x + H[1][1] * y + H[1][2]) / w + dy,
    ];
  };
  return [
    [[105, 13.84], [88.5, 13.84], [88.5, 54.16], [105, 54.16]],
    [[105, 24.84], [99.5, 24.84], [99.5, 43.16], [105, 43.16]],
    [[52.5, 0], [52.5, 68]],
  ].map((line) => line.map(project));
}

/** What the detector returns in these runs: a missed player and three mistakes to correct. */
function arStartPlayers() {
  const players = buildPlayers().filter((player) => player.id !== 14);
  for (const player of players) {
    if (player.id === 18) player.team = 1;
    if (player.id === 15) {
      player.team = 2;
      player.role = "player";
    }
  }
  players.push({ id: 30, bbox: AR_FALSE_BOX, team: 1, role: "player", confidence: 0.41 });
  return players;
}

/** The events for one run. A `{ wait: true }` step holds until the question is answered. */
function arScript(kind) {
  const check = (name) => ({ type: "check_started", check: name });
  const setKeeper = {
    type: "role_changed",
    player_id: 1,
    role: "goalkeeper",
    from_role: "player",
    reason: "Marked #1 as keeper: kit differs from both teams",
    flips_call: false,
    undo_id: "d-keeper",
  };
  const opening = [
    check("frame"),
    check("side"),
    {
      type: "side_set",
      side: "right",
      reason: "Set the attack to the right: goal in view and run to the right",
      flips_call: false,
      undo_id: "d-side",
    },
    check("team"),
    {
      type: "attacking_team_set",
      team: 2,
      reason: "Set Team 2 as attacking: has the ball, and Team 1 forms the back line",
      flips_call: false,
      undo_id: "d-team",
    },
    check("keeper"),
  ];
  const camera = [
    check("camera"),
    {
      type: "camera_refined",
      from_lines: arPitchLines(-20, 9),
      to_lines: arPitchLines(),
      reason: "Refined the camera: pitch lines now sit on the paint",
      flips_call: false,
      undo_id: "d-camera",
    },
  ];
  const corrections = [
    check("players"),
    {
      type: "player_added",
      player_id: 14,
      bbox: AR_MISSED_BOX,
      team: 1,
      reason: "Added missed defender #14",
      flips_call: true,
      undo_id: "d-missed",
    },
    {
      type: "box_removed",
      player_id: 30,
      bbox: AR_FALSE_BOX,
      team: 1,
      reason: "Removed #30: a ball boy behind the touchline",
      flips_call: false,
      undo_id: "d-false",
    },
    check("teams"),
    {
      type: "team_changed",
      player_id: 18,
      from_team: 1,
      to_team: 2,
      reason: "Moved #18 to Team 2: the kit matches Team 2",
      flips_call: false,
      undo_id: "d-team18",
    },
    check("officials"),
    {
      type: "role_changed",
      player_id: 15,
      role: "referee",
      from_role: "player",
      reason: "Marked #15 as referee: black kit, matches neither team",
      flips_call: false,
      undo_id: "d-referee",
    },
    check("stability"),
    check("call"),
    { type: "line_moved", from_points: null, to_points: clone(AR_SAFE_LINE), player_id: 14 },
    { type: "call_changed", from: "unknown", to: "onside", margin_m: 2.0 },
    { type: "run_ended", ending: "cleared", reason: null },
  ];

  if (kind === "not_usable") {
    return [check("frame"), { type: "run_ended", ending: "not_usable", reason: "Close-up: 5 people and no pitch lines." }];
  }
  if (kind === "too_close") {
    return [
      ...opening,
      setKeeper,
      ...camera,
      check("stability"),
      check("call"),
      { type: "line_moved", from_points: null, to_points: clone(OFFSIDE_LINE), player_id: 7 },
      { type: "run_ended", ending: "too_close", reason: AR_TOO_CLOSE },
    ];
  }
  if (kind === "question") {
    return [
      ...opening,
      {
        type: "question_asked",
        question_id: "q-keeper",
        text: "Is #1 the keeper?",
        player_ids: [1],
        choices: ["Yes", "Another player", "Not in frame"],
      },
      { wait: true },
      ...camera,
      ...corrections,
    ];
  }
  return [...opening, setKeeper, ...camera, ...corrections];
}

/** A verdict for the scripted scene. Its numbers come from the script, not from geometry. */
function arVerdict(run) {
  const side = run.side;
  if (!side) return unknownVerdict("right", "Choose the attack direction.");
  const problem = choiceError(activePlayers, run.team, run.keeper);
  if (problem) return unknownVerdict(side, problem);
  if (run.kind === "too_close") {
    return { ...unknownVerdict(side, AR_TOO_CLOSE), line_image_points: clone(OFFSIDE_LINE) };
  }
  const added = activePlayers.some((player) => player.id === 14);
  const verdict = added ? "onside" : "offside";
  const margin = added ? -2.0 : 1.0;
  const defender = added ? 14 : 7;
  const label = verdict === "onside" ? "Onside" : "Offside";
  return {
    verdict,
    estimate: false,
    message: null,
    margin_m: margin,
    attacking_side: side,
    last_defender: { id: defender, part: "right foot" },
    offside_player_ids: added ? [] : [2],
    line_image_points: clone(added ? AR_SAFE_LINE : OFFSIDE_LINE),
    line_field_x: added ? 86.2 : 80.1,
    calibration: { method: "pnl", full_3d: true, edited: false },
    checks: [
      { key: "players", label: "Players", status: "ok", detail: String(activePlayers.length) },
      { key: "teams", label: "Teams", status: "ok", detail: "2" },
      { key: "side", label: "Attack direction", status: "ok", detail: side === "left" ? "Left" : "Right" },
      { key: "calibration", label: "Calibration", status: "ok", detail: "PnL · 3D" },
      { key: "defender", label: "Last defender", status: "ok", detail: `#${defender} · right foot` },
      { key: "verdict", label: "Verdict", status: "ok", detail: `${label} · ${Math.abs(margin).toFixed(2)} m` },
    ],
    stages: ["original", "detection", "teams", "calibration", "offside", "birdseye", "result"],
    time_ms: 2210,
    view3d_available: false,
  };
}

/** Change the mock's own scene for an event, as the server would. */
function arApply(run, event) {
  switch (event.type) {
    case "player_added":
      activePlayers.push({ id: event.player_id, bbox: event.bbox, team: event.team, role: "player", confidence: 0.8 });
      break;
    case "box_removed":
      activePlayers = activePlayers.filter((player) => player.id !== event.player_id);
      break;
    case "team_changed":
      activePlayers.find((player) => player.id === event.player_id).team = event.to_team;
      break;
    case "role_changed":
      activePlayers.find((player) => player.id === event.player_id).role = event.role;
      if (event.role === "goalkeeper") run.keeper = "player";
      break;
    case "side_set":
      run.side = event.side;
      break;
    case "attacking_team_set":
      run.team = event.team;
      break;
    default:
      break;
  }
}

function arResult(run, ending, reason) {
  return {
    ending,
    reason: reason || null,
    players: clone(activePlayers),
    attacking_side: run.side,
    attacking_team: run.team,
    keeper: run.keeper,
    verdict: ending === "not_usable" ? null : arVerdict(run),
    stability: ending === "cleared" ? { alternatives_tested: 13, changed_the_call: 0 } : null,
  };
}

export function createMockApi() {
  let sessionCount = 0;
  let mockApplied = null;
  let arRun = null;

  return {
    mode: "mock",

    /** The health probe the app uses to decide whether a backend exists. */
    async health() {
      return { status: "ok", device: "cpu" };
    },

    /** Forget the mock's scene if this is the open session. */
    async deleteSession(sessionId) {
      await wait(80);
      if (sessionId === `mock-session-${sessionCount}`) {
        arRun = null;
        mockApplied = null;
        activePlayers = buildPlayers();
        stageCache.clear();
      }
      return { status: "deleted" };
    },

    /** Pretend to upload a file and return a session descriptor. */
    async upload(file) {
      await wait(280);
      sessionCount += 1;
      mockApplied = null;
      const name = (file && file.name) || "mock-pitch.jpg";
      const type = (file && file.type) || "";
      const isVideo =
        /^video\//.test(type) || /\.(mp4|mov|mkv|webm|avi)$/i.test(name);
      return {
        id: `mock-session-${sessionCount}`,
        kind: isVideo ? "video" : "image",
        width: IMAGE_W,
        height: IMAGE_H,
        frame_count: isVideo ? 240 : 1,
        fps: isVideo ? 25 : undefined,
        filename: name,
      };
    },

    /** Always the painted pitch. */
    frameUrl() {
      return getPitchDataUrl();
    },

    /** Keep the chosen pass frame (the mock image does not change). */
    async selectFrame(_sessionId, index) {
      await wait(40);
      return { index };
    },

    /** Phase 1: 19 people and two teams; the choices are left to the user. */
    async detect() {
      await wait(900);
      activePlayers = buildPlayers();
      arRun = null;
      stageCache.clear();
      return phase1(activePlayers);
    },

    /** Persist client edits into the fake session and redraw the stages. */
    async putPlayers(_sessionId, players) {
      await wait(260);
      stageCache.clear();
      if (arRun) arRun.edited = true;
      const next = clone(players).map((player, index) => {
        if (player.id === null || player.id === undefined) {
          return { ...player, id: 20 + index };
        }
        return player;
      });
      activePlayers = next;
      return phase1(next);
    },

    /** Calibration for the painted pitch, ready after a short wait. */
    async getCalibration() {
      await wait(180);
      return {
        status: "ready",
        mode: mockApplied ? mockApplied.mode : "edit",
        source: mockApplied ? (mockApplied.mode === "hand" ? "hand" : "edited") : "auto",
        method: mockApplied ? "manual" : "pnl",
        full_3d: true,
        auto_3d: true,
        field_to_image: mockApplied ? mockSolve(mockApplied).field_to_image : MOCK_AUTO_H,
        camera: null,
        fit: mockApplied ? mockSolve(mockApplied).fit : mockAutoFit(),
        pins: mockApplied ? clone(mockApplied.pins) : [],
        pitch: MOCK_PITCH,
      };
    },

    /** Browser-side trial solve. */
    async solveCalibration(_sessionId, body) {
      return mockSolve(body);
    },

    /** Remember the pins. */
    async applyCalibration(sessionId, body) {
      mockApplied = clone(body);
      return this.getCalibration(sessionId);
    },

    /** Back to the automatic result. */
    async resetCalibration(sessionId) {
      mockApplied = null;
      return this.getCalibration(sessionId);
    },

    /** Phase 2: a fixed offside verdict with a 0.84 m margin, once the choices are made. */
    async analyze(_sessionId, attackingSide, attackingTeam, keeper) {
      await wait(1150);
      if (arRun && !arRun.edited && arRun.ended) return arVerdict(arRun);
      const side = attackingSide || "right";
      const problem = choiceError(activePlayers, attackingTeam, keeper);
      if (problem) return unknownVerdict(side, problem);
      return {
        verdict: "offside",
        estimate: false,
        message: null,
        margin_m: 0.84,
        attacking_side: side,
        last_defender: { id: 7, part: "left heel" },
        offside_player_ids: [2],
        line_image_points: clone(OFFSIDE_LINE),
        line_field_x: mockApplied ? 80.42 : 80.1,
        calibration: {
          method: mockApplied ? "manual" : "pnl",
          full_3d: true,
          edited: Boolean(mockApplied),
        },
        checks: [
          { key: "players", label: "Players", status: "ok", detail: "19" },
          { key: "teams", label: "Teams", status: "ok", detail: "2" },
          {
            key: "side",
            label: "Attack direction",
            status: "ok",
            detail: side === "left" ? "Left" : "Right",
          },
          { key: "calibration", label: "Calibration", status: "ok", detail: "PnL · 3D" },
          { key: "defender", label: "Last defender", status: "ok", detail: "#7 · left heel" },
          { key: "verdict", label: "Verdict", status: "ok", detail: "Offside · 0.84 m" },
        ],
        stages: [
          "original",
          "detection",
          "teams",
          "calibration",
          "offside",
          "birdseye",
          "result",
        ],
        time_ms: 2210,
        view3d_available: true,
      };
    },

    /** Start a scripted Auto-review run. The real server will run the checks here. */
    async startAutoReview(_sessionId, kind = "cleared") {
      await wait(120);
      activePlayers = arStartPlayers();
      stageCache.clear();
      arRun = {
        kind,
        steps: arScript(kind),
        events: [],
        side: null,
        team: null,
        keeper: null,
        result: null,
        ended: false,
        edited: false,
        applied: new Map(),
        waiting: false,
        nextAt: performance.now() + AR_RUN_GAP_MS,
      };
      return { job_id: "mock-auto-review", phase1: phase1(activePlayers) };
    },

    /** New events since `since`, and the result once the run has ended. */
    async getAutoReview(_sessionId, since = 0) {
      const run = arRun;
      if (!run) throw new Error("No Auto-review run.");
      while (!run.waiting && !run.ended && performance.now() >= run.nextAt && run.steps.length) {
        const step = run.steps.shift();
        if (step.wait) {
          run.waiting = true;
          break;
        }
        const event = { ...clone(step), seq: run.events.length + 1 };
        arApply(run, event);
        if (event.undo_id) run.applied.set(event.undo_id, event);
        run.events.push(event);
        run.nextAt = performance.now() + AR_RUN_GAP_MS;
        if (event.type === "run_ended") {
          run.ended = true;
          run.result = arResult(run, event.ending, event.reason);
        }
      }
      const current = [...run.events].reverse().find((event) => event.type === "check_started");
      return {
        status: run.ended ? "done" : run.waiting ? "needs_answer" : "running",
        current_check: current ? current.check : null,
        events: clone(run.events.slice(since)),
        result: run.ended ? clone(run.result) : null,
      };
    },

    /** Apply the answer to the question the run is waiting on, then carry on. */
    async answerAutoReview(_sessionId, _questionId, choice) {
      await wait(120);
      const run = arRun;
      if (choice === "Review by hand") return { status: "stopped" };
      if (choice === "Another player") {
        run.steps = [{ type: "run_ended", ending: "needs_answer", reason: "Mark the keeper in Review." }];
      } else if (choice === "Yes") {
        run.steps.unshift({
          type: "role_changed",
          player_id: 1,
          role: "goalkeeper",
          from_role: "player",
          reason: "Marked #1 as keeper: you confirmed it",
          flips_call: false,
          undo_id: "d-keeper",
        });
      } else {
        run.keeper = "not_in_frame";
      }
      run.waiting = false;
      run.nextAt = performance.now();
      return { status: "running" };
    },

    /** Revert one correction and rerun the call. */
    async undoAutoReview(_sessionId, undoId) {
      await wait(160);
      const run = arRun;
      const event = run.applied.get(undoId);
      if (!event) throw new Error("That correction can't be undone.");
      run.applied.delete(undoId);
      const before = arVerdict(run);
      switch (event.type) {
        case "player_added":
          activePlayers = activePlayers.filter((player) => player.id !== event.player_id);
          break;
        case "box_removed":
          activePlayers.push({ id: event.player_id, bbox: event.bbox, team: event.team, role: "player", confidence: 0.41 });
          break;
        case "team_changed":
          activePlayers.find((player) => player.id === event.player_id).team = event.from_team;
          break;
        case "role_changed":
          activePlayers.find((player) => player.id === event.player_id).role = event.from_role;
          if (event.role === "goalkeeper") run.keeper = null;
          break;
        case "side_set":
          run.side = null;
          break;
        case "attacking_team_set":
          run.team = null;
          break;
        default:
          break;
      }
      const after = arVerdict(run);
      const events = [];
      const hasLines = before.line_image_points.length >= 2 && after.line_image_points.length >= 2;
      if (hasLines && before.line_image_points[0][0] !== after.line_image_points[0][0]) {
        events.push({
          type: "line_moved",
          from_points: before.line_image_points,
          to_points: after.line_image_points,
          player_id: after.last_defender.id,
        });
      }
      if (before.verdict !== after.verdict) {
        events.push({
          type: "call_changed",
          from: before.verdict,
          to: after.verdict,
          margin_m: after.margin_m,
        });
      }
      run.result = arResult(run, run.result.ending, run.result.reason);
      return { events, result: clone(run.result) };
    },

    /** Data URL for a debug stage. */
    stageUrl(_sessionId, key) {
      return stageDataUrl(key);
    },

    /** Start a fake export job. */
    async exportSession(_sessionId, kind) {
      await wait(120);
      return { job_id: startJob(kind) };
    },

    /** Poll a fake export job. */
    async getJob(jobId) {
      await wait(80);
      return readJob(jobId);
    },

    /** Cancel a fake export job. */
    async cancelJob(jobId) {
      await wait(80);
      const job = jobs.get(jobId);
      if (job) job.status = "cancelled";
      return { status: "cancelled" };
    },

    /** Mock file URLs are already absolute data URLs. */
    fileUrl(url) {
      return url;
    },

    /** Open the canvas-drawing fake 3D view. */
    openView(_sessionId, options) {
      return new MockView(options);
    },
  };
}
