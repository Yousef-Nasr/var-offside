/**
 * VAR Offside — browser frontend.
 *
 * One module, no build step. Talks to the FastAPI server through the single
 * `api` object below (real or mock); `mock.js` provides the replacement used when
 * `?mock=1` is present or `/api/health` is unreachable. All dynamic text is
 * inserted with textContent; the canvas overlay holds the only drawing code.
 */

import { createCalibrator } from "./calibrate.js";
import { createMockApi } from "./mock.js";
import { createAutoReview } from "./auto_review.js";
import {
  TEAM_COLORS,
  drawHoverBrackets,
  drawPlayerLabels,
  drawPlayerRing,
  markerColor,
} from "./players.js";

/* ==========================================================================
   Constants and state
   ========================================================================== */

const API = "/api";
const DEFAULT_CAMERA = {
  target: [0, 0, 0],
  yaw: 35,
  pitch: 22,
  distance: 18,
  fov: 40,
};
const PRESETS = ["broadcast", "birdseye", "line", "free"];
const PRESET_KEYS = { 1: "broadcast", 2: "birdseye", 3: "line", 4: "free" };
const STAGE_LABELS = {
  original: "Original",
  detection: "Detection",
  teams: "Teams",
  calibration: "Calibration",
  offside: "Offside line",
  birdseye: "Bird's eye",
  result: "Result",
};
const EXPORT_LABELS = {
  image: "Image with overlay",
  replay: "Replay video",
  birdseye: "Bird's-eye video",
  debug: "Debug video",
  report: "Report (zip)",
};
const EXPORT_EXT = {
  image: "png",
  replay: "mp4",
  birdseye: "mp4",
  debug: "mp4",
  report: "zip",
};
const BASE_CHECKS = [
  { key: "players", label: "Players", status: "pending", detail: "" },
  { key: "teams", label: "Teams", status: "pending", detail: "" },
  { key: "side", label: "Attack direction", status: "pending", detail: "" },
  { key: "calibration", label: "Calibration", status: "pending", detail: "" },
  { key: "defender", label: "Last defender", status: "pending", detail: "" },
  { key: "verdict", label: "Verdict", status: "pending", detail: "" },
];
const GLYPHS = { ok: "✓", check: "?", fail: "✕", pending: "" };

/** CSS custom property with a fallback, used for canvas colours. */
function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return value || fallback;
}

const COLORS = {
  red: cssVar("--red", "#ff5a4e"),
  green: cssVar("--green", "#4fc27a"),
  amber: cssVar("--amber", "#f0b44c"),
  muted: cssVar("--muted", "#8a9a90"),
  ink: cssVar("--ink", "#e3ece5"),
  accent: cssVar("--accent", "#4fc27a"),
  referee: "#8a9a90",
};

const state = {
  mock: false,
  phase: "empty",
  busy: false,
  uploading: false,
  session: null,
  frameIndex: 0,
  phase1: null,
  verdict: null,
  players: [],
  teamColors: { ...TEAM_COLORS },
  attackingSide: null,
  attackingTeam: null,
  keeperNotInFrame: false,
  showSteps: true,
  stepsOpen: true,
  linePlane: true,
  keyOnly: false,
  photo3d: false,
  showLabels: true,
  pulse: true,
  view: "photo",
  preset: "broadcast",
  camera: clone(DEFAULT_CAMERA),
  hoveredIndex: null,
  clickCandidate: null,
  dragBox: null,
  pendingPlayer: null,
  playerSaving: false,
  replaying: false,
  pendingReplay: false,
  has3d: false,
  stageVersion: 0,
  // Verdict shown before opening Calibrate from the verdict screen.
  calReturnVerdict: null,
  // Set after an edit, to show before and after on the new verdict.
  beforeEdit: null,
  autoReviewOn: false,
};

/** Runs Auto-review and plays its corrections. Created at boot. */
let autoReview = null;

/* ==========================================================================
   DOM
   ========================================================================== */

const byId = (id) => document.getElementById(id);

const dom = {
  app: byId("app"),
  mockChip: byId("mockChip"),
  fileName: byId("fileName"),
  fileRes: byId("fileRes"),
  openBtn: byId("openBtn"),
  clearBtn: byId("clearBtn"),
  primaryBtn: byId("primaryBtn"),
  arSwitchWrap: byId("arSwitchWrap"),
  arSwitch: byId("arSwitch"),
  arScenario: byId("arScenario"),
  primarySpinner: byId("primarySpinner"),
  primaryLabel: byId("primaryLabel"),
  exportBtn: byId("exportBtn"),
  exportMenu: byId("exportMenu"),
  settingsBtn: byId("settingsBtn"),
  settingsPopover: byId("settingsPopover"),
  stepsSwitch: byId("stepsSwitch"),
  helpBtn: byId("helpBtn"),
  helpPopover: byId("helpPopover"),
  progressLine: byId("progressLine"),
  viewport: byId("viewport"),
  dropzone: byId("dropzone"),
  mockSample: byId("mockSample"),
  photoLayer: byId("photoLayer"),
  photoImg: byId("photoImg"),
  overlayCanvas: byId("overlayCanvas"),
  view3dLayer: byId("view3dLayer"),
  view3dCanvas: byId("view3dCanvas"),
  viewHint: byId("viewHint"),
  gizmoX: byId("gizmoX"),
  gizmoZ: byId("gizmoZ"),
  gizmoY: byId("gizmoY"),
  gizmoXLabel: byId("gizmoXLabel"),
  gizmoZLabel: byId("gizmoZLabel"),
  gizmoYLabel: byId("gizmoYLabel"),
  reconnectChip: byId("reconnectChip"),
  viewToggle: byId("viewToggle"),
  viewPhotoBtn: byId("viewPhotoBtn"),
  view3dBtn: byId("view3dBtn"),
  statusChip: byId("statusChip"),
  reviewCard: byId("reviewCard"),
  reviewHint: byId("reviewHint"),
  keeperStatus: byId("keeperStatus"),
  notInFrameBtn: byId("notInFrameBtn"),
  stageLightbox: byId("stageLightbox"),
  stageLarge: byId("stageLarge"),
  stageCaption: byId("stageCaption"),
  stageClose: byId("stageClose"),
  playerTip: byId("playerTip"),
  ctxMenu: byId("ctxMenu"),
  bottomBar: byId("bottomBar"),
  timeline: byId("timeline"),
  prevFrame: byId("prevFrame"),
  scrubber: byId("scrubber"),
  nextFrame: byId("nextFrame"),
  frameCounter: byId("frameCounter"),
  cameraControls: byId("cameraControls"),
  presetGroup: byId("presetGroup"),
  toggles: byId("toggles"),
  replayBtn: byId("replayBtn"),
  verdictBlock: byId("verdictBlock"),
  verdictEyebrow: byId("verdictEyebrow"),
  verdictMain: byId("verdictMain"),
  verdictSub: byId("verdictSub"),
  verdictMeta: byId("verdictMeta"),
  verdictFigure: byId("verdictFigure"),
  verdictNum: byId("verdictNum"),
  verdictBand: byId("verdictBand"),
  vbCall: byId("vbCall"),
  vbCaption: byId("vbCaption"),
  vbMargin: byId("vbMargin"),
  vbNum: byId("vbNum"),
  phaseSteps: byId("phaseSteps"),
  checksList: byId("checksList"),
  actionList: byId("actionList"),
  actionHint: byId("actionHint"),
  stepsDrawer: byId("stepsDrawer"),
  stepsToggle: byId("stepsToggle"),
  stepsCaret: byId("stepsCaret"),
  stepsGrid: byId("stepsGrid"),
  stepsEmpty: byId("stepsEmpty"),
  toasts: byId("toasts"),
  fileInput: byId("fileInput"),
  calibLayer: byId("calibLayer"),
  calibCanvas: byId("calibCanvas"),
  calibPanel: byId("calibPanel"),
  calibBack: byId("calibBack"),
  calibWithout3d: byId("calibWithout3d"),
};

/* ==========================================================================
   Small utilities
   ========================================================================== */

/** Structured clone for plain JSON-ish data. */
function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

/** Clamp `value` into [min, max]. */
function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/** Pad an integer for the frame counter. */
function pad(value, width = 4) {
  return String(Math.max(0, Math.trunc(value))).padStart(width, "0");
}

/** Degrees to radians. */
function rad(degrees) {
  return (degrees * Math.PI) / 180;
}

/** Wrap an angle in degrees into [-180, 180). */
function wrapDegrees(degrees) {
  return ((((degrees + 180) % 360) + 360) % 360) - 180;
}

/** Finite number or fallback. */
function num(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/** Set textContent safely. */
function setText(element, text) {
  if (element) element.textContent = text == null ? "" : String(text);
}

/** Toggle the hidden attribute. */
function show(element, visible) {
  if (element) element.hidden = !visible;
}

/** Create an element with class and text. */
function make(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text != null) element.textContent = String(text);
  return element;
}

/** Two-digit confidence label. */
function confidenceLabel(player) {
  if (typeof player.confidence !== "number") return null;
  return `${Math.round(clamp(player.confidence, 0, 1) * 100)}%`;
}

/** Human label for an attack side. */
function sideLabel(side) {
  if (side === "left") return "Left";
  if (side === "right") return "Right";
  return "—";
}

/** Convert #rgb/#rrggbb to rgba(); falls back to the colour string. */
function hexToRgba(hex, alpha) {
  const value = String(hex || "").trim();
  const short = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(value);
  const long = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(value);
  if (!short && !long) return `rgba(138, 154, 144, ${alpha})`;
  const parts = short
    ? [short[1] + short[1], short[2] + short[2], short[3] + short[3]]
    : [long[1], long[2], long[3]];
  const [r, g, b] = parts.map((part) => parseInt(part, 16));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* ==========================================================================
   HTTP API (real)
   ========================================================================== */

/** Fetch JSON, surfacing `{detail}` from error responses. */
async function request(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!response.ok) {
    const detail =
      data && typeof data.detail === "string"
        ? data.detail
        : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return data;
}

/** JSON request options. */
function jsonRequest(method, body) {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body == null ? {} : body),
  };
}

/**
 * Paint a JPEG blob into the 3D canvas. Uses createImageBitmap when available
 * and an object URL (always revoked) as a fallback.
 */
async function paintBlob(canvas, blob, isClosed) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let bitmap = null;
  try {
    if (typeof createImageBitmap === "function") {
      bitmap = await createImageBitmap(blob);
    }
  } catch {
    bitmap = null;
  }

  if (isClosed()) {
    if (bitmap) bitmap.close();
    return;
  }

  if (bitmap) {
    if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
    if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    return;
  }

  const url = URL.createObjectURL(blob);
  await new Promise((resolve) => {
    const image = new Image();
    image.onload = () => {
      if (!isClosed()) {
        if (canvas.width !== image.naturalWidth) canvas.width = image.naturalWidth;
        if (canvas.height !== image.naturalHeight) canvas.height = image.naturalHeight;
        ctx.drawImage(image, 0, 0);
      }
      URL.revokeObjectURL(url);
      resolve();
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      resolve();
    };
    image.src = url;
  });
}

/** The real HTTP + WebSocket implementation. */
const realApi = {
  mode: "real",

  /** Readiness probe (also used by callers that want the device string). */
  async health() {
    return request(`${API}/health`);
  },

  /** Upload an image or video, returning the session descriptor. */
  async upload(file) {
    const form = new FormData();
    form.append("file", file);
    return request(`${API}/sessions`, { method: "POST", body: form });
  },

  /** Delete a session and everything made from it. */
  async deleteSession(sessionId) {
    return request(`${API}/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  },

  /** URL of one source frame. */
  frameUrl(sessionId, index) {
    return `${API}/sessions/${encodeURIComponent(sessionId)}/frame.jpg?index=${Number(index) || 0}`;
  },

  /** Mark which frame later phases analyse. */
  async selectFrame(sessionId, index) {
    return request(
      `${API}/sessions/${encodeURIComponent(sessionId)}/select-frame`,
      jsonRequest("POST", { index: Number(index) || 0 }),
    );
  },

  /** Phase 1: detection, teams and roles. */
  async detect(sessionId) {
    return request(
      `${API}/sessions/${encodeURIComponent(sessionId)}/detect`,
      jsonRequest("POST", {}),
    );
  },

  /** Apply the user's review edits and get redrawn stages back. */
  async putPlayers(sessionId, players) {
    return request(
      `${API}/sessions/${encodeURIComponent(sessionId)}/players`,
      { ...jsonRequest("PUT", { players }), method: "PUT" },
    );
  },

  /** Phase 2: calibration, offside line and verdict. */
  /** Start Auto-review on the detected frame. */
  async startAutoReview(sessionId) {
    return request(`${API}/sessions/${encodeURIComponent(sessionId)}/auto-review`, { method: "POST" });
  },

  /** New events since `since`, and the result once the run has ended. */
  async getAutoReview(sessionId, since = 0) {
    return request(`${API}/sessions/${encodeURIComponent(sessionId)}/auto-review?since=${since}`);
  },

  /** Answer the question the run is waiting on. */
  async answerAutoReview(sessionId, questionId, choice) {
    return request(
      `${API}/sessions/${encodeURIComponent(sessionId)}/auto-review/answers/${encodeURIComponent(questionId)}`,
      jsonRequest("POST", { choice }),
    );
  },

  /** Revert one correction and rerun the call. */
  async undoAutoReview(sessionId, undoId) {
    return request(
      `${API}/sessions/${encodeURIComponent(sessionId)}/auto-review/decisions/${encodeURIComponent(undoId)}/undo`,
      { method: "POST" },
    );
  },

  async analyze(sessionId, attackingSide, attackingTeam, keeper) {
    return request(
      `${API}/sessions/${encodeURIComponent(sessionId)}/analyze`,
      jsonRequest("POST", {
        attacking_side: attackingSide,
        attacking_team: attackingTeam,
        keeper,
      }),
    );
  },

  /** The frame's calibration; starts automatic calibration if needed. */
  async getCalibration(sessionId) {
    return request(`${API}/sessions/${encodeURIComponent(sessionId)}/calibration`);
  },

  /** Trial solve from pins; changes nothing on the server. */
  async solveCalibration(sessionId, body) {
    return request(
      `${API}/sessions/${encodeURIComponent(sessionId)}/calibration/solve`,
      jsonRequest("POST", body),
    );
  },

  /** Apply pins as the frame's calibration. */
  async applyCalibration(sessionId, body) {
    return request(
      `${API}/sessions/${encodeURIComponent(sessionId)}/calibration`,
      jsonRequest("PUT", body),
    );
  },

  /** Go back to the automatic result. */
  async resetCalibration(sessionId) {
    return request(`${API}/sessions/${encodeURIComponent(sessionId)}/calibration`, {
      method: "DELETE",
    });
  },

  /** URL of one debug stage; `version` keeps edits from being cache-stuck. */
  stageUrl(sessionId, key, version = 0) {
    return `${API}/sessions/${encodeURIComponent(sessionId)}/stages/${encodeURIComponent(key)}.png?v=${Number(version) || 0}`;
  },

  /** Start an export job. */
  async exportSession(sessionId, kind, options) {
    return request(
      `${API}/sessions/${encodeURIComponent(sessionId)}/export`,
      jsonRequest("POST", { kind, options }),
    );
  },

  /** Poll an export job. */
  async getJob(jobId) {
    return request(`${API}/jobs/${encodeURIComponent(jobId)}`);
  },

  /** Cancel an export job. */
  async cancelJob(jobId) {
    return request(`${API}/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
  },

  /** The API returns absolute `/api/files/...` paths already. */
  fileUrl(url) {
    return url;
  },

  /** Open the live 3D WebSocket and return a small send/close handle. */
  openView(sessionId, options) {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${location.host}${API}/sessions/${encodeURIComponent(sessionId)}/view`;
    const socket = new WebSocket(url);
    socket.binaryType = "blob";

    let closedByUs = false;
    let frameBusy = false;
    let pendingBlob = null;

    /** Draw one binary frame, keeping only the freshest pending frame. */
    const paint = (blob) => {
      if (frameBusy) {
        pendingBlob = blob;
        return;
      }
      frameBusy = true;
      paintBlob(options.canvas, blob, () => closedByUs).finally(() => {
        frameBusy = false;
        if (pendingBlob && !closedByUs) {
          const next = pendingBlob;
          pendingBlob = null;
          paint(next);
        }
      });
    };

    /** Unexpected close: detach socket listeners, then report disconnection. */
    const handleClose = () => {
      socket.removeEventListener("open", handleOpen);
      socket.removeEventListener("message", handleMessage);
      socket.removeEventListener("close", handleClose);
      socket.removeEventListener("error", handleError);
      if (!closedByUs && options.onClose) options.onClose();
    };

    /** Socket opened: the app sends its first preset. */
    const handleOpen = () => {
      if (!closedByUs && options.onOpen) options.onOpen();
    };

    /** JSON control messages and binary JPEG frames. */
    const handleMessage = (event) => {
      if (typeof event.data === "string") {
        let message = null;
        try {
          message = JSON.parse(event.data);
        } catch {
          message = null;
        }
        if (!message || typeof message.type !== "string") return;
        if (message.type === "status" && options.onStatus) {
          options.onStatus(message.stage, message.message);
        } else if (message.type === "camera" && options.onCamera) {
          options.onCamera(message.camera);
        } else if (message.type === "replay_end" && options.onReplayEnd) {
          options.onReplayEnd();
        } else if (message.type === "defender_part" && options.onDefenderPart) {
          options.onDefenderPart(message.name);
        }
        return;
      }
      if (event.data instanceof Blob) paint(event.data);
    };

    // Errors are followed by close; the close handler owns the disconnected UI.
    const handleError = () => {};

    socket.addEventListener("open", handleOpen);
    socket.addEventListener("message", handleMessage);
    socket.addEventListener("close", handleClose);
    socket.addEventListener("error", handleError);

    const send = (payload) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(payload));
      }
    };

    return {
      sendCamera(camera, width, height, quality, viewOptions) {
        send({
          type: "camera",
          camera,
          width,
          height,
          quality,
          options: viewOptions,
        });
      },
      sendPreset(name, width, height, viewOptions) {
        send({ type: "preset", name, width, height, options: viewOptions });
      },
      sendReplay(width, height, viewOptions) {
        send({ type: "replay", width, height, options: viewOptions });
      },
      sendStop() {
        send({ type: "stop" });
      },
      sendVisible(visible) {
        send({ type: "visible", visible });
      },
      close() {
        closedByUs = true;
        if (pendingBlob) pendingBlob = null;
        try {
          socket.close();
        } catch {
          // Already closed.
        }
      },
    };
  },
};

/** The single API object used by the UI; swapped for the mock at boot. */
let api = realApi;

/* ==========================================================================
   Phase helpers / rendering
   ========================================================================== */

/** Copy the current batch of server checks into the six stable rows. */
function displayChecks() {
  if (state.phase === "detecting" || state.phase === "analyzing" || state.phase === "autoreview") {
    return BASE_CHECKS.map((check) => ({
      ...check,
      status: "pending",
      detail: "",
    }));
  }

  let source = BASE_CHECKS;
  if (state.phase === "verdict" && state.verdict && Array.isArray(state.verdict.checks)) {
    source = state.verdict.checks;
  } else if (state.phase1 && Array.isArray(state.phase1.checks)) {
    source = state.phase1.checks;
  }

  const byKey = new Map();
  for (const check of source) {
    if (check && typeof check.key === "string") byKey.set(check.key, check);
  }

  return BASE_CHECKS.map((base) => {
    const found = byKey.get(base.key);
    const merged = found ? { ...base, ...found } : { ...base };
    if (state.phase === "calibrate") {
      if (base.key === "side" && state.attackingSide) {
        merged.status = "ok";
        merged.detail = sideLabel(state.attackingSide);
      } else if (base.key === "calibration") {
        const fit = calibrator.info().fit;
        const tone = fit ? fit.tone : "idle";
        merged.status = tone === "ok" ? "ok" : tone === "bad" ? "fail" : tone === "warn" ? "check" : "pending";
        merged.detail = fit ? fit.label.toLowerCase() : "";
      }
    }
    if (base.key === "side" && state.phase === "review") {
      if (state.attackingSide) {
        merged.status = "ok";
        merged.detail = sideLabel(state.attackingSide);
      }
    }
    return merged;
  });
}

/** Render the Checks list. */
function renderChecks() {
  const checks = displayChecks();
  const animating = state.phase === "detecting" || state.phase === "analyzing";
  dom.checksList.classList.toggle("is-animating", animating);

  const rows = checks.map((check) => {
    const row = make("li", "check");
    row.dataset.status = check.status || "pending";
    const glyph = make("span", "glyph", GLYPHS[check.status] || GLYPHS.pending);
    glyph.setAttribute("aria-hidden", "true");
    const label = make("span", "label", check.label || check.key);
    const detail = make("span", "detail", check.detail || "");
    row.append(glyph, label, detail);
    row.setAttribute("aria-label", `${label.textContent}: ${check.status || "pending"}`);
    return row;
  });

  dom.checksList.replaceChildren(...rows);
}

/** Primary button label, spinner and disabled state for the current phase. */
function renderPrimary() {
  let label = "Detect";
  let disabled = true;
  let spinner = false;

  if (state.uploading) {
    label = "Loading…";
    disabled = true;
    spinner = true;
  } else {
    switch (state.phase) {
      case "empty":
        label = "Detect";
        disabled = true;
        break;
      case "loaded":
        label = autoReviewEnabled() ? "Auto-review" : "Detect";
        disabled = false;
        break;
      case "autoreview":
        label = "Auto-review…";
        disabled = true;
        spinner = true;
        break;
      case "detecting":
        label = "Detecting…";
        disabled = true;
        spinner = true;
        break;
      case "review":
        label = "Calibrate";
        disabled = reviewMissing().length > 0 || state.playerSaving;
        break;
      case "calibrate": {
        const info = calibrator.info();
        label = info.dirty ? "Apply & continue" : "Continue";
        disabled = !info.canContinue || state.busy;
        break;
      }
      case "analyzing":
        label = "Analyzing…";
        disabled = true;
        spinner = true;
        break;
      case "verdict":
        label = "Re-analyze";
        disabled = false;
        break;
      default:
        break;
    }
  }

  setText(dom.primaryLabel, label);
  dom.primarySpinner.hidden = !spinner;
  dom.primaryBtn.disabled = disabled;
}

/** Top-bar file name and resolution. */
function renderFileMeta() {
  if (!state.session) {
    setText(dom.fileName, "No frame");
    setText(dom.fileRes, "—");
    return;
  }
  setText(dom.fileName, state.session.filename || "Untitled");
  const resolution = `${state.session.width || 0} × ${state.session.height || 0}`;
  const fps = state.session.kind === "video" ? ` · ${num(state.session.fps, 0)} fps` : "";
  setText(dom.fileRes, `${resolution}${fps}`);
}

/** The verdict block on top of the right panel. */
function renderVerdict() {
  let tone = "muted";
  let eyebrow = "Verdict";
  let main = "NO FRAME YET";
  let sub = "Drop a photo or clip to begin";
  let meta = [];
  let figure = null;

  if (state.phase === "empty") {
    tone = "muted";
    main = "NO FRAME YET";
    sub = "Drop a photo or clip to begin";
  } else if (state.phase === "loaded") {
    tone = "amber";
    main = "NO CALL";
    sub = autoReviewEnabled()
      ? "Press Auto-review to check and correct the scene"
      : "Press Detect to find the players and teams";
    meta = [];
  } else if (state.phase === "detecting") {
    tone = "muted";
    main = "DETECTING";
    sub = "Finding players, teams and roles…";
  } else if (state.phase === "autoreview") {
    tone = "muted";
    eyebrow = "Auto-review";
    main = "CHECKING";
    sub = "Checking the scene and correcting what it can prove…";
    if (state.attackingSide) meta.push(["Attack", sideLabel(state.attackingSide)]);
    if (state.attackingTeam) meta.push(["Attacking team", `Team ${state.attackingTeam}`]);
  } else if (state.phase === "review") {
    tone = "review";
    eyebrow = "Verdict · Review";
    main = "REVIEW";
    sub = "Check the players, then choose the direction, team and keeper";
    if (state.attackingSide) meta.push(["Attack", sideLabel(state.attackingSide)]);
    if (state.attackingTeam) meta.push(["Attacking team", `Team ${state.attackingTeam}`]);
    if (keeperChoice()) meta.push(["Keeper", keeperText()]);
  } else if (state.phase === "calibrate") {
    const fit = calibrator.info().fit;
    const toneOf = { ok: "green", warn: "amber", bad: "red", idle: "muted" };
    tone = fit ? toneOf[fit.tone] || "muted" : "muted";
    eyebrow = "Calibration";
    main = fit ? fit.label : "CALIBRATE";
    sub = fit ? fit.message : "";
    if (fit && fit.line_error_m != null) meta.push(["Line within", `about ${fit.line_error_m.toFixed(2)} m`]);
    if (fit && fit.miss_px != null) meta.push(["Average miss", `${fit.miss_px.toFixed(1)} px`]);
  } else if (state.phase === "analyzing") {
    tone = "muted";
    main = "ANALYZING";
    sub = "Checking the offside line…";
  } else if (state.phase === "verdict" && state.verdict) {
    const verdict = state.verdict;
    const value = verdict.verdict;
    tone = value === "offside" ? "red" : value === "onside" ? "green" : "amber";
    main = value === "offside" ? "OFFSIDE" : value === "onside" ? "ONSIDE" : "NO CALL";
    eyebrow = verdict.estimate ? "Verdict · Estimate" : "Verdict";

    if (value === "unknown") {
      sub = verdict.message || "Verdict could not be determined";
    } else if (verdict.estimate) {
      sub = verdict.message || "Estimate from the drawn line, with no measured margin";
    } else if (verdict.margin_m == null) {
      sub =
        value === "offside"
          ? "Attacker ahead · margin unknown"
          : "Attacker level or behind · margin unknown";
    } else if (value === "offside") {
      sub = "Attacker ahead of the last defender";
      figure = Math.abs(Number(verdict.margin_m));
    } else {
      sub = "Attacker level or behind";
      if (value === "onside") figure = Math.abs(Number(verdict.margin_m));
    }

    const parts = [["Attack", sideLabel(verdict.attacking_side || state.attackingSide)]];
    if (verdict.estimate) parts.push(["Margin", "Estimate, not measured"]);
    if (verdict.last_defender && verdict.last_defender.id != null) {
      const part = verdict.last_defender.part ? ` · ${verdict.last_defender.part}` : "";
      parts.push(["Last defender", `#${verdict.last_defender.id}${part}`]);
    }
    if (Array.isArray(verdict.offside_player_ids) && verdict.offside_player_ids.length) {
      parts.push(["Offside", verdict.offside_player_ids.map((id) => `#${id}`).join(", ")]);
    }
    if (
      verdict.calibration &&
      verdict.calibration.method &&
      verdict.calibration.method !== "failed"
    ) {
      let label = verdict.calibration.full_3d ? "Full 3D" : "2D line";
      if (verdict.calibration.edited) label += ", edited by hand";
      parts.push(["Calibration", label]);
    }
    if (state.beforeEdit) parts.push(...beforeEditRows(state.beforeEdit, verdict));
    meta = parts;
  }

  dom.verdictBlock.dataset.tone = tone;
  setText(dom.verdictEyebrow, eyebrow);
  setText(dom.verdictMain, main);
  setText(dom.verdictSub, sub);

  const hasFigure = figure != null && Number.isFinite(figure);
  dom.verdictFigure.hidden = !hasFigure;
  if (hasFigure && !marginAnimating) setText(dom.verdictNum, figure.toFixed(2));

  const rows = [];
  for (const [term, value] of meta) {
    rows.push(make("dt", null, term), make("dd", null, value));
  }
  dom.verdictMeta.replaceChildren(...rows);
  dom.verdictMeta.hidden = rows.length === 0;
  renderVerdictBand();
}

/** "Before edit" rows comparing the last verdict with the new one. */
function beforeEditRows(before, verdict) {
  const name = { offside: "Offside", onside: "Onside", unknown: "No call" }[before.verdict] || "No call";
  const margin =
    before.margin_m == null
      ? ""
      : ` · ${before.margin_m >= 0 ? "+" : ""}${Number(before.margin_m).toFixed(2)} m`;
  const rows = [["Before edit", `${name}${margin}`]];
  if (before.line_field_x != null && verdict.line_field_x != null) {
    const moved = Math.abs(Number(verdict.line_field_x) - Number(before.line_field_x));
    rows.push(["Line moved", `${moved.toFixed(2)} m`]);
  }
  return rows;
}

/** Short call caption for the broadcast band. */
function verdictCaption(verdict) {
  if (verdict.verdict === "unknown") {
    return verdict.message || "Verdict could not be determined";
  }
  const lead = verdict.estimate ? "Estimate · " : "";
  const defender =
    verdict.last_defender && verdict.last_defender.id != null
      ? `last defender #${verdict.last_defender.id}`
      : "the last defender";
  const ids = Array.isArray(verdict.offside_player_ids) ? verdict.offside_player_ids : [];
  if (verdict.verdict === "offside") {
    const who = ids.length ? `Attacker #${ids[0]}` : "Attacker";
    return `${lead}${who} ahead of ${defender}`;
  }
  return `${lead}Attacker level with or behind ${defender}`;
}

/** Keep the broadcast verdict graphic in sync with the phase. */
function renderVerdictBand() {
  const band = dom.verdictBand;
  if (!band) return;
  const verdict = state.verdict;
  const visible = state.phase === "verdict" && Boolean(verdict);
  band.hidden = !visible;
  if (!visible) return;
  const value = verdict.verdict;
  band.dataset.tone = value === "offside" ? "red" : value === "onside" ? "green" : "amber";
  setText(dom.vbCall, value === "offside" ? "OFFSIDE" : value === "onside" ? "ONSIDE" : "NO CALL");
  setText(dom.vbCaption, verdictCaption(verdict));
  const margin = Math.abs(Number(verdict.margin_m));
  const hasMargin = verdict.margin_m != null && Number.isFinite(margin) && value !== "unknown";
  dom.vbMargin.hidden = !hasMargin;
  if (hasMargin && !marginAnimating) setText(dom.vbNum, margin.toFixed(2));
}

/** Progress stepper in the top bar. */
function renderPhaseSteps() {
  if (!dom.phaseSteps) return;
  const order = ["load", "detect", "review", "calibrate", "verdict"];
  const current =
    {
      empty: "load",
      loaded: "detect",
      detecting: "detect",
      review: "review",
      calibrate: "calibrate",
      analyzing: "verdict",
      verdict: "verdict",
      autoreview: "review",
    }[state.phase] || "load";
  const working = state.phase === "detecting" || state.phase === "analyzing" || state.phase === "autoreview";
  const index = order.indexOf(current);
  for (const item of dom.phaseSteps.querySelectorAll("[data-step]")) {
    const at = order.indexOf(item.dataset.step);
    let status = at < index ? "done" : at === index ? "current" : "todo";
    if (at === index && state.phase === "verdict") status = "current final";
    if (at === index && working) status = "current working";
    item.dataset.state = status;
    if (at === index) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  }
}

/** Panel actions for the current phase. */
function renderActions() {
  const nodes = [];

  if (state.phase === "verdict") {
    const replay = make(
      "button",
      `btn ghost accent has-icon ${state.replaying ? "is-stop" : "is-play"}`,
      state.replaying ? "Stop replay" : "Play replay",
    );
    replay.type = "button";
    replay.disabled = !state.has3d;
    if (!state.has3d) replay.title = "3D view is not available for this result";
    replay.addEventListener("click", toggleReplay);
    nodes.push(replay);

    const edit = make("button", "btn ghost has-icon is-edit", "Edit players");
    edit.type = "button";
    edit.addEventListener("click", editPlayers);
    nodes.push(edit);

    const calibration = make("button", "btn ghost has-icon is-edit", "Edit calibration");
    calibration.type = "button";
    calibration.addEventListener("click", editCalibration);
    nodes.push(calibration);
  }

  dom.actionList.replaceChildren(...nodes);

  const hints = {
    empty: "Drop a photo or clip to begin.",
    loaded: "Detect finds people, teams and roles. You always review before the call.",
    detecting: "Working…",
    review: "Choose the direction, attacking team and keeper in the card, then Calibrate.",
    calibrate: "",
    analyzing: "Working…",
    verdict: "Edit the players or calibration if a call looks wrong.",
  };
  setText(dom.actionHint, hints[state.phase] || "");
}

/** Collapsible debug-stage drawer. */
function renderSteps() {
  const visible = Boolean(state.session) && state.showSteps;
  show(dom.stepsDrawer, visible);
  dom.stepsToggle.setAttribute("aria-expanded", String(state.stepsOpen));
  if (!visible) return;

  const stages =
    state.phase === "verdict" && state.verdict && Array.isArray(state.verdict.stages)
      ? state.verdict.stages
      : state.phase1 && Array.isArray(state.phase1.stages)
        ? state.phase1.stages
        : [];

  dom.stepsGrid.replaceChildren();
  if (!stages.length) {
    show(dom.stepsEmpty, true);
    return;
  }
  show(dom.stepsEmpty, false);

  const seen = new Set();
  for (const key of stages) {
    if (!key || seen.has(key)) continue;
    seen.add(key);
    const label = STAGE_LABELS[key] || key;
    const button = make("button", "step-thumb");
    button.type = "button";
    button.title = `Open ${label} full size`;
    const image = document.createElement("img");
    image.alt = `${label} stage`;
    image.loading = "lazy";
    image.src = api.stageUrl(state.session.id, key, state.stageVersion);
    button.append(image, make("span", null, label));
    button.addEventListener("click", () => openStage(key, label));
    dom.stepsGrid.append(button);
  }
}

/** The keeper choice the API expects, or null while it is still open. */
function keeperChoice() {
  const marked = state.players.filter((player) => player.role === "goalkeeper").length;
  if (marked === 1) return "player";
  if (marked > 1) return null;
  return state.keeperNotInFrame ? "not_in_frame" : null;
}

/** The keeper choice in words. */
function keeperText() {
  const choice = keeperChoice();
  if (choice === "not_in_frame") return "Not in frame";
  if (choice === "player") {
    const keeper = state.players.find((player) => player.role === "goalkeeper");
    return keeper.id == null ? "Marked player" : `#${keeper.id}`;
  }
  return "";
}

/** What the user still has to choose before the frame can be analyzed. */
function reviewMissing() {
  const missing = [];
  if (!state.attackingSide) missing.push("attack direction");
  if (!state.attackingTeam) missing.push("attacking team");
  if (!keeperChoice()) missing.push("keeper");
  return missing;
}

/** Update the review card's selections. */
function renderReviewCard() {
  dom.reviewCard.hidden = state.phase !== "review";
  if (state.phase !== "review") return;

  for (const button of dom.reviewCard.querySelectorAll(".side-btn")) {
    const side = button.dataset.side;
    const selected = state.attackingSide === side;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  }

  for (const button of dom.reviewCard.querySelectorAll(".team-btn")) {
    const team = Number(button.dataset.team);
    const selected = state.attackingTeam === team;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.querySelector(".team-dot").style.background = state.teamColors[team];
  }

  const choice = keeperChoice();
  const notInFrame = choice === "not_in_frame";
  dom.notInFrameBtn.classList.toggle("is-selected", notInFrame);
  dom.notInFrameBtn.setAttribute("aria-pressed", String(notInFrame));
  dom.keeperStatus.classList.toggle("is-chosen", choice === "player");
  setText(
    dom.keeperStatus,
    choice === "player" ? `Keeper: ${keeperText()}` : "Click a player, then Mark goalkeeper",
  );

  const missing = reviewMissing();
  setText(
    dom.reviewHint,
    missing.length ? `Still needed: ${missing.join(", ")}.` : "All set. Press Calibrate to continue.",
  );
}

/** Frame counter and range bounds. */
function updateTimeline() {
  if (!state.session || state.session.kind !== "video") {
    dom.timeline.hidden = true;
    return;
  }
  const total = Math.max(1, Number(state.session.frame_count) || 1);
  const max = total - 1;
  state.frameIndex = clamp(state.frameIndex, 0, max);
  dom.scrubber.max = String(max);
  dom.scrubber.value = String(state.frameIndex);
  const width = Math.max(4, String(max).length);
  setText(dom.frameCounter, `${pad(state.frameIndex, width)} / ${pad(max, width)}`);
  dom.scrubber.disabled = state.busy;
  dom.prevFrame.disabled = state.busy || state.frameIndex <= 0;
  dom.nextFrame.disabled = state.busy || state.frameIndex >= max;
}

/** Selected preset styling. */
function updatePresetButtons() {
  for (const button of dom.presetGroup.querySelectorAll("[data-preset]")) {
    const active = button.dataset.preset === state.preset;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }
}

/** View-option toggle styling. */
function renderOptionButtons() {
  const values = {
    line_plane: state.linePlane,
    key_only: state.keyOnly,
    photo: state.photo3d,
    labels: state.showLabels,
    pulse: state.pulse,
  };
  for (const button of dom.toggles.querySelectorAll("[data-opt]")) {
    const on = Boolean(values[button.dataset.opt]);
    button.classList.toggle("is-on", on);
    button.setAttribute("aria-pressed", String(on));
  }
}

/** Photo | 3D segmented state. */
function updateViewToggle() {
  show(dom.viewToggle, state.phase === "verdict" && state.has3d);
  const photo = state.view === "photo";
  dom.viewPhotoBtn.classList.toggle("is-active", photo);
  dom.view3dBtn.classList.toggle("is-active", !photo);
  dom.viewPhotoBtn.setAttribute("aria-pressed", String(photo));
  dom.view3dBtn.setAttribute("aria-pressed", String(!photo));
}

/** Show/hide the main layers and bottom bar. */
function renderStates() {
  const hasSession = Boolean(state.session);
  const show3d =
    hasSession && state.phase === "verdict" && state.has3d && state.view === "3d";

  const calibrating = hasSession && state.phase === "calibrate";

  show(dom.dropzone, !hasSession);
  dom.clearBtn.hidden = !hasSession;
  dom.clearBtn.disabled = !canClear();
  show(dom.photoLayer, hasSession && !show3d && !calibrating);
  show(dom.view3dLayer, show3d);
  show(dom.calibLayer, calibrating);
  show(dom.calibPanel, calibrating);
  dom.actionList.parentElement.hidden = calibrating;
  if (calibrating) calibrator.redraw();
  show(dom.viewToggle, state.phase === "verdict" && state.has3d);
  state.phase === "review" ? show(dom.reviewCard, true) : show(dom.reviewCard, false);

  const showTimeline = hasSession && state.session.kind === "video";
  const showCamera = state.phase === "verdict" && state.has3d && hasSession;
  show(dom.timeline, showTimeline);
  show(dom.cameraControls, showCamera);
  show(dom.bottomBar, showTimeline || showCamera);

  dom.replayBtn.disabled = !showCamera;
  updatePresetButtons();
  renderOptionButtons();
  updateViewToggle();
  updateTimeline();
  applyStatus();
  applyReconnect();
}

/** Top-bar mock indicator + sample shortcut. */
function renderMockChip() {
  renderAutoReviewSwitch();
  dom.mockChip.hidden = !state.mock;
  dom.mockSample.hidden = !state.mock;
}

/** Full re-render of everything that is not a canvas frame. */
function renderAll() {
  renderPhaseSteps();
  renderFileMeta();
  renderPrimary();
  renderVerdict();
  renderChecks();
  renderActions();
  renderSteps();
  renderReviewCard();
  renderStates();
  renderExport();
  renderMockChip();
  updateGizmo();
  updateReplayButton();
  drawPhoto();
}

/** Export menu availability — only enabled at VERDICT. */
function renderExport() {
  const enabled =
    state.phase === "verdict" && Boolean(state.verdict) && state.verdict.verdict !== "unknown";
  dom.exportBtn.disabled = !enabled;
  for (const item of dom.exportMenu.querySelectorAll(".menu-item")) {
    item.disabled = !enabled;
  }
  if (!enabled && !dom.exportMenu.hidden) closePopover(dom.exportMenu, dom.exportBtn);
}

/* ==========================================================================
   Phase transitions
   ========================================================================== */

let verdictAnimTimer = null;
let bandSettleTimer = null;
let marginRaf = 0;
let marginAnimating = false;

/** How long the full-size broadcast band holds before it docks. */
const BAND_HOLD_MS = 3400;

/** Count a margin from 0 up to its value in both the band and the panel. */
function countMargin(target, delay, duration) {
  cancelAnimationFrame(marginRaf);
  const finalText = target.toFixed(2);
  marginAnimating = true;
  setText(dom.vbNum, "0.00");
  setText(dom.verdictNum, "0.00");
  const start = performance.now() + delay;
  const step = (now) => {
    const t = clamp((now - start) / duration, 0, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    const text = t >= 1 ? finalText : (target * eased).toFixed(2);
    setText(dom.vbNum, text);
    setText(dom.verdictNum, text);
    if (t < 1) {
      marginRaf = requestAnimationFrame(step);
    } else {
      marginRaf = 0;
      marginAnimating = false;
    }
  };
  marginRaf = requestAnimationFrame(step);
}

/** One orchestrated broadcast reveal per transition into VERDICT. */
function triggerVerdictAnimation() {
  const band = dom.verdictBand;
  const reduced =
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  dom.verdictBlock.classList.remove("is-enter");
  band.classList.remove("is-reveal", "is-settled");
  void band.offsetWidth;
  clearTimeout(verdictAnimTimer);
  clearTimeout(bandSettleTimer);
  cancelAnimationFrame(marginRaf);
  marginAnimating = false;

  const verdict = state.verdict;
  const margin = verdict ? Math.abs(Number(verdict.margin_m)) : NaN;
  const hasMargin =
    Boolean(verdict) &&
    verdict.margin_m != null &&
    Number.isFinite(margin) &&
    verdict.verdict !== "unknown";

  if (reduced) {
    band.classList.add("is-settled");
    renderVerdict();
    return;
  }

  band.classList.add("is-reveal");
  dom.verdictBlock.classList.add("is-enter");
  if (hasMargin) countMargin(margin, 700, 950);

  verdictAnimTimer = setTimeout(() => {
    dom.verdictBlock.classList.remove("is-enter");
  }, 1800);
  bandSettleTimer = setTimeout(() => {
    band.classList.remove("is-reveal");
    band.classList.add("is-settled");
  }, BAND_HOLD_MS);
}

/** Change phase, updating the viewport and every derived block. */
function setPhase(next) {
  const previous = state.phase;
  state.phase = next;
  dom.viewport.dataset.phase = next;
  dom.app.dataset.phase = next;
  renderAll();
  if (next === "verdict" && previous !== "verdict") triggerVerdictAnimation();
}

/** Show/hide the indeterminate progress line. */
function showProgress(visible) {
  dom.progressLine.hidden = !visible;
}

/** Merge a Phase1 response into state without dropping the user's choices. */
function applyPhase1(payload) {
  if (!payload || typeof payload !== "object") return;
  state.phase1 = payload;
  state.players = Array.isArray(payload.players) ? payload.players : [];
  state.teamColors = normaliseColors(payload.team_colors);
  state.stageVersion += 1;
  state.hoveredIndex = null;
  dom.playerTip.hidden = true;
  applyTeamColors();
}

/** Normalise team_colors keys (the API uses "1"/"2"). */
function normaliseColors(colors) {
  const result = { ...TEAM_COLORS };
  if (colors && typeof colors === "object") {
    if (colors["1"]) result[1] = String(colors["1"]);
    if (colors["2"]) result[2] = String(colors["2"]);
  }
  return result;
}

/** Expose team colours as CSS properties (overlay uses the JS copy). */
function applyTeamColors() {
  const root = document.documentElement;
  root.style.setProperty("--team1", state.teamColors[1]);
  root.style.setProperty("--team2", state.teamColors[2]);
}

/* ==========================================================================
   File open / frame scrubbing
   ========================================================================== */

let frameTimer = null;

/** Forget everything in the page that belongs to the open file. */
function resetFileState() {
  close3D();
  clearTimeout(frameTimer);
  frameTimer = null;
  clearExportToasts();
  dom.view3dCanvas.width = dom.view3dCanvas.width;
  state.busy = false;
  state.phase1 = null;
  state.players = [];
  state.verdict = null;
  state.has3d = false;
  state.calReturnVerdict = null;
  state.beforeEdit = null;
  calibrator.close();
  if (autoReview) autoReview.reset();
  resetReviewChoices();
  state.frameIndex = 0;
  state.view = "photo";
  state.preset = "broadcast";
  state.replaying = false;
  state.pendingReplay = false;
  state.camera = clone(DEFAULT_CAMERA);
  state.hoveredIndex = null;
  state.dragBox = null;
  state.pendingPlayer = null;
  state.stageVersion = 0;
  state.teamColors = normaliseColors(null);
  applyTeamColors();
  closeCtxMenu();
  closeLightbox();
  hintHidden = false;
  dom.viewHint.classList.remove("is-hidden");
  setStatus(null);
}

/** Show a newly opened file. */
function loadSession(session) {
  resetFileState();
  state.session = session;
  dom.photoImg.src = api.frameUrl(session.id, 0);
  setPhase("loaded");
}

/** Whether the open file can be deleted now. An Auto-review run can be stopped, but Detect and Analyze can't. */
function canStopWork() {
  return !state.uploading && !state.playerSaving && (!state.busy || state.phase === "autoreview");
}

function canClear() {
  return Boolean(state.session) && canStopWork();
}

/** Delete a session on the server. A failure is reported, since its data may still be there. */
function deleteOnServer(sessionId) {
  api.deleteSession(sessionId).catch((error) => {
    pushNotice("Could not delete the old frame", error.message || String(error), true);
  });
}

/** Delete the open file and everything made from it, and go back to the empty screen. */
function clearEverything() {
  if (!canClear()) return;
  if (!window.confirm("Delete this frame and everything made from it, including its exports on the server?")) return;
  const sessionId = state.session.id;
  closeAllPopovers();
  resetFileState();
  state.session = null;
  dom.photoImg.removeAttribute("src");
  showProgress(false);
  setPhase("empty");
  deleteOnServer(sessionId);
}

/** Upload a file chosen or dropped by the user. The old file goes once the new one has loaded. */
async function handleFile(file) {
  if (!file || !canStopWork()) return;
  const previous = state.session ? state.session.id : null;
  closeAllPopovers();
  state.uploading = true;
  showProgress(true);
  renderPrimary();
  try {
    const session = await api.upload(file);
    loadSession(session);
    if (previous) deleteOnServer(previous);
  } catch (error) {
    pushNotice("Could not open file", error.message || String(error), true);
  } finally {
    state.uploading = false;
    showProgress(state.phase === "detecting" || state.phase === "analyzing");
    renderAll();
  }
}

/** Load frame N (debounced while the range is being dragged). */
function requestFrame(index, immediate = false) {
  if (!state.session || state.session.kind !== "video" || state.busy) return;

  const total = Math.max(1, Number(state.session.frame_count) || 1);
  const next = clamp(Math.round(index), 0, total - 1);
  if (next === state.frameIndex) return;

  // A different pass frame invalidates detections: force a fresh review.
  if (state.phase === "review" || state.phase === "verdict" || state.phase === "calibrate") {
    resetDetections();
    pushNotice("Frame changed", "Run Detect again for the new pass frame.");
  }

  state.frameIndex = next;
  updateTimeline();

  clearTimeout(frameTimer);
  const load = () => {
    frameTimer = null;
    if (!state.session) return;
    const sessionId = state.session.id;
    const frameIndex = state.frameIndex;
    dom.photoImg.src = api.frameUrl(sessionId, frameIndex);
    api.selectFrame(sessionId, frameIndex).catch((error) => {
      if (!state.session || state.session.id !== sessionId) return;
      pushNotice("Frame not selected", error.message || String(error), true);
    });
  };

  if (immediate) load();
  else frameTimer = setTimeout(load, 80);
}

/** Clear Phase1/verdict data after the pass frame changes. */
function resetDetections() {
  close3D();
  calibrator.close();
  if (autoReview) autoReview.reset();
  state.calReturnVerdict = null;
  state.beforeEdit = null;
  state.phase1 = null;
  state.players = [];
  state.verdict = null;
  state.has3d = false;
  resetReviewChoices();
  state.view = "photo";
  state.replaying = false;
  state.pendingReplay = false;
  clearTimeout(frameTimer);
  frameTimer = null;
  setPhase("loaded");
}

/* ==========================================================================
   Detect / analyze / edit players
   ========================================================================== */

/** Forget the Review choices: they belong to one detection of one frame. */
function resetReviewChoices() {
  state.attackingSide = null;
  state.attackingTeam = null;
  state.keeperNotInFrame = false;
}

/** Phase 1. Always lands in REVIEW. */
async function detect() {
  if (!state.session || state.busy) return;
  state.busy = true;
  closeCtxMenu();
  if (autoReview) autoReview.reset();
  setPhase("detecting");
  showProgress(true);
  try {
    const payload = await api.detect(state.session.id);
    applyPhase1(payload);
    resetReviewChoices();
    state.verdict = null;
    state.calReturnVerdict = null;
    state.beforeEdit = null;
    state.has3d = false;
    state.view = "photo";
    state.replaying = false;
    state.pendingReplay = false;
    setPhase("review");
  } catch (error) {
    pushNotice("Detect failed", error.message || String(error), true);
    setPhase("loaded");
  } finally {
    state.busy = false;
    showProgress(false);
    renderAll();
  }
}

/* ==========================================================================
   Auto-review
   ========================================================================== */

/** Available only when the data source can run it. */
function autoReviewAvailable() {
  return typeof api.startAutoReview === "function";
}

function autoReviewEnabled() {
  return state.autoReviewOn && autoReviewAvailable();
}

/** Detect, then follow the run. `busy` stays set until the run reaches an ending. */
async function runAutoReview() {
  if (!state.session || state.busy) return;
  const sessionId = state.session.id;
  state.busy = true;
  closeCtxMenu();
  autoReview.reset();
  setPhase("detecting");
  showProgress(true);
  try {
    const payload = await api.detect(sessionId);
    const started = await api.startAutoReview(sessionId, dom.arScenario.value);
    if (!state.session || state.session.id !== sessionId) return;
    applyPhase1(started.phase1 || payload);
    resetReviewChoices();
    state.verdict = null;
    state.calReturnVerdict = null;
    state.beforeEdit = null;
    state.has3d = false;
    state.view = "photo";
    setPhase("autoreview");
    autoReview.start(sessionId, started.job_id);
  } catch (error) {
    state.busy = false;
    pushNotice("Auto-review failed", error.message || String(error), true);
    setPhase("loaded");
  } finally {
    showProgress(false);
    renderAll();
  }
}

/** Change the local scene for one event, so the overlay stays in step with the effects. */
function applyAutoReviewEvent(event) {
  const findIndex = () => state.players.findIndex((player) => player.id === event.player_id);
  switch (event.type) {
    case "player_added":
      state.players = [
        ...state.players,
        { id: event.player_id, bbox: event.bbox, team: event.team, role: "player", confidence: 0.8 },
      ];
      break;
    case "box_removed":
      state.players = state.players.filter((player) => player.id !== event.player_id);
      break;
    case "team_changed": {
      const index = findIndex();
      if (index >= 0) state.players[index] = { ...state.players[index], team: event.to_team };
      break;
    }
    case "role_changed": {
      const index = findIndex();
      if (index >= 0) state.players[index] = { ...state.players[index], role: event.role };
      break;
    }
    case "side_set":
      state.attackingSide = event.side;
      break;
    case "attacking_team_set":
      state.attackingTeam = event.team;
      break;
    default:
      return;
  }
  renderVerdict();
  drawPhoto();
}

/** Copy the saved scene from a finished run, so what shows is what the call used. */
function applyAutoReviewScene(result) {
  state.players = result.players;
  state.attackingSide = result.attacking_side;
  state.attackingTeam = result.attacking_team;
  state.keeperNotInFrame = result.keeper === "not_in_frame";
}

function onAutoReviewResult(result, info) {
  state.busy = false;
  state.replaying = false;
  state.pendingReplay = false;
  close3D();
  if (info.ending === "not_usable") {
    state.players = [];
    state.verdict = null;
    state.has3d = false;
    setPhase("loaded");
    return;
  }
  applyAutoReviewScene(result);
  state.verdict = result.verdict;
  state.has3d = Boolean(result.verdict && result.verdict.view3d_available);
  state.view = "photo";
  state.stageVersion += 1;
  if (!result.verdict || reviewMissing().length) {
    state.verdict = null;
    state.has3d = false;
    setPhase("review");
  } else {
    setPhase("verdict");
  }
  renderAll();
  drawPhoto();
}

function onAutoReviewByHand(result) {
  state.busy = false;
  if (result && result.ending === "not_usable") {
    detect();
    return;
  }
  if (result && result.reason) pushNotice("Auto-review needs you", result.reason);
  if (result) applyAutoReviewScene(result);
  state.verdict = null;
  state.has3d = false;
  setPhase("review");
  renderAll();
  drawPhoto();
}

function onAutoReviewFailed(message) {
  state.busy = false;
  pushNotice("Auto-review stopped", `${message} Check this frame by hand.`, true);
  state.verdict = null;
  setPhase(state.players.length ? "review" : "loaded");
  renderAll();
  drawPhoto();
}

/** Wire the switch and create the controller. */
function setupAutoReview() {
  autoReview = createAutoReview({
    api: {
      startAutoReview: (...args) => api.startAutoReview(...args),
      getAutoReview: (...args) => api.getAutoReview(...args),
      answerAutoReview: (...args) => api.answerAutoReview(...args),
      undoAutoReview: (...args) => api.undoAutoReview(...args),
      cancelJob: (...args) => api.cancelJob(...args),
    },
    canvas: byId("autoReviewCanvas"),
    viewport: dom.viewport,
    getImageRect: imageRect,
    lineColor: LINE_COLOR,
    applyEvent: applyAutoReviewEvent,
    findPlayer: (id) => state.players.find((player) => player.id === id),
    playersOfTeam: (team) => state.players.filter((player) => player.team === team),
    teamColor: (team) => state.teamColors[team] || COLORS.accent,
    onShow: () => setView("photo"),
    onResult: onAutoReviewResult,
    onReviewByHand: onAutoReviewByHand,
    onFailed: onAutoReviewFailed,
    onUndoFailed: (message) => pushNotice("Undo failed", message, true),
  });
  dom.arSwitch.addEventListener("click", () => {
    state.autoReviewOn = !state.autoReviewOn;
    try {
      localStorage.setItem("var.autoReview", state.autoReviewOn ? "1" : "0");
    } catch {
      // The switch still works for this page without saved settings.
    }
    renderAutoReviewSwitch();
    renderPrimary();
    renderVerdict();
  });
  try {
    state.autoReviewOn = localStorage.getItem("var.autoReview") === "1";
  } catch {
    state.autoReviewOn = false;
  }
}

function renderAutoReviewSwitch() {
  dom.arSwitchWrap.hidden = !autoReviewAvailable();
  dom.arScenario.hidden = !state.mock || !state.autoReviewOn || state.phase !== "loaded";
  dom.arSwitch.setAttribute("aria-checked", state.autoReviewOn ? "true" : "false");
  dom.arSwitch.classList.toggle("is-on", state.autoReviewOn);
  const locked = state.busy || state.phase === "autoreview";
  dom.arSwitch.disabled = locked;
  dom.arScenario.disabled = locked;
}

/** Phase 2. The Review choices must all have been made. */
async function analyze() {
  if (!state.session || state.busy || state.playerSaving) return;
  const missing = reviewMissing();
  if (missing.length) {
    pushNotice("Finish the review first", `Still needed: ${missing.join(", ")}.`);
    return;
  }

  const previousPhase = state.phase;
  const previousVerdict = state.verdict;
  const previousHas3d = state.has3d;
  const previousView = state.view;
  const wasCalibrating = previousPhase === "calibrate";

  state.busy = true;
  closeCtxMenu();
  if (wasCalibrating) calibrator.close();
  state.replaying = false;
  state.pendingReplay = false;
  if (state.has3d) {
    close3D();
    state.has3d = false;
    state.view = "photo";
  }
  setPhase("analyzing");
  showProgress(true);

  try {
    const verdict = await api.analyze(
      state.session.id,
      state.attackingSide,
      state.attackingTeam,
      keeperChoice(),
    );
    state.verdict = verdict;
    state.attackingSide = verdict.attacking_side || state.attackingSide;
    state.has3d = Boolean(verdict.view3d_available);
    state.view = state.has3d ? "3d" : "photo";
    state.stageVersion += 1;
    setPhase("verdict");
    if (state.has3d) connect3D();
  } catch (error) {
    pushNotice("Analyze failed", error.message || String(error), true);
    state.verdict = previousVerdict;
    state.has3d = previousHas3d;
    state.view = previousView;
    setPhase(previousPhase);
    if (wasCalibrating) calibrator.open();
    if (previousHas3d) connect3D();
  } finally {
    state.busy = false;
    showProgress(false);
    renderAll();
  }
}

/** The single primary-action dispatcher (button and F5). */
function primaryAction() {
  if (state.uploading || state.busy) return;
  if (state.phase === "loaded") {
    if (autoReviewEnabled()) runAutoReview();
    else detect();
    return;
  }
  if (state.phase === "review") {
    const missing = reviewMissing();
    if (missing.length) {
      const selector = !state.attackingSide
        ? ".side-btn"
        : !state.attackingTeam
          ? ".team-btn"
          : "#notInFrameBtn";
      const target = dom.reviewCard.querySelector(selector);
      if (target) target.focus();
      pushNotice("Finish the review first", `Still needed: ${missing.join(", ")}.`);
      return;
    }
    openCalibrate();
    return;
  }
  if (state.phase === "calibrate") {
    finishCalibrate();
    return;
  }
  if (state.phase === "verdict") analyze();
}

/** Enter the Calibrate step. */
function openCalibrate() {
  setPhase("calibrate");
  calibrator.open();
  dom.calibCanvas.focus({ preventScroll: true });
}

/** Save what's on screen, if anything changed, then run the line and verdict. */
async function finishCalibrate() {
  const info = calibrator.info();
  if (!info.canContinue || state.busy) return;
  state.busy = true;
  renderPrimary();
  const before = state.calReturnVerdict;
  try {
    const changed = await calibrator.commit();
    state.beforeEdit = changed ? before : null;
    state.calReturnVerdict = null;
  } catch (error) {
    pushNotice("Could not apply the calibration", error.message || String(error), true);
    return;
  } finally {
    state.busy = false;
    renderPrimary();
  }
  analyze();
}

/** No camera is possible here: analyze with the automatic result as it is. */
async function continueWithout3d() {
  if (state.busy) return;
  state.busy = true;
  try {
    await calibrator.discard();
    state.calReturnVerdict = null;
    state.beforeEdit = null;
  } catch (error) {
    pushNotice("Could not continue", error.message || String(error), true);
    return;
  } finally {
    state.busy = false;
  }
  analyze();
}

/** Leave Calibrate without saving: back to the verdict or to the review. */
function leaveCalibrate() {
  calibrator.close();
  const verdict = state.calReturnVerdict;
  state.calReturnVerdict = null;
  if (verdict) {
    state.verdict = verdict;
    state.has3d = Boolean(verdict.view3d_available);
    state.view = state.has3d ? "3d" : "photo";
    setPhase("verdict");
    if (state.has3d) connect3D();
    return;
  }
  setPhase("review");
}

/** Open Calibrate from the verdict; the verdict stays hidden while editing. */
function editCalibration() {
  state.calReturnVerdict = state.verdict;
  close3D();
  state.has3d = false;
  state.verdict = null;
  state.view = "photo";
  state.replaying = false;
  state.pendingReplay = false;
  openCalibrate();
}

/** Confirm the attack direction. */
function chooseSide(side) {
  if (state.phase !== "review") return;
  if (side !== "left" && side !== "right") return;
  state.attackingSide = side;
  renderReviewChoices();
}

/** Choose the attacking team. */
function chooseTeam(team) {
  if (state.phase !== "review") return;
  state.attackingTeam = team;
  renderReviewChoices();
}

/** Say that the keeper isn't in the frame, or take that back. */
function toggleNotInFrame() {
  if (state.phase !== "review") return;
  state.keeperNotInFrame = keeperChoice() !== "not_in_frame";
  if (state.keeperNotInFrame && state.players.some((player) => player.role === "goalkeeper")) {
    commitPlayers(
      state.players.map((player) =>
        player.role === "goalkeeper" ? { ...player, role: "player" } : player,
      ),
    );
  }
  renderReviewChoices();
}

/** Make one player the keeper; there is only ever one. */
function markKeeper(index) {
  state.keeperNotInFrame = false;
  closeCtxMenu();
  commitPlayers(
    state.players.map((player, i) => {
      if (i === index) return { ...player, role: "goalkeeper" };
      return player.role === "goalkeeper" ? { ...player, role: "player" } : player;
    }),
  );
}

/** Redraw everything that shows a Review choice. */
function renderReviewChoices() {
  renderReviewCard();
  renderChecks();
  renderPrimary();
  renderVerdict();
}

/** PUT the new player list and redraw everything from the response. */
async function commitPlayers(players) {
  if (!state.session || state.playerSaving) return;
  const sessionId = state.session.id;
  state.playerSaving = true;
  renderPrimary();
  try {
    const payload = await api.putPlayers(sessionId, players);
    // A slow PUT must not leak into a file opened while it was in flight.
    if (!state.session || state.session.id !== sessionId) return;
    applyPhase1(payload);
    renderAll();
  } catch (error) {
    pushNotice("Could not update players", error.message || String(error), true);
  } finally {
    state.playerSaving = false;
    renderPrimary();
    drawPhoto();
  }
}

/** Apply a patch to one reviewed player. */
function updatePlayer(index, patch) {
  const current = state.players[index];
  if (!current) return;
  const next = state.players.map((player, i) =>
    i === index ? { ...player, ...patch } : player,
  );
  closeCtxMenu();
  commitPlayers(next);
}

/** Return from VERDICT to REVIEW without re-running detection. */
function editPlayers() {
  close3D();
  state.has3d = false;
  state.verdict = null;
  state.view = "photo";
  state.replaying = false;
  state.pendingReplay = false;
  setPhase("review");
}

/* ==========================================================================
   Photo overlay + player interactions
   ========================================================================== */

let pulseRaf = 0;

/** Image rectangle (contain) inside the viewport, in CSS pixels. */
function imageRect(width, height) {
  const imageWidth = Number(state.session && state.session.width) || 16;
  const imageHeight = Number(state.session && state.session.height) || 9;
  const scale = Math.min(width / imageWidth, height / imageHeight);
  const w = imageWidth * scale;
  const h = imageHeight * scale;
  return { x: (width - w) / 2, y: (height - h) / 2, w, h, scale };
}

/** Tooltip text for a player. */
function playerTipText(player) {
  const parts = [`#${player.id == null ? "new" : player.id}`];
  if (player.role === "referee") parts.push("Referee");
  else if (player.team) parts.push(`Team ${player.team}`);
  else parts.push("No team");
  if (player.role === "goalkeeper") parts.push("GK");
  const confidence = confidenceLabel(player);
  if (confidence) parts.push(confidence);
  return parts.join(" · ");
}

/** Draw the photo overlay (boxes, line, rings, drag preview). */
function drawPhoto(timestamp) {
  const canvas = dom.overlayCanvas;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const vw = dom.viewport.clientWidth;
  const vh = dom.viewport.clientHeight;
  if (!vw || !vh) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const targetW = Math.round(vw * dpr);
  const targetH = Math.round(vh * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, vw, vh);

  if (!state.session || dom.photoLayer.hidden) return;

  const rect = imageRect(vw, vh);
  if (rect.scale <= 0) return;
  const toScreen = (x, y) => [rect.x + x * rect.scale, rect.y + y * rect.scale];

  // Offside line first: it sits under the boxes and rings.
  const lineColor = drawOffsideLine(ctx, rect);

  // Reviewed players: a ring on the ground and a label above the head.
  if (state.phase === "review" || state.phase === "verdict" || state.phase === "autoreview") {
    const time = timestamp == null ? performance.now() : timestamp;
    const beat = state.pulse ? (Math.sin(time / 380) + 1) / 2 : 0.5;
    const labels = [];
    // Nearest players last, so their rings sit on top.
    const ordered = state.players
      .map((player, index) => ({ player, index }))
      .filter(({ player }) => player && Array.isArray(player.bbox) && player.bbox.length >= 4)
      .sort((a, b) => a.player.bbox[3] - b.player.bbox[3]);
    for (const { player, index } of ordered) {
      const [x1, y1, x2, y2] = player.bbox;
      const [sx1, sy1] = toScreen(x1, y1);
      const [sx2, sy2] = toScreen(x2, y2);
      if (sx2 <= sx1 || sy2 <= sy1) continue;
      const box = { x1: sx1, y1: sy1, x2: sx2, y2: sy2 };
      const color = markerColor(player);
      const hovered = state.hoveredIndex === index;
      const emphasis = playerEmphasis(player);
      drawPlayerRing(ctx, box, {
        color,
        keeper: player.role === "goalkeeper",
        hovered,
        emphasis,
        beat,
      });
      if (hovered) drawHoverBrackets(ctx, box, color);
      labels.push({ player, box, color, emphasis });
    }
    if (state.showLabels) drawPlayerLabels(ctx, labels);
  }

  // Drag-to-add preview.
  if (state.dragBox) {
    const { start, current } = state.dragBox;
    const x1 = Math.min(start.x, current.x);
    const y1 = Math.min(start.y, current.y);
    const x2 = Math.max(start.x, current.x);
    const y2 = Math.max(start.y, current.y);
    const [sx1, sy1] = toScreen(x1, y1);
    const [sx2, sy2] = toScreen(x2, y2);
    ctx.save();
    ctx.fillStyle = hexToRgba(COLORS.accent, 0.1);
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.fillRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
    ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
    ctx.restore();
  }

  // Provisional box waiting for Team 1 / Team 2.
  if (state.pendingPlayer) {
    const [x1, y1, x2, y2] = state.pendingPlayer.bbox;
    const [sx1, sy1] = toScreen(x1, y1);
    const [sx2, sy2] = toScreen(x2, y2);
    ctx.save();
    ctx.strokeStyle = COLORS.ink;
    ctx.lineWidth = 1.6;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
    ctx.restore();
  }

  // Slow pulse on the offside ring (only when the option is on).
  const shouldPulse =
    state.phase === "verdict" &&
    state.pulse &&
    !dom.photoLayer.hidden &&
    dom.stageLightbox.hidden &&
    Boolean(lineColor) &&
    Boolean(
      state.verdict &&
        Array.isArray(state.verdict.offside_player_ids) &&
        state.verdict.offside_player_ids.length,
    );
  if (shouldPulse && !pulseRaf) {
    pulseRaf = requestAnimationFrame((nextTimestamp) => {
      pulseRaf = 0;
      drawPhoto(nextTimestamp);
    });
  }
}

/** Which of the two deciding players this is, on the verdict screen. */
function playerEmphasis(player) {
  const verdict = state.verdict;
  if (state.phase !== "verdict" || !verdict) return null;
  if (Array.isArray(verdict.offside_player_ids) && verdict.offside_player_ids.includes(player.id)) {
    return "offside";
  }
  if (verdict.last_defender && verdict.last_defender.id === player.id) return "defender";
  return null;
}

/** Draw the offside line; returns the call's colour or null. */
function drawOffsideLine(ctx, rect) {
  const verdict = state.verdict;
  if (
    state.phase !== "verdict" ||
    !verdict ||
    (autoReview && autoReview.ownsLine()) ||
    !Array.isArray(verdict.line_image_points) ||
    verdict.line_image_points.length < 2
  ) {
    return null;
  }

  const color =
    verdict.verdict === "offside"
      ? COLORS.red
      : verdict.verdict === "onside"
        ? COLORS.green
        : COLORS.amber;
  const [a, b] = verdict.line_image_points;
  const ax = rect.x + a[0] * rect.scale;
  const ay = rect.y + a[1] * rect.scale;
  const bx = rect.x + b[0] * rect.scale;
  const by = rect.y + b[1] * rect.scale;

  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = hexToRgba(LINE_COLOR, 0.18);
  ctx.lineWidth = 10;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.strokeStyle = LINE_COLOR;
  ctx.lineWidth = 2;
  ctx.shadowColor = LINE_COLOR;
  ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.restore();
  return color;
}

/** Convert a pointer event to image coordinates and viewport CSS coordinates. */
function pointerToImage(event) {
  if (!state.session) return null;
  const bounds = dom.overlayCanvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return null;
  const rect = imageRect(bounds.width, bounds.height);
  if (!rect.scale) return null;
  const sx = event.clientX - bounds.left;
  const sy = event.clientY - bounds.top;
  return {
    x: (sx - rect.x) / rect.scale,
    y: (sy - rect.y) / rect.scale,
    sx,
    sy,
  };
}

/** Topmost player under the pointer, or null. */
function hitTestPlayer(event) {
  if (state.phase !== "review" && state.phase !== "verdict") return null;
  const point = pointerToImage(event);
  if (!point) return null;
  for (let index = state.players.length - 1; index >= 0; index -= 1) {
    const player = state.players[index];
    if (!player || !Array.isArray(player.bbox) || player.bbox.length < 4) continue;
    const [x1, y1, x2, y2] = player.bbox;
    if (point.x >= x1 && point.x <= x2 && point.y >= y1 && point.y <= y2) {
      return index;
    }
  }
  return null;
}

/** Clear hover state + tooltip. */
function clearHover() {
  if (state.hoveredIndex == null) {
    dom.playerTip.hidden = true;
    return;
  }
  state.hoveredIndex = null;
  dom.playerTip.hidden = true;
  drawPhoto();
}

/** Hover a player: highlight it and position the tooltip. */
function updateHover(event) {
  if (state.dragBox) return;
  const index = hitTestPlayer(event);
  if (index == null) {
    clearHover();
    return;
  }
  if (state.hoveredIndex !== index) {
    state.hoveredIndex = index;
    drawPhoto();
  }
  const player = state.players[index];
  if (!player) return;
  setText(dom.playerTip, playerTipText(player));
  dom.playerTip.hidden = false;
  const viewportBounds = dom.viewport.getBoundingClientRect();
  const x = clamp(event.clientX - viewportBounds.left, 0, Math.max(0, viewportBounds.width - 150));
  const y = clamp(event.clientY - viewportBounds.top, 0, Math.max(0, viewportBounds.height - 30));
  dom.playerTip.style.left = `${x}px`;
  dom.playerTip.style.top = `${y}px`;
}

/** Pointer down on the photo overlay. */
function onOverlayPointerDown(event) {
  if (event.button !== 0 || state.phase !== "review") return;
  const index = hitTestPlayer(event);
  if (index != null) {
    state.clickCandidate = index;
    return;
  }
  const point = pointerToImage(event);
  if (!point) return;
  state.clickCandidate = null;
  state.dragBox = {
    start: { x: point.x, y: point.y },
    current: { x: point.x, y: point.y },
    moved: false,
  };
  dom.overlayCanvas.setPointerCapture(event.pointerId);
  event.preventDefault();
}

/** Pointer move: hover or resize the new box. */
function onOverlayPointerMove(event) {
  if (state.dragBox) {
    const point = pointerToImage(event);
    if (!point) return;
    state.dragBox.current = { x: point.x, y: point.y };
    const dx = (state.dragBox.current.x - state.dragBox.start.x) * imageRect(
      dom.viewport.clientWidth,
      dom.viewport.clientHeight,
    ).scale;
    const dy = (state.dragBox.current.y - state.dragBox.start.y) * imageRect(
      dom.viewport.clientWidth,
      dom.viewport.clientHeight,
    ).scale;
    if (Math.hypot(dx, dy) > 4) state.dragBox.moved = true;
    drawPhoto();
    return;
  }
  updateHover(event);
}

/** Pointer up: finish a drag or open the player menu. */
function onOverlayPointerUp(event) {
  if (state.dragBox) {
    const drag = state.dragBox;
    state.dragBox = null;
    if (dom.overlayCanvas.hasPointerCapture(event.pointerId)) {
      dom.overlayCanvas.releasePointerCapture(event.pointerId);
    }

    if (drag.moved) {
      const x1 = clamp(Math.min(drag.start.x, drag.current.x), 0, Number(state.session.width) || 0);
      const y1 = clamp(Math.min(drag.start.y, drag.current.y), 0, Number(state.session.height) || 0);
      const x2 = clamp(Math.max(drag.start.x, drag.current.x), 0, Number(state.session.width) || 0);
      const y2 = clamp(Math.max(drag.start.y, drag.current.y), 0, Number(state.session.height) || 0);
      if (x2 - x1 >= 6 && y2 - y1 >= 6) {
        state.pendingPlayer = {
          id: null,
          bbox: [x1, y1, x2, y2],
          team: null,
          role: "player",
          confidence: null,
        };
        drawPhoto();
        openNewPlayerMenu(event.clientX, event.clientY);
      } else {
        drawPhoto();
      }
      return;
    }
    drawPhoto();
    return;
  }

  if (state.clickCandidate != null) {
    const index = state.clickCandidate;
    state.clickCandidate = null;
    if (state.players[index]) openPlayerMenu(index, event.clientX, event.clientY);
  }
}

/** Cancel a drag if the pointer is lost. */
function onOverlayPointerCancel() {
  state.dragBox = null;
  state.clickCandidate = null;
  drawPhoto();
}

/* ==========================================================================
   Context menu
   ========================================================================== */

let menuOutsideHandlerAttached = false;

/** Position and open the floating context menu. */
function openCtxMenu(items, header, clientX, clientY) {
  const menu = dom.ctxMenu;
  menu.replaceChildren();
  const head = make("div", "ctx-head", header);
  menu.append(head);

  for (const item of items) {
    const button = make(
      "button",
      item.danger ? "ctx-danger" : null,
      item.label,
    );
    button.type = "button";
    button.setAttribute("role", "menuitem");
    button.addEventListener("click", () => {
      item.onSelect();
    });
    menu.append(button);
  }

  menu.hidden = false;

  const bounds = dom.viewport.getBoundingClientRect();
  menu.style.left = "0px";
  menu.style.top = "0px";
  const width = menu.offsetWidth;
  const height = menu.offsetHeight;
  const left = clamp(clientX - bounds.left, 8, Math.max(8, bounds.width - width - 8));
  const top = clamp(clientY - bounds.top, 8, Math.max(8, bounds.height - height - 8));
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;

  const first = menu.querySelector("button");
  if (first) first.focus();
}

/** Close and discard any pending new player. */
function closeCtxMenu() {
  dom.ctxMenu.hidden = true;
  dom.ctxMenu.replaceChildren();
  if (state.pendingPlayer) {
    state.pendingPlayer = null;
    drawPhoto();
  }
}

/** Menu for an existing reviewed player. */
function openPlayerMenu(index, clientX, clientY) {
  const player = state.players[index];
  if (!player) return;
  const nextTeam = player.team === 1 ? 2 : player.team === 2 ? 1 : 1;
  const items = [
    {
      label: player.team ? `Switch to Team ${nextTeam}` : `Team ${nextTeam}`,
      onSelect: () => updatePlayer(index, { team: nextTeam }),
    },
    {
      label: player.role === "goalkeeper" ? "Mark as player" : "Mark goalkeeper",
      onSelect: () =>
        player.role === "goalkeeper"
          ? updatePlayer(index, { role: "player" })
          : markKeeper(index),
    },
    {
      label: player.role === "referee" ? "Mark as player" : "Not a player",
      danger: player.role !== "referee",
      onSelect: () =>
        updatePlayer(
          index,
          player.role === "referee"
            ? { role: "player", team: null }
            : { role: "referee", team: null },
        ),
    },
  ];
  openCtxMenu(items, playerTipText(player), clientX, clientY);
}

/** Menu shown after drawing a box: Team 1 / Team 2. */
function openNewPlayerMenu(clientX, clientY) {
  const pending = state.pendingPlayer;
  if (!pending) return;
  const items = [
    {
      label: "Team 1",
      onSelect: () => {
        const player = { ...pending, team: 1 };
        state.pendingPlayer = null;
        closeCtxMenu();
        commitPlayers([...state.players, player]);
      },
    },
    {
      label: "Team 2",
      onSelect: () => {
        const player = { ...pending, team: 2 };
        state.pendingPlayer = null;
        closeCtxMenu();
        commitPlayers([...state.players, player]);
      },
    },
    {
      label: "Cancel",
      onSelect: () => closeCtxMenu(),
    },
  ];
  openCtxMenu(items, "New player · choose a team", clientX, clientY);
}

/** Dismiss the player menu when clicking elsewhere. */
function installOutsideHandlers() {
  if (menuOutsideHandlerAttached) return;
  menuOutsideHandlerAttached = true;
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest(".menu-wrap")) closeAllPopovers();
    if (!dom.ctxMenu.hidden) {
      if (!event.target.closest("#ctxMenu")) closeCtxMenu();
    }
  });
}

/* ==========================================================================
   Live 3D view
   ========================================================================== */

let viewHandle = null;
let viewConnected = false;
let statusText = null;
let statusHideTimer = null;
let camRaf = 0;
let lastDraftAt = 0;
let drag3d = null;
let wheelTimer = null;
// No camera is sent until the server has told us the real one.
let cameraSynced = false;
const MAX_CAMERA_DISTANCE = 300;
// Offside line colour; matches LiveScene.LINE_RGB in the 3D view.
const LINE_COLOR = "#38d6ff";
let resizeTimer = null;
let hintHidden = false;

/** Whether the 3D canvas is the active surface. */
function is3dActive() {
  return (
    state.phase === "verdict" &&
    state.has3d &&
    state.view === "3d" &&
    !dom.view3dLayer.hidden
  );
}

/** Render size in device pixels, capped at 1920×1080 and aspect-preserving. */
function renderSize() {
  const rect = dom.viewport.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let width = Math.max(1, Math.round(rect.width * dpr));
  let height = Math.max(1, Math.round(rect.height * dpr));
  const factor = Math.min(1, 1920 / width, 1080 / height);
  width = Math.max(1, Math.round(width * factor));
  height = Math.max(1, Math.round(height * factor));
  return { width, height };
}

/** Current view options, sent with every render request. */
function viewOptions() {
  return {
    line_plane: state.linePlane,
    key_only: state.keyOnly,
    photo: state.photo3d,
    labels: state.showLabels,
    pulse: state.pulse,
  };
}

/** Normalise a camera message from the server. */
function normaliseCamera(camera) {
  if (!camera || typeof camera !== "object") return clone(DEFAULT_CAMERA);
  const target = Array.isArray(camera.target)
    ? [num(camera.target[0], 0), num(camera.target[1], 0), num(camera.target[2], 0)]
    : [0, 0, 0];
  return {
    target,
    yaw: wrapDegrees(num(camera.yaw, DEFAULT_CAMERA.yaw)),
    pitch: clamp(num(camera.pitch, DEFAULT_CAMERA.pitch), 3, 89),
    distance: clamp(num(camera.distance, DEFAULT_CAMERA.distance), 2, MAX_CAMERA_DISTANCE),
    fov: clamp(num(camera.fov, DEFAULT_CAMERA.fov), 1, 120),
  };
}

/** Send the current camera; called from the rAF throttle and on release. */
function sendCamera(quality) {
  if (!viewHandle || !viewConnected || !cameraSynced) return;
  const { width, height } = renderSize();
  viewHandle.sendCamera(clone(state.camera), width, height, quality, viewOptions());
}

/** Throttled draft render: one rAF, at most ~30 messages per second. */
function queueDraft() {
  if (camRaf) return;
  const tick = (timestamp) => {
    camRaf = 0;
    if (timestamp - lastDraftAt < 32) {
      camRaf = requestAnimationFrame(tick);
      return;
    }
    lastDraftAt = timestamp;
    sendCamera("draft");
  };
  camRaf = requestAnimationFrame(tick);
}

/** One full-quality render, cancelling any queued draft. */
function sendFullRender() {
  if (camRaf) {
    cancelAnimationFrame(camRaf);
    camRaf = 0;
  }
  sendCamera("full");
}

/** Create/replace the socket view. */
function connect3D() {
  close3D();
  if (!state.session || !state.has3d || state.phase !== "verdict") return;

  setStatus("Loading 3D…");
  cameraSynced = false;

  try {
    viewHandle = api.openView(state.session.id, {
      canvas: dom.view3dCanvas,
      onOpen: () => {
        viewConnected = true;
        applyReconnect();
        const size = renderSize();
        sendPreset(state.preset, size.width, size.height);
        if (state.pendingReplay) {
          state.pendingReplay = false;
          sendReplayFrames();
        }
      },
      onStatus: handle3dStatus,
      onCamera: (camera) => {
        if (!camera) return;
        state.camera = normaliseCamera(camera);
        cameraSynced = true;
        updateGizmo();
      },
      onDefenderPart: (name) => {
        if (!name || !state.verdict || !state.verdict.last_defender) return;
        state.verdict.last_defender.part = name.toLowerCase();
        renderVerdict();
      },
      onReplayEnd: () => {
        state.replaying = false;
        state.pendingReplay = false;
        updateReplayButton();
      },
      onClose: () => {
        viewConnected = false;
        viewHandle = null;
        state.replaying = false;
        state.pendingReplay = false;
        setStatus(null);
        updateReplayButton();
        applyReconnect();
      },
    });
    applyReconnect();
  } catch (error) {
    viewHandle = null;
    viewConnected = false;
    setStatus(null);
    applyReconnect();
    pushNotice("3D view unavailable", error.message || String(error), true);
  }
}

/** Close the socket without triggering the disconnected chip. */
function close3D() {
  if (viewHandle) {
    try {
      viewHandle.close();
    } catch {
      // Already gone.
    }
    viewHandle = null;
  }
  viewConnected = false;
  setStatus(null);
  if (camRaf) {
    cancelAnimationFrame(camRaf);
    camRaf = 0;
  }
  if (dom.view3dLayer) dom.view3dLayer.classList.remove("is-dragging");
  drag3d = null;
  applyReconnect();
}

/** Handle one status message from the server. */
function handle3dStatus(stage, message) {
  if (!state.has3d) return;
  clearTimeout(statusHideTimer);

  if (stage === "ready") {
    setStatus(message ? `Ready · ${message}` : "3D ready");
    statusHideTimer = setTimeout(() => setStatus(null), 1500);
    return;
  }
  if (stage === "error") {
    setStatus(message || "3D unavailable");
    if (state.replaying) {
      state.replaying = false;
      updateReplayButton();
    }
    return;
  }
  if (stage === "loading") {
    setStatus(message || "Loading 3D…");
    return;
  }
  if (stage === "proxy") {
    const text = message || "";
    if (!text) setStatus("Stand-in players");
    else if (/stand-in/i.test(text)) setStatus(text);
    else setStatus(`Stand-in players · ${text}`);
    return;
  }
  if (stage === "meshes") {
    setStatus(message || "Building body meshes…");
    return;
  }
  setStatus(message || stage);
}

/** Store + display the 3D status chip (visible only in the 3D view). */
function setStatus(text) {
  clearTimeout(statusHideTimer);
  statusHideTimer = null;
  statusText = text || null;
  applyStatus();
}

/** Apply chip visibility without discarding the last status text. */
function applyStatus() {
  const visible =
    Boolean(statusText) &&
    state.has3d &&
    state.phase === "verdict" &&
    state.view === "3d";
  show(dom.statusChip, visible);
  if (visible) setText(dom.statusChip, statusText);
}

/** Show the disconnected chip only when a live view should be present. */
function applyReconnect() {
  const disconnected =
    state.has3d &&
    state.phase === "verdict" &&
    state.view === "3d" &&
    !viewConnected &&
    !viewHandle;
  show(dom.reconnectChip, disconnected);
}

/** Send a preset request. */
function sendPreset(name, width, height) {
  state.preset = name;
  updatePresetButtons();
  if (!viewHandle || !viewConnected) return;
  const size = width && height ? { width, height } : renderSize();
  viewHandle.sendPreset(name, size.width, size.height, viewOptions());
}

/** User chose a camera preset (button, 1–4, R, double-click). */
function choosePreset(name) {
  if (!PRESETS.includes(name) || state.phase !== "verdict" || !state.has3d) return;
  state.preset = name;
  updatePresetButtons();
  hideHint();
  setView("3d");
  if (!viewHandle) {
    connect3D();
  } else if (viewConnected) {
    sendPreset(name);
  }
}

/** Send one replay request. */
function sendReplayFrames() {
  if (!viewHandle || !viewConnected) return;
  const { width, height } = renderSize();
  viewHandle.sendReplay(width, height, viewOptions());
}

/** Start a replay (switch to 3D first). */
function startReplay() {
  if (!state.has3d || state.phase !== "verdict") return;
  state.replaying = true;
  updateReplayButton();
  hideHint();
  setView("3d");

  if (!viewHandle) {
    state.pendingReplay = true;
    connect3D();
  } else if (viewConnected) {
    sendReplayFrames();
  } else {
    state.pendingReplay = true;
  }
}

/** Stop a running replay. */
function stopReplay() {
  if (viewHandle && viewConnected) viewHandle.sendStop();
  state.replaying = false;
  state.pendingReplay = false;
  updateReplayButton();
}

/** Replay button + panel action label. */
function toggleReplay() {
  if (state.replaying) stopReplay();
  else startReplay();
}

/** Refresh the bottom replay button and the matching panel action. */
function updateReplayButton() {
  setText(dom.replayBtn, state.replaying ? "Stop" : "Replay");
  dom.replayBtn.classList.toggle("is-playing", state.replaying);
  dom.replayBtn.setAttribute("aria-pressed", String(state.replaying));
  if (state.phase === "verdict") renderActions();
}

/** Photo | 3D. */
function setView(mode) {
  if (mode !== "photo" && mode !== "3d") return;
  if (mode === "3d" && (!state.has3d || state.phase !== "verdict")) return;
  state.view = mode;
  renderStates();
  drawPhoto();
  if (viewHandle && viewHandle.sendVisible) viewHandle.sendVisible(mode === "3d");
  if (mode === "3d") {
    applyStatus();
    applyReconnect();
    if (!viewHandle && state.has3d) connect3D();
    else if (viewConnected) sendFullRender();
  }
}

/** Hide the 3D hint after the first interaction. */
function hideHint() {
  if (hintHidden) return;
  hintHidden = true;
  dom.viewHint.classList.add("is-hidden");
}

/** Any manual camera manipulation becomes the Free preset. */
function markFreeCamera() {
  if (state.preset !== "free") {
    state.preset = "free";
    updatePresetButtons();
  }
}

/** Metres of pitch per screen pixel at the camera's target. */
function metresPerPixel() {
  const height = dom.view3dCanvas.clientHeight || 720;
  return (2 * state.camera.distance * Math.tan(rad(state.camera.fov) / 2)) / height;
}

/** Move the orbit target along the ground: metres to the camera's right, and away. */
function moveTarget(right, forward) {
  const yaw = rad(state.camera.yaw);
  state.camera.target[0] += Math.cos(yaw) * right - Math.sin(yaw) * forward;
  state.camera.target[2] += -Math.sin(yaw) * right - Math.cos(yaw) * forward;
}

/** Ground foreshortening: a screen pixel up covers more pitch at a low camera. */
function groundStretch() {
  return 1 / Math.max(Math.sin(rad(state.camera.pitch)), 0.2);
}

/** Pan so the pitch follows the pointer. */
function panCamera(dx, dy) {
  const scale = metresPerPixel();
  moveTarget(-dx * scale, dy * scale * groundStretch());
}

/** Orbit/pan pointer down. */
function on3dPointerDown(event) {
  if (!is3dActive()) return;
  if (event.button !== 0 && event.button !== 2) return;
  event.preventDefault();
  if (state.replaying) stopReplay();
  const mode = event.button === 2 || event.shiftKey ? "pan" : "orbit";
  drag3d = {
    mode,
    x: event.clientX,
    y: event.clientY,
    pointerId: event.pointerId,
  };
  try {
    dom.view3dCanvas.setPointerCapture(event.pointerId);
  } catch {
    // Pointer capture is best-effort.
  }
  dom.view3dLayer.classList.add("is-dragging");
  hideHint();
}

/** Orbit/pan pointer move. */
function on3dPointerMove(event) {
  if (!drag3d) return;
  const dx = event.clientX - drag3d.x;
  const dy = event.clientY - drag3d.y;
  if (!dx && !dy) return;
  drag3d.x = event.clientX;
  drag3d.y = event.clientY;

  markFreeCamera();
  if (drag3d.mode === "orbit") {
    // The scene follows the hand, so the camera moves the opposite way.
    state.camera.yaw = wrapDegrees(state.camera.yaw - dx * 0.3);
    state.camera.pitch = clamp(state.camera.pitch + dy * 0.2, 3, 89);
  } else {
    panCamera(dx, dy);
  }
  updateGizmo();
  queueDraft();
}

/** End an orbit/pan with one full render. */
function on3dPointerUp(event) {
  if (!drag3d) return;
  const pointerId = drag3d.pointerId;
  drag3d = null;
  dom.view3dLayer.classList.remove("is-dragging");
  try {
    if (dom.view3dCanvas.hasPointerCapture(pointerId)) {
      dom.view3dCanvas.releasePointerCapture(pointerId);
    }
  } catch {
    // Already released.
  }
  sendFullRender();
}

/** Wheel zoom toward the cursor; full render 150 ms after the last event. */
function on3dWheel(event) {
  if (!is3dActive()) return;
  event.preventDefault();
  hideHint();
  if (state.replaying) stopReplay();

  if (event.deltaY !== 0) {
    markFreeCamera();
    // Mouse notches and trackpad scrolls both feel even: ~10% per 100 delta units.
    const notches = clamp(event.deltaY / 100, -3, 3);
    const before = metresPerPixel();
    state.camera.distance = clamp(
      state.camera.distance * Math.pow(1.1, notches),
      2,
      MAX_CAMERA_DISTANCE,
    );
    const change = before - metresPerPixel();
    const rect = dom.view3dCanvas.getBoundingClientRect();
    const fromCentreX = event.clientX - rect.left - rect.width / 2;
    const fromCentreY = event.clientY - rect.top - rect.height / 2;
    moveTarget(fromCentreX * change, -fromCentreY * change * groundStretch());
    updateGizmo();
    queueDraft();
  }

  clearTimeout(wheelTimer);
  wheelTimer = setTimeout(() => {
    wheelTimer = null;
    sendFullRender();
  }, 150);
}

/** Draw the bottom-right axis gizmo for the current yaw/pitch. */
function updateGizmo() {
  const yaw = rad(state.camera.yaw);
  const pitch = rad(state.camera.pitch);
  const scale = 22;
  const origin = 38;
  const axes = {
    x: [Math.cos(yaw), Math.sin(yaw) * Math.sin(pitch)],
    z: [-Math.sin(yaw), Math.cos(yaw) * Math.sin(pitch)],
    y: [0, -Math.cos(pitch)],
  };

  const place = (line, label, vector) => {
    const x = origin + vector[0] * scale;
    const y = origin + vector[1] * scale;
    if (line) {
      line.setAttribute("x2", x.toFixed(1));
      line.setAttribute("y2", y.toFixed(1));
    }
    if (label) {
      label.setAttribute("x", (origin + vector[0] * (scale + 7)).toFixed(1));
      label.setAttribute("y", (origin + vector[1] * (scale + 7)).toFixed(1));
    }
  };

  place(dom.gizmoX, dom.gizmoXLabel, axes.x);
  place(dom.gizmoZ, dom.gizmoZLabel, axes.z);
  place(dom.gizmoY, dom.gizmoYLabel, axes.y);
}

/* ==========================================================================
   Export toasts
   ========================================================================== */

/** Create a stacked toast; returns its parts. */
function makeToast(title, extraClass = "") {
  const element = make("div", `toast ${extraClass}`.trim());
  const top = make("div", "toast-top");
  const titleElement = make("div", "toast-title", title);
  const close = make("button", "toast-close", "×");
  close.type = "button";
  close.setAttribute("aria-label", "Dismiss notification");
  top.append(titleElement, close);

  const message = make("div", "toast-msg", "");
  const track = make("div", "toast-track");
  const bar = make("div", "toast-bar");
  track.append(bar);
  const actions = make("div", "toast-actions");

  element.append(top, message, track, actions);
  dom.toasts.prepend(element);
  close.addEventListener("click", () => element.remove());

  return { element, titleElement, message, track, bar, actions, close };
}

/** Short-lived notice used for errors and tips. */
function pushNotice(title, detail, isError = false) {
  const toast = makeToast(title, isError ? "is-error" : "");
  toast.track.hidden = true;
  setText(toast.message, detail || "");
  const timer = setTimeout(() => toast.element.remove(), isError ? 7000 : 4200);
  toast.close.addEventListener("click", () => clearTimeout(timer));
}

/** Filename for the download link. */
function exportFilename(kind) {
  return `var-offside-${kind}.${EXPORT_EXT[kind] || "bin"}`;
}

/** Stops polling for each export with a toast, so deleting a file can end them all. */
const exportPollers = new Set();

/** Remove every toast. A deleted file's exports and notices no longer apply. */
function clearExportToasts() {
  for (const stop of exportPollers) stop();
  dom.toasts.replaceChildren();
}

/** Start an export and poll the job into a toast. */
async function startExport(kind) {
  if (!state.session || state.phase !== "verdict") return;
  const sessionId = state.session.id;

  let jobId = null;
  try {
    const started = await api.exportSession(sessionId, kind, viewOptions());
    jobId = started && started.job_id;
    if (!jobId) throw new Error("No job id returned");
    if (!state.session || state.session.id !== sessionId) {
      api.cancelJob(jobId).catch(() => {});
      return;
    }
  } catch (error) {
    pushNotice("Export failed", error.message || String(error), true);
    return;
  }

  const toast = makeToast(EXPORT_LABELS[kind] || "Export");
  toast.bar.style.width = "0%";
  setText(toast.message, "Starting…");

  const cancel = make("button", "toast-link muted", "Cancel");
  cancel.type = "button";
  const download = document.createElement("a");
  download.className = "toast-link";
  download.textContent = "Download ↓";
  download.hidden = true;
  toast.actions.append(cancel, download);

  let active = true;
  let timer = null;

  const finish = () => {
    active = false;
    if (timer) clearTimeout(timer);
    timer = null;
    exportPollers.delete(finish);
  };
  exportPollers.add(finish);

  const tick = async () => {
    if (!active) return;
    try {
      const job = await api.getJob(jobId);
      if (!active) return;
      const progress = clamp(num(job.progress, 0), 0, 100);
      toast.bar.style.width = `${progress}%`;
      setText(
        toast.message,
        `${Math.round(progress)}% · ${job.message || job.status || "Working"}`,
      );

      if (job.status === "done") {
        finish();
        toast.element.classList.add("is-done");
        toast.bar.style.width = "100%";
        setText(toast.message, "Ready · 100%");
        cancel.hidden = true;
        if (job.file_url) {
          download.href = api.fileUrl(job.file_url);
          download.download = exportFilename(kind);
          download.hidden = false;
        }
        return;
      }
      if (job.status === "error") {
        finish();
        toast.element.classList.add("is-error");
        setText(toast.message, job.message || "Export failed");
        cancel.hidden = true;
        return;
      }
      if (job.status === "cancelled") {
        finish();
        toast.element.classList.add("is-cancelled");
        setText(toast.message, "Cancelled");
        cancel.hidden = true;
        return;
      }
      timer = setTimeout(tick, 700);
    } catch (error) {
      finish();
      toast.element.classList.add("is-error");
      setText(toast.message, error.message || String(error));
      cancel.hidden = true;
    }
  };

  cancel.addEventListener("click", async () => {
    if (!active) return;
    cancel.disabled = true;
    try {
      await api.cancelJob(jobId);
      finish();
      toast.element.classList.add("is-cancelled");
      setText(toast.message, "Cancelled");
    } catch (error) {
      setText(toast.message, error.message || String(error));
      cancel.disabled = false;
      return;
    }
    cancel.hidden = true;
  });

  // Stop polling when the toast is removed manually.
  toast.close.addEventListener("click", finish);
  timer = setTimeout(tick, 700);
}

/* ==========================================================================
   Popovers, menus, lightbox, keyboard
   ========================================================================== */

/** Open/close a popover and keep aria-expanded in sync. */
function togglePopover(popover, button) {
  const willOpen = popover.hidden;
  closeAllPopovers();
  if (!willOpen) return;
  popover.hidden = false;
  button.setAttribute("aria-expanded", "true");
}

/** Close every popover. */
function closeAllPopovers() {
  for (const [popover, button] of [
    [dom.exportMenu, dom.exportBtn],
    [dom.settingsPopover, dom.settingsBtn],
    [dom.helpPopover, dom.helpBtn],
  ]) {
    if (!popover.hidden) {
      popover.hidden = true;
      button.setAttribute("aria-expanded", "false");
    }
  }
}

/** Close a specific popover. */
function closePopover(popover, button) {
  if (!popover.hidden) {
    popover.hidden = true;
    button.setAttribute("aria-expanded", "false");
  }
}

/** Element focused before the stage lightbox opened. */
let stageReturnFocus = null;

/** Open one debug stage full size inside the viewport. */
function openStage(key, label) {
  if (!state.session) return;
  stageReturnFocus = document.activeElement;
  setText(dom.stageCaption, `${label} · ${key}.png`);
  dom.stageLarge.alt = `${label} debug stage`;
  dom.stageLarge.onerror = () => {
    closeLightbox();
    pushNotice("Stage not available", `${label} has not been produced yet.`, true);
  };
  dom.stageLarge.src = api.stageUrl(state.session.id, key, state.stageVersion);
  dom.stageLightbox.hidden = false;
  dom.stageClose.focus();
}

/** Close the stage lightbox. */
function closeLightbox() {
  dom.stageLightbox.hidden = true;
  dom.stageLarge.onerror = null;
  dom.stageLarge.removeAttribute("src");
  if (stageReturnFocus && typeof stageReturnFocus.focus === "function") {
    stageReturnFocus.focus();
  }
  stageReturnFocus = null;
}

/** Toggle the steps drawer body. */
function toggleSteps() {
  state.stepsOpen = !state.stepsOpen;
  dom.stepsToggle.setAttribute("aria-expanded", String(state.stepsOpen));
}

/** Toggle a settings switch. */
function toggleSwitch(button, on) {
  button.setAttribute("aria-checked", String(on));
}

/** Global keyboard map. */
function onKeyDown(event) {
  const key = event.key;
  const tag = (event.target && event.target.tagName ? event.target.tagName : "").toLowerCase();
  const typing =
    tag === "input" ||
    tag === "textarea" ||
    tag === "select" ||
    (event.target && event.target.isContentEditable);
  const inControl = Boolean(
    event.target &&
      event.target.closest &&
      event.target.closest(
        'button, a, input, textarea, select, [role="switch"], [role="menuitem"]',
      ),
  );

  if (key === "Escape") {
    closeAllPopovers();
    closeCtxMenu();
    if (!dom.stageLightbox.hidden) closeLightbox();
    return;
  }

  if (state.phase === "calibrate" && !typing && calibrator.handleKey(event)) {
    event.preventDefault();
    return;
  }

  if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "o") {
    event.preventDefault();
    dom.fileInput.click();
    return;
  }

  if (key === "F5") {
    event.preventDefault();
    primaryAction();
    return;
  }

  if (typing) return;

  if ((key === "ArrowLeft" || key === "ArrowRight") && state.phase === "review") {
    event.preventDefault();
    chooseSide(key === "ArrowLeft" ? "left" : "right");
    return;
  }

  if (key === " " && !inControl && state.phase === "verdict" && state.has3d) {
    event.preventDefault();
    toggleReplay();
    return;
  }

  // Tab toggles Photo/3D at VERDICT when focus is not on a control, so
  // keyboard navigation between actual controls still works.
  if (key === "Tab" && !event.shiftKey && !inControl && state.phase === "verdict" && state.has3d) {
    event.preventDefault();
    setView(state.view === "3d" ? "photo" : "3d");
    return;
  }

  if (PRESET_KEYS[key] && state.phase === "verdict" && state.has3d) {
    event.preventDefault();
    choosePreset(PRESET_KEYS[key]);
    return;
  }

  if ((key === "r" || key === "R") && state.phase === "verdict" && state.has3d) {
    event.preventDefault();
    choosePreset("broadcast");
  }
}

/* ==========================================================================
   Calibrate step
   ========================================================================== */

const calibrator = createCalibrator({
  canvas: byId("calibCanvas"),
  image: dom.photoImg,
  ui: {
    map: byId("calibMap"),
    handBox: byId("calibHandBox"),
    prompt: byId("calibPrompt"),
    skip: byId("calibSkip"),
    help: byId("calibHelp"),
    undo: byId("calibUndo"),
    reset: byId("calibReset"),
    mode: byId("calibMode"),
    without3d: dom.calibWithout3d,
  },
  getApi: () => api,
  getSession: () => state.session,
  getSide: () => state.attackingSide,
  onChange: () => {
    if (state.phase !== "calibrate") return;
    renderVerdict();
    renderChecks();
    renderPrimary();
  },
  onError: (title, text) => pushNotice(title, text, true),
});
// Kept for automated checks of the running page.
window.__varCalibrator = calibrator;

/* ==========================================================================
   Event wiring
   ========================================================================== */

/** Attach every listener once. */
function bindEvents() {
  dom.openBtn.addEventListener("click", () => dom.fileInput.click());
  dom.clearBtn.addEventListener("click", clearEverything);
  dom.fileInput.addEventListener("change", () => {
    const file = dom.fileInput.files && dom.fileInput.files[0];
    dom.fileInput.value = "";
    if (file) handleFile(file);
  });

  dom.primaryBtn.addEventListener("click", primaryAction);
  dom.exportBtn.addEventListener("click", () => {
    if (dom.exportBtn.disabled) return;
    togglePopover(dom.exportMenu, dom.exportBtn);
  });
  dom.exportMenu.addEventListener("click", (event) => {
    const item = event.target.closest("[data-kind]");
    if (!item || item.disabled) return;
    closeAllPopovers();
    startExport(item.dataset.kind);
  });
  dom.settingsBtn.addEventListener("click", () =>
    togglePopover(dom.settingsPopover, dom.settingsBtn),
  );
  dom.helpBtn.addEventListener("click", () =>
    togglePopover(dom.helpPopover, dom.helpBtn),
  );

  dom.stepsSwitch.addEventListener("click", () => {
    state.showSteps = !state.showSteps;
    toggleSwitch(dom.stepsSwitch, state.showSteps);
    renderSteps();
  });

  // Empty state: click / keyboard / drag & drop.
  dom.dropzone.addEventListener("click", (event) => {
    if (event.target.closest("#mockSample")) return;
    dom.fileInput.click();
  });
  dom.dropzone.addEventListener("keydown", (event) => {
    if (event.target !== dom.dropzone) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      dom.fileInput.click();
    }
  });
  dom.mockSample.addEventListener("click", (event) => {
    event.stopPropagation();
    const file = new File(["mock"], "stadium-frame.jpg", { type: "image/jpeg" });
    handleFile(file);
  });

  for (const type of ["dragenter", "dragover"]) {
    dom.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dom.dropzone.classList.add("is-dragover");
    });
  }
  dom.dropzone.addEventListener("dragleave", (event) => {
    if (!dom.dropzone.contains(event.relatedTarget)) {
      dom.dropzone.classList.remove("is-dragover");
    }
  });
  dom.dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dom.dropzone.classList.remove("is-dragover");
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // Review card sides.
  for (const button of dom.reviewCard.querySelectorAll(".side-btn")) {
    button.addEventListener("click", () => chooseSide(button.dataset.side));
  }
  for (const button of dom.reviewCard.querySelectorAll(".team-btn")) {
    button.addEventListener("click", () => chooseTeam(Number(button.dataset.team)));
  }
  dom.notInFrameBtn.addEventListener("click", toggleNotInFrame);

  // Timeline.
  dom.scrubber.addEventListener("input", () => {
    requestFrame(Number(dom.scrubber.value));
  });
  dom.prevFrame.addEventListener("click", () => requestFrame(state.frameIndex - 1, true));
  dom.nextFrame.addEventListener("click", () => requestFrame(state.frameIndex + 1, true));

  // Camera presets.
  dom.presetGroup.addEventListener("click", (event) => {
    const button = event.target.closest("[data-preset]");
    if (button) choosePreset(button.dataset.preset);
  });

  // 3D options.
  dom.toggles.addEventListener("click", (event) => {
    const button = event.target.closest("[data-opt]");
    if (!button) return;
    const option = button.dataset.opt;
    if (option === "line_plane") state.linePlane = !state.linePlane;
    else if (option === "key_only") state.keyOnly = !state.keyOnly;
    else if (option === "photo") state.photo3d = !state.photo3d;
    else if (option === "labels") state.showLabels = !state.showLabels;
    else if (option === "pulse") state.pulse = !state.pulse;
    renderOptionButtons();
    drawPhoto();
    if (option === "photo" && state.photo3d) {
      // The blend only lines up from the broadcast camera, so go there.
      choosePreset("broadcast");
    } else if (is3dActive() && viewConnected) {
      sendFullRender();
    }
  });

  dom.replayBtn.addEventListener("click", toggleReplay);

  // Photo | 3D.
  dom.viewPhotoBtn.addEventListener("click", () => setView("photo"));
  dom.view3dBtn.addEventListener("click", () => setView("3d"));

  // 3D canvas pointer controls.
  dom.view3dCanvas.addEventListener("pointerdown", on3dPointerDown);
  dom.view3dCanvas.addEventListener("pointermove", on3dPointerMove);
  dom.view3dCanvas.addEventListener("pointerup", on3dPointerUp);
  dom.view3dCanvas.addEventListener("pointercancel", on3dPointerUp);
  dom.view3dCanvas.addEventListener("wheel", on3dWheel, { passive: false });
  dom.view3dCanvas.addEventListener("dblclick", (event) => {
    if (!is3dActive()) return;
    event.preventDefault();
    choosePreset(state.preset);
  });
  dom.view3dCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
  dom.viewport.addEventListener("contextmenu", (event) => event.preventDefault());

  dom.reconnectChip.addEventListener("click", () => connect3D());

  // Overlay interactions.
  dom.overlayCanvas.addEventListener("pointerdown", onOverlayPointerDown);
  dom.overlayCanvas.addEventListener("pointermove", onOverlayPointerMove);
  dom.overlayCanvas.addEventListener("pointerup", onOverlayPointerUp);
  dom.overlayCanvas.addEventListener("pointercancel", onOverlayPointerCancel);
  dom.overlayCanvas.addEventListener("pointerleave", () => {
    if (!state.dragBox) {
      state.clickCandidate = null;
      clearHover();
    }
  });

  // Calibrate step.
  dom.calibBack.addEventListener("click", leaveCalibrate);
  dom.calibWithout3d.addEventListener("click", continueWithout3d);

  // Stage lightbox.
  dom.stageClose.addEventListener("click", closeLightbox);
  dom.stageLightbox.addEventListener("click", (event) => {
    if (event.target === dom.stageLightbox) closeLightbox();
  });

  // Steps drawer.
  dom.stepsToggle.addEventListener("click", toggleSteps);

  // Photo load redraw.
  dom.photoImg.addEventListener("load", () => drawPhoto());
  dom.photoImg.addEventListener("error", () => {
    pushNotice("Frame unavailable", "The backend could not return this frame.", true);
  });

  document.addEventListener("keydown", onKeyDown);
  installOutsideHandlers();

  // Redraw/refresh on resize (debounced for the 3D render).
  const observer = new ResizeObserver(() => {
    drawPhoto();
    calibrator.redraw();
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      resizeTimer = null;
      if (is3dActive()) sendFullRender();
    }, 150);
  });
  observer.observe(dom.viewport);
}

/* ==========================================================================
   Boot
   ========================================================================== */

/** Prefer the real API; fall back to the mock when health is unreachable. */
async function setupApi() {
  const params = new URLSearchParams(location.search);
  if (params.get("mock") === "1") {
    state.mock = true;
    api = createMockApi();
    return;
  }

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1800);
    const response = await fetch(`${API}/health`, {
      signal: controller.signal,
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    clearTimeout(timer);
    if (!response.ok) throw new Error("health probe failed");
    const health = await response.json();
    if (!health || health.status !== "ok") throw new Error("unhealthy");
    state.mock = false;
    api = realApi;
  } catch {
    state.mock = true;
    api = createMockApi();
  }
}

/** Start the UI. */
async function init() {
  api = realApi;
  applyTeamColors();
  dom.viewport.dataset.phase = "empty";
  bindEvents();
  setPhase("empty");
  await setupApi();
  setupAutoReview();
  renderMockChip();
  renderAll();
}

init();
