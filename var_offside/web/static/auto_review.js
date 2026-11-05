/**
 * Auto-review in the browser: runs a review and plays each correction on the photo.
 *
 * Every effect comes from a real event in the run, and the final burst plays only
 * on a cleared result. Effects draw on their own canvas above the player overlay,
 * and the normal overlay redraws from the saved scene when the run ends.
 */

const POLL_MS = 350;

/** How long each event holds the stage, in milliseconds. */
const DURATIONS = {
  check_started: 420,
  player_added: 1250,
  box_removed: 950,
  team_changed: 950,
  role_changed: 900,
  side_set: 1200,
  attacking_team_set: 900,
  camera_refined: 1250,
  line_moved: 1000,
  call_changed: 1000,
  question_asked: 300,
  run_ended: 1300,
};

const BOX_DRAW_MS = 800;

const PITCH_LINE_COLOR = "#ffd84a";
const GLOW = "#3ddc84";
const SPARK = ["#fff6c9", "#3ddc84", "#ffffff"];

const CHECK_NAMES = {
  frame: "Checking the frame",
  side: "Finding the attack direction",
  team: "Finding the attacking team",
  keeper: "Finding the keeper",
  camera: "Checking the camera",
  call: "Working out the call",
  players: "Looking for missed players",
  teams: "Checking the teams",
  officials: "Looking for officials",
  stability: "Testing the call",
};

const CALL_NAMES = { offside: "Offside", onside: "Onside", unknown: "No call" };

function reducedMotion() {
  return Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

const easeOut = (t) => 1 - (1 - t) ** 3;
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/** Interpolate between two polylines with the same number of points. */
function mixPoints(from, to, t) {
  return to.map((point, index) => {
    const start = (from && from[index]) || point;
    return [start[0] + (point[0] - start[0]) * t, start[1] + (point[1] - start[1]) * t];
  });
}

/* ==========================================================================
   Effects canvas
   ========================================================================== */

class Effects {
  constructor(canvas, viewport, getImageRect, lineColor) {
    this.lineColor = lineColor;
    this.canvas = canvas;
    this.viewport = viewport;
    this.getImageRect = getImageRect;
    this.items = [];
    this.scanning = false;
    this.scanStart = 0;
    this.line = null;
    this.spotlight = null;
    this.raf = 0;
    this.frame = this.frame.bind(this);
  }

  /** Image pixels to viewport CSS pixels. */
  toScreen(x, y) {
    const rect = this.getImageRect(this.viewport.clientWidth, this.viewport.clientHeight);
    return [rect.x + x * rect.scale, rect.y + y * rect.scale];
  }

  add(duration, draw, delay = 0) {
    this.items.push({ start: performance.now() + delay, duration, draw });
    this.wake();
  }

  wake() {
    if (!this.raf) this.raf = requestAnimationFrame(this.frame);
  }

  clear() {
    this.items = [];
    this.scanning = false;
    this.line = null;
    this.spotlight = null;
    this.paintEmpty();
  }

  paintEmpty() {
    const ctx = this.canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  frame(now) {
    this.raf = 0;
    const ctx = this.canvas.getContext("2d");
    const vw = this.viewport.clientWidth;
    const vh = this.viewport.clientHeight;
    if (!ctx || !vw || !vh) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(vw * dpr);
    const height = Math.round(vh * dpr);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, vw, vh);

    if (this.scanning) this.drawBeam(ctx, now, vw, vh);
    if (this.spotlight) this.drawSpotlight(ctx, now, vw, vh);
    if (this.line) this.drawLine(ctx, now);

    this.items = this.items.filter((item) => now < item.start + item.duration);
    for (const item of this.items) {
      if (now < item.start) continue;
      ctx.save();
      item.draw(ctx, Math.min(1, (now - item.start) / item.duration));
      ctx.restore();
    }

    if (this.items.length || this.scanning || this.spotlight || (this.line && this.line.moving)) {
      this.wake();
    }
  }

  /* --- Persistent layers ----------------------------------------------- */

  drawBeam(ctx, now, vw, vh) {
    const rect = this.getImageRect(vw, vh);
    const period = 2200;
    const t = ((now - this.scanStart) % period) / period;
    const x = rect.x - 120 + easeInOut(t) * (rect.w + 240);
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
    ctx.clip();
    const gradient = ctx.createLinearGradient(x - 130, 0, x, 0);
    gradient.addColorStop(0, "rgba(61,220,132,0)");
    gradient.addColorStop(0.82, "rgba(61,220,132,0.2)");
    gradient.addColorStop(1, "rgba(234,255,243,0.8)");
    ctx.fillStyle = gradient;
    ctx.fillRect(x - 130, rect.y, 132, rect.h);
    ctx.restore();
  }

  drawSpotlight(ctx, now, vw, vh) {
    const [x1, y1] = this.toScreen(this.spotlight[0], this.spotlight[1]);
    const [x2, y2] = this.toScreen(this.spotlight[2], this.spotlight[3]);
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    const radius = Math.max(x2 - x1, y2 - y1) * (1.05 + 0.05 * Math.sin(now / 340));
    const gradient = ctx.createRadialGradient(cx, cy, radius * 0.75, cx, cy, radius * 2.4);
    gradient.addColorStop(0, "rgba(0,0,0,0)");
    gradient.addColorStop(1, "rgba(0,0,0,0.62)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, vw, vh);
  }

  drawLine(ctx, now) {
    const line = this.line;
    const t = line.moving ? Math.min(1, (now - line.start) / line.duration) : 1;
    if (t >= 1) line.moving = false;
    const eased = easeInOut(t);
    const points = mixPoints(line.from, line.to, eased).map(([x, y]) => this.toScreen(x, y));
    ctx.save();
    ctx.globalAlpha = line.from ? 1 : eased;
    ctx.lineCap = "round";
    for (const [width, alpha] of [[10, 0.18], [2.2, 1]]) {
      ctx.globalAlpha = (line.from ? 1 : eased) * alpha;
      ctx.lineWidth = width;
      ctx.strokeStyle = this.lineColor;
      ctx.shadowColor = this.lineColor;
      ctx.shadowBlur = width > 5 ? 0 : 10;
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (const point of points.slice(1)) ctx.lineTo(point[0], point[1]);
      ctx.stroke();
    }
    ctx.restore();
  }

  glideLine(from, to, duration) {
    this.line = { from, to, start: performance.now(), duration, moving: true };
    this.wake();
  }

  /* --- One-shot effects ------------------------------------------------ */

  sparkles(imageX, imageY, count = 14, delay = 0, spread = 46) {
    const [x, y] = this.toScreen(imageX, imageY);
    for (let index = 0; index < count; index += 1) {
      const angle = Math.random() * Math.PI * 2;
      const distance = spread * (0.4 + Math.random() * 0.8);
      const size = 2.5 + Math.random() * 4;
      const color = SPARK[index % SPARK.length];
      const stagger = delay + Math.random() * 180;
      this.add(760, (ctx, t) => {
        const eased = easeOut(t);
        const px = x + Math.cos(angle) * distance * eased;
        const py = y + Math.sin(angle) * distance * eased - 10 * t;
        const r = size * Math.sin(Math.PI * Math.min(1, t * 1.1));
        ctx.translate(px, py);
        ctx.rotate(t * 2);
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        for (let point = 0; point < 8; point += 1) {
          const reach = point % 2 ? r * 0.28 : r;
          const a = (point / 8) * Math.PI * 2;
          ctx.lineTo(Math.cos(a) * reach, Math.sin(a) * reach);
        }
        ctx.closePath();
        ctx.fill();
      }, stagger);
    }
  }

  drawBox(bbox, color, label) {
    const [x1, y1] = this.toScreen(bbox[0], bbox[1]);
    const [x2, y2] = this.toScreen(bbox[2], bbox[3]);
    const width = x2 - x1;
    const height = y2 - y1;
    const perimeter = 2 * (width + height);
    // Lasts until the scene applies the player, when the overlay takes over the box and number.
    this.add(BOX_DRAW_MS, (ctx, t) => {
      const drawn = easeOut(Math.min(1, t / 0.85));
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = 14;
      ctx.setLineDash([perimeter * drawn, perimeter]);
      ctx.strokeRect(x1, y1, width, height);
      ctx.setLineDash([]);
      if (label && t > 0.4) {
        const pop = easeOut(Math.min(1, (t - 0.4) / 0.25));
        ctx.font = `600 ${Math.round(14 * pop) || 1}px "IBM Plex Mono", monospace`;
        ctx.textAlign = "center";
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, (x1 + x2) / 2, y1 - 8);
      }
    });
  }

  dissolve(bbox, color) {
    const [x1, y1] = this.toScreen(bbox[0], bbox[1]);
    const [x2, y2] = this.toScreen(bbox[2], bbox[3]);
    this.add(420, (ctx, t) => {
      ctx.globalAlpha = 1 - t;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
    for (let index = 0; index < 34; index += 1) {
      const onSide = index % 2 === 0;
      const px = onSide ? (index % 4 === 0 ? x1 : x2) : x1 + Math.random() * (x2 - x1);
      const py = onSide ? y1 + Math.random() * (y2 - y1) : index % 4 === 1 ? y1 : y2;
      const dx = (Math.random() - 0.5) * 40;
      const dy = 12 + Math.random() * 46;
      this.add(800, (ctx, t) => {
        const eased = easeOut(t);
        ctx.globalAlpha = 1 - t;
        ctx.fillStyle = color;
        ctx.fillRect(px + dx * eased, py + dy * eased * t, 2.4, 2.4);
      }, Math.random() * 130);
    }
  }

  ripple(imageX, imageY, color) {
    const [x, y] = this.toScreen(imageX, imageY);
    for (const delay of [0, 170]) {
      this.add(760, (ctx, t) => {
        ctx.globalAlpha = 1 - t;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(x, y, 8 + easeOut(t) * 58, 0, Math.PI * 2);
        ctx.stroke();
      }, delay);
    }
  }

  glowBox(bbox, color, delay = 0) {
    const [x1, y1] = this.toScreen(bbox[0], bbox[1]);
    const [x2, y2] = this.toScreen(bbox[2], bbox[3]);
    this.add(760, (ctx, t) => {
      ctx.globalAlpha = Math.sin(Math.PI * t);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.shadowColor = color;
      ctx.shadowBlur = 18;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }, delay);
  }

  badge(bbox, text, color) {
    const [x1, , x2, y2] = bbox;
    const [cx, cy] = this.toScreen((x1 + x2) / 2, y2);
    this.add(1000, (ctx, t) => {
      const pop = t < 0.25 ? easeOut(t / 0.25) * 1.18 : 1.18 - 0.18 * Math.min(1, (t - 0.25) / 0.2);
      ctx.globalAlpha = t > 0.8 ? (1 - t) / 0.2 : 1;
      ctx.translate(cx, cy + 18);
      ctx.scale(pop, pop);
      ctx.font = '600 11px "IBM Plex Mono", monospace';
      const width = ctx.measureText(text).width + 14;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(-width / 2, -10, width, 20, 4);
      ctx.fill();
      ctx.fillStyle = "#04120a";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 0, 0);
    });
  }

  arrow(side, viewWidth, viewHeight) {
    const rect = this.getImageRect(viewWidth, viewHeight);
    const direction = side === "left" ? -1 : 1;
    const y = rect.y + rect.h * 0.16;
    this.add(1100, (ctx, t) => {
      const cx = rect.x + rect.w / 2 + direction * (easeInOut(t) - 0.5) * rect.w * 0.55;
      ctx.globalAlpha = Math.sin(Math.PI * t);
      ctx.strokeStyle = GLOW;
      ctx.lineWidth = 5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowColor = GLOW;
      ctx.shadowBlur = 16;
      for (const offset of [-34, -8, 18]) {
        const x = cx + direction * offset;
        ctx.beginPath();
        ctx.moveTo(x - direction * 12, y - 20);
        ctx.lineTo(x + direction * 8, y);
        ctx.lineTo(x - direction * 12, y + 20);
        ctx.stroke();
      }
    });
  }

  pitchLines(from, to) {
    this.add(1200, (ctx, t) => {
      const eased = easeInOut(Math.min(1, t / 0.72));
      ctx.globalAlpha = t > 0.8 ? (1 - t) / 0.2 : Math.min(1, t * 6);
      ctx.strokeStyle = PITCH_LINE_COLOR;
      ctx.lineWidth = 2;
      ctx.shadowColor = PITCH_LINE_COLOR;
      ctx.shadowBlur = 8;
      to.forEach((line, index) => {
        const points = mixPoints(from[index], line, eased).map(([x, y]) => this.toScreen(x, y));
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        for (const point of points.slice(1)) ctx.lineTo(point[0], point[1]);
        ctx.stroke();
      });
    });
  }
}

/* ==========================================================================
   Scene changes
   ========================================================================== */

const centerOf = (bbox) => [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];

/** The event that undoes a correction, or null when it has no opposite. */
function reverseOf(event) {
  switch (event.type) {
    case "player_added":
      return { ...event, type: "box_removed" };
    case "box_removed":
      return { ...event, type: "player_added" };
    case "team_changed":
      return { ...event, from_team: event.to_team, to_team: event.from_team };
    case "role_changed":
      return { ...event, role: event.from_role || "player", from_role: event.role };
    case "side_set":
      return { ...event, side: null };
    case "attacking_team_set":
      return { ...event, team: null };
    default:
      return null;
  }
}

/** One line of text for the corrections list. */
function correctionText(event) {
  if (event.reason) return event.reason;
  return event.type.replace(/_/g, " ");
}

/* ==========================================================================
   Controller
   ========================================================================== */

/**
 * @param deps.api             the app's API object
 * @param deps.canvas          the effects canvas
 * @param deps.viewport        the element the canvas covers
 * @param deps.getImageRect    (width, height) => image rectangle in the viewport
 * @param deps.lineColor       colour of the offside line, matching the overlay
 * @param deps.applyEvent      (event) => change the app's scene for this event
 * @param deps.findPlayer      (id) => the player, or undefined
 * @param deps.playersOfTeam   (team) => players on that team
 * @param deps.teamColor       (team) => colour
 * @param deps.onResult        (result, info) => show the final scene and verdict
 * @param deps.onReviewByHand  (result) => go to manual Review
 * @param deps.onFailed        (message) => report a failed run
 */
export function createAutoReview(deps) {
  const byId = (id) => document.getElementById(id);
  const ui = {
    hud: byId("arHud"),
    status: byId("arStatus"),
    statusText: byId("arStatusText"),
    skip: byId("arSkip"),
    call: byId("arCall"),
    question: byId("arQuestion"),
    questionText: byId("arQuestionText"),
    questionChoices: byId("arQuestionChoices"),
    notice: byId("arNotice"),
    noticeText: byId("arNoticeText"),
    noticeHand: byId("arNoticeHand"),
    panel: byId("arPanel"),
    summary: byId("arSummary"),
    list: byId("arList"),
  };
  const fx = new Effects(deps.canvas, deps.viewport, deps.getImageRect, deps.lineColor);

  let run = 0; // Bumped to abandon a run or queue that belongs to an old frame.
  let active = false;
  let instant = false;
  let ownsLine = false;
  let queue = [];
  let pumping = null;
  let wakeDelay = null;
  let sessionId = null;
  let corrections = [];
  let lastResult = null;

  function resetUi() {
    ui.hud.hidden = true;
    ui.status.hidden = true;
    ui.call.hidden = true;
    ui.call.className = "ar-call";
    ui.question.hidden = true;
    ui.notice.hidden = true;
    ui.panel.hidden = true;
    ui.summary.textContent = "";
    ui.list.textContent = "";
  }

  function setStatusText(text) {
    ui.hud.hidden = false;
    ui.status.hidden = false;
    ui.statusText.textContent = text;
  }

  function delay(ms) {
    if (instant) return Promise.resolve();
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        wakeDelay = null;
        resolve();
      }, ms);
      wakeDelay = () => {
        clearTimeout(timer);
        wakeDelay = null;
        resolve();
      };
    });
  }

  function showCall(event) {
    const to = CALL_NAMES[event.to] || "No call";
    const from = CALL_NAMES[event.from];
    const margin = event.margin_m == null ? "" : ` · ${Math.abs(event.margin_m).toFixed(1)} m`;
    ui.hud.hidden = false;
    ui.call.hidden = false;
    ui.call.textContent = event.from && event.from !== "unknown" ? `${from} → ${to}${margin}` : `${to}${margin}`;
    ui.call.className = `ar-call is-${event.to}`;
    if (!instant) {
      void ui.call.offsetWidth;
      ui.call.classList.add("is-flip");
    }
  }

  function addCorrection(event) {
    ui.panel.hidden = false;
    const entry = { event, undone: false };
    corrections.push(entry);
    const item = document.createElement("li");
    item.className = "ar-item";
    item.dataset.undoId = event.undo_id || "";
    const text = document.createElement("div");
    text.className = "ar-item-text";
    text.textContent = correctionText(event);
    item.appendChild(text);
    if (event.flips_call) {
      const chip = document.createElement("span");
      chip.className = "ar-chip";
      chip.textContent = "Changed the call";
      item.appendChild(chip);
    }
    const actions = document.createElement("div");
    actions.className = "ar-item-actions";
    if (event.bbox || playerBox(event)) {
      const showBtn = document.createElement("button");
      showBtn.type = "button";
      showBtn.className = "btn ghost ar-mini";
      showBtn.textContent = "Show";
      showBtn.addEventListener("click", () => showCorrection(event));
      actions.appendChild(showBtn);
    }
    if (event.undo_id) {
      const undoBtn = document.createElement("button");
      undoBtn.type = "button";
      undoBtn.className = "btn ghost ar-mini ar-undo";
      undoBtn.textContent = "Undo";
      undoBtn.addEventListener("click", () => undo(entry, item));
      actions.appendChild(undoBtn);
    }
    item.appendChild(actions);
    ui.list.appendChild(item);
    entry.item = item;
  }

  function playerBox(event) {
    if (event.bbox) return event.bbox;
    const player = event.player_id == null ? null : deps.findPlayer(event.player_id);
    return player ? player.bbox : null;
  }

  function showCorrection(event) {
    if (deps.onShow) deps.onShow();
    const bbox = playerBox(event);
    if (!bbox) return;
    if (reducedMotion()) return;
    const color = event.to_team ? deps.teamColor(event.to_team) : GLOW;
    fx.glowBox(bbox, color);
    fx.glowBox(bbox, color, 420);
  }

  /** Change the scene for one event and play its effect. */
  async function play(event, token, { reversed = false } = {}) {
    const animate = !instant;
    const duration = DURATIONS[event.type] || 500;
    const viewW = deps.viewport.clientWidth;
    const viewH = deps.viewport.clientHeight;

    switch (event.type) {
      case "check_started":
        setStatusText(`${CHECK_NAMES[event.check] || "Checking"}…`);
        break;
      case "player_added": {
        const color = deps.teamColor(event.team);
        if (animate) {
          const [cx, cy] = centerOf(event.bbox);
          fx.sparkles(cx, cy, 20);
          fx.drawBox(event.bbox, color, `#${event.player_id}`);
          await delay(BOX_DRAW_MS);
          if (token !== run) return;
        }
        deps.applyEvent(event);
        break;
      }
      case "box_removed": {
        const color = deps.teamColor(event.team);
        deps.applyEvent(event);
        if (animate) fx.dissolve(event.bbox, color);
        break;
      }
      case "team_changed": {
        deps.applyEvent(event);
        const bbox = playerBox(event);
        if (animate && bbox) {
          const color = deps.teamColor(event.to_team);
          const [cx, cy] = centerOf(bbox);
          fx.ripple(cx, cy, color);
          fx.glowBox(bbox, color, 120);
        }
        break;
      }
      case "role_changed": {
        deps.applyEvent(event);
        const bbox = playerBox(event);
        if (animate && bbox) {
          const [cx, cy] = centerOf(bbox);
          if (event.role === "goalkeeper") {
            fx.sparkles(cx, cy, 10, 0, 34);
            fx.badge(bbox, "KEEPER", GLOW);
          } else if (event.role === "referee") {
            fx.badge(bbox, "REFEREE", "#b8c6bd");
          } else {
            fx.badge(bbox, "PLAYER", "#b8c6bd");
          }
        }
        break;
      }
      case "side_set":
        deps.applyEvent(event);
        if (animate && event.side) fx.arrow(event.side, viewW, viewH);
        break;
      case "attacking_team_set":
        deps.applyEvent(event);
        if (animate && event.team) {
          deps.playersOfTeam(event.team).forEach((player, index) => {
            fx.glowBox(player.bbox, deps.teamColor(event.team), index * 30);
          });
        }
        break;
      case "camera_refined":
        if (animate && event.from_lines && event.to_lines) {
          fx.pitchLines(reversed ? event.to_lines : event.from_lines, reversed ? event.from_lines : event.to_lines);
        }
        break;
      case "line_moved":
        if (animate) {
          fx.glideLine(event.from_points, event.to_points, 850);
        } else {
          fx.line = { from: null, to: event.to_points, start: 0, duration: 1, moving: false };
          fx.wake();
        }
        break;
      case "call_changed":
        showCall(event);
        break;
      case "question_asked":
        showQuestion(event);
        break;
      case "run_ended":
        fx.scanning = false;
        if (event.ending === "cleared" && animate && fx.line) {
          const [a, b] = fx.line.to;
          for (let step = 0; step <= 4; step += 1) {
            const t = step / 4;
            fx.sparkles(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 18, step * 70, 70);
          }
        }
        break;
      default:
        break;
    }

    if (!reversed && event.undo_id) addCorrection(event);
    if (animate) await delay(duration);
  }

  async function pump(token) {
    while (token === run && queue.length) {
      await play(queue.shift(), token);
    }
  }

  /** Queue events to play in order. Resolves once everything queued has played. */
  function enqueue(events, token) {
    queue.push(...events);
    if (!pumping) {
      const current = pump(token).finally(() => {
        if (pumping === current) pumping = null;
      });
      pumping = current;
    }
    return pumping;
  }

  function showQuestion(event) {
    fx.scanning = false;
    const player = (event.player_ids || []).map(deps.findPlayer).find(Boolean);
    fx.spotlight = player ? player.bbox : null;
    fx.wake();
    ui.status.hidden = true;
    ui.questionText.textContent = event.text;
    ui.questionChoices.textContent = "";
    for (const choice of event.choices || []) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn ghost ar-choice";
      button.textContent = choice;
      button.addEventListener("click", () => answer(event.question_id, choice));
      ui.questionChoices.appendChild(button);
    }
    const hand = document.createElement("button");
    hand.type = "button";
    hand.className = "btn ghost ar-hand";
    hand.textContent = "Review by hand";
    hand.addEventListener("click", () => {
      run += 1;
      stopRun();
      deps.onReviewByHand(null);
    });
    ui.questionChoices.appendChild(hand);
    ui.question.hidden = false;
  }

  async function answer(questionId, choice) {
    const token = run;
    ui.question.hidden = true;
    fx.spotlight = null;
    if (!instant && !reducedMotion()) {
      fx.scanning = true;
      fx.scanStart = performance.now();
      fx.wake();
    }
    setStatusText("Continuing…");
    try {
      await deps.api.answerAutoReview(sessionId, questionId, choice);
      if (token === run) poll(token);
    } catch (error) {
      if (token === run) fail(error);
    }
  }

  function stopRun() {
    active = false;
    instant = false;
    ui.list.classList.remove("is-running");
    fx.clear();
    ui.hud.hidden = true;
    ui.status.hidden = true;
    ui.call.hidden = true;
    ui.question.hidden = true;
  }

  function fail(error) {
    run += 1;
    stopRun();
    deps.onFailed(error && error.message ? error.message : String(error));
  }

  let since = 0;

  async function poll(token) {
    try {
      for (;;) {
        const status = await deps.api.getAutoReview(sessionId, since);
        if (token !== run) return;
        const events = status.events || [];
        if (events.length) since = events[events.length - 1].seq;
        await enqueue(events, token);
        if (token !== run) return;
        if (status.status === "needs_answer") return;
        if (status.status === "error") throw new Error(status.message || "Auto-review failed");
        if (status.status === "done") {
          finish(status.result);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
        if (token !== run) return;
      }
    } catch (error) {
      if (token === run) fail(error);
    }
  }

  function finish(result) {
    lastResult = result;
    const panelHasItems = corrections.length > 0;
    stopRun();
    if (result.ending === "not_usable") {
      ui.hud.hidden = true;
      ui.noticeText.textContent = result.reason || "This frame can't be judged.";
      ui.notice.hidden = false;
      deps.onResult(result, { ending: result.ending });
      return;
    }
    if (result.ending === "needs_answer") {
      ui.panel.hidden = !panelHasItems;
      deps.onReviewByHand(result);
      return;
    }
    ui.panel.hidden = !panelHasItems;
    ui.summary.textContent =
      result.ending === "cleared" && result.stability
        ? `${result.stability.alternatives_tested} doubts tested · ${result.stability.changed_the_call} changed the call`
        : result.reason || "";
    deps.onResult(result, { ending: result.ending });
  }

  /** Play a correction backwards, then show the rerun call. */
  async function undo(entry, item) {
    if (entry.undone || active) return;
    const token = run;
    const buttons = item.querySelectorAll("button");
    buttons.forEach((button) => {
      button.disabled = true;
    });
    if (deps.onShow) deps.onShow();
    instant = reducedMotion();
    try {
      const response = await deps.api.undoAutoReview(sessionId, entry.event.undo_id);
      if (token !== run) return;
      active = true;
      const old = lastResult && lastResult.verdict && lastResult.verdict.line_image_points;
      if (old && old.length >= 2) {
        ownsLine = true;
        fx.line = { from: null, to: old, start: 0, duration: 1, moving: false };
        fx.wake();
      }
      const reversed = reverseOf(entry.event);
      if (reversed) await play(reversed, token, { reversed: true });
      else if (entry.event.type === "camera_refined") await play(entry.event, token, { reversed: true });
      for (const event of response.events || []) {
        if (token !== run) return;
        await play(event, token, { reversed: true });
      }
      if (token !== run) return;
      entry.undone = true;
      item.classList.add("is-undone");
      item.querySelector(".ar-item-actions").remove();
      lastResult = response.result;
      active = false;
      ownsLine = false;
      fx.clear();
      ui.hud.hidden = true;
      ui.call.hidden = true;
      deps.onResult(response.result, { ending: response.result.ending, undo: true });
    } catch (error) {
      active = false;
      ownsLine = false;
      fx.clear();
      buttons.forEach((button) => {
        button.disabled = false;
      });
      deps.onUndoFailed(error && error.message ? error.message : String(error));
    } finally {
      instant = false;
    }
  }

  /** Jump to the final result, without changing what it is. */
  function skip() {
    if (!active || !ui.question.hidden) return;
    instant = true;
    fx.items = [];
    fx.scanning = false;
    if (fx.line) fx.line.moving = false;
    fx.wake();
    ui.status.hidden = true;
    if (wakeDelay) wakeDelay();
  }

  ui.skip.addEventListener("click", skip);
  ui.noticeHand.addEventListener("click", () => {
    ui.notice.hidden = true;
    deps.onReviewByHand(lastResult);
  });
  deps.viewport.addEventListener(
    "pointerdown",
    (event) => {
      if (active && !event.target.closest("button")) skip();
    },
    true,
  );
  window.addEventListener("resize", () => fx.wake());

  resetUi();

  return {
    /** True while a correction owns the line, so the overlay doesn't draw a second one. */
    ownsLine: () => ownsLine,

    /** Forget any run, for example when a new file opens. */
    reset() {
      run += 1;
      queue = [];
      pumping = null;
      corrections = [];
      lastResult = null;
      since = 0;
      active = false;
      instant = false;
      ownsLine = false;
      fx.clear();
      resetUi();
    },

    /** Start following a run that the server has already begun on this frame. */
    start(id) {
      this.reset();
      sessionId = id;
      const token = run;
      active = true;
      instant = reducedMotion();
      if (!instant) {
        fx.scanning = true;
        fx.scanStart = performance.now();
        fx.wake();
      }
      ui.list.classList.add("is-running");
      setStatusText("Starting Auto-review…");
      ui.skip.hidden = false;
      poll(token);
    },
  };
}
