/**
 * Calibrate step: the pitch drawn over the photo, which the user can correct.
 *
 * Drag a yellow point onto its marking, drag a pitch line onto the real line, or
 * place points from a pitch map when there's no automatic result. The server solves
 * the camera; this module draws, handles input and keeps one solve in flight so a
 * drag stays smooth. The moved point follows the pointer immediately either way.
 */

const WIRE = "#ffd84d";
const WIRE_DIM = "rgba(255, 216, 77, 0.92)";
const PINNED = "#3ddc84";
const POLL_MS = 350;
const POINT_HIT_PX = 13;
const LINE_HIT_PX = 7;
const DRAG_START_PX = 3;
const LOUPE_PX = 176;
const LOUPE_ZOOM = 4;
const MAX_ZOOM = 12;
const MAP_W = 576;
const MAP_H = 376;
const MAP_PAD = 16;

/** Project field metres through a homography; null when behind the camera. */
function project(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  if (w <= 1e-6) return null;
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

/** Distance from a point to a segment, plus the nearest point's position along it. */
function segmentDistance(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const len = dx * dx + dy * dy;
  const t = len > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len)) : 0;
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/** The part of a field segment in front of the camera, in image pixels. */
function clipSegment(H, a, b) {
  const wa = H[2][0] * a[0] + H[2][1] * a[1] + H[2][2];
  const wb = H[2][0] * b[0] + H[2][1] * b[1] + H[2][2];
  const near = 1e-3;
  if (wa <= near && wb <= near) return null;
  let p = a;
  let q = b;
  if (wa <= near || wb <= near) {
    const t = (near - wa) / (wb - wa);
    const mid = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    if (wa <= near) p = mid;
    else q = mid;
  }
  const pp = project(H, p[0], p[1]);
  const qq = project(H, q[0], q[1]);
  return pp && qq ? [pp, qq] : null;
}

/**
 * Create the Calibrate controller.
 * @param {object} options
 * @param {HTMLCanvasElement} options.canvas Main drawing surface.
 * @param {HTMLImageElement} options.image The source frame.
 * @param {object} options.ui Panel elements.
 * @param {() => object} options.getApi The current API object.
 * @param {() => object|null} options.getSession The current session.
 * @param {() => string|null} options.getSide The confirmed attack direction.
 * @param {() => void} options.onChange Called when status or buttons may have changed.
 * @param {(title: string, text: string) => void} options.onError Reports a failure.
 */
export function createCalibrator(options) {
  const { canvas, image, ui, getApi, getSession, getSide, onChange, onError } = options;
  const ctx = canvas.getContext("2d");
  const mapCtx = ui.map ? ui.map.getContext("2d") : null;

  const s = {
    open: false,
    status: "idle",
    mode: "edit",
    autoAvailable: false,
    source: "auto",
    H: null,
    camera: null,
    fit: null,
    pins: [],
    savedPins: "[]",
    savedMode: "edit",
    pitch: null,
    active: null,
    selected: null,
    hover: null,
    drag: null,
    pan: null,
    history: [],
    zoom: 1,
    panX: 0,
    panY: 0,
    pointer: null,
  };

  let raf = 0;
  let pollTimer = null;
  let solving = false;
  let solveAgain = false;
  let openToken = 0;
  let pinCounter = 0;

  /* -------------------------------- view -------------------------------- */

  /** Fit the frame into the canvas, then apply the user's zoom and pan. */
  function viewRect() {
    const session = getSession();
    const iw = Number(session && session.width) || 16;
    const ih = Number(session && session.height) || 9;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    const fit = Math.min(cw / iw, ch / ih);
    const scale = fit * s.zoom;
    const w = iw * scale;
    const h = ih * scale;
    return {
      scale,
      iw,
      ih,
      x: (cw - iw * fit) / 2 + s.panX,
      y: (ch - ih * fit) / 2 + s.panY,
      w,
      h,
      cw,
      ch,
    };
  }

  /** Keep the zoomed frame covering the canvas. */
  function clampPan() {
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    const session = getSession();
    const iw = Number(session && session.width) || 16;
    const ih = Number(session && session.height) || 9;
    const fit = Math.min(cw / iw, ch / ih);
    const baseW = iw * fit;
    const baseH = ih * fit;
    const w = baseW * s.zoom;
    const h = baseH * s.zoom;
    const baseX = (cw - baseW) / 2;
    const baseY = (ch - baseH) / 2;
    const clampAxis = (pan, size, view, base) => {
      if (size <= view) return (view - size) / 2 - base;
      return Math.min(-base, Math.max(view - size - base, pan));
    };
    s.panX = clampAxis(s.panX, w, cw, baseX);
    s.panY = clampAxis(s.panY, h, ch, baseY);
  }

  const toScreen = (rect, x, y) => [rect.x + x * rect.scale, rect.y + y * rect.scale];
  const toImage = (rect, sx, sy) => [(sx - rect.x) / rect.scale, (sy - rect.y) / rect.scale];

  function eventPoint(event) {
    const bounds = canvas.getBoundingClientRect();
    return [event.clientX - bounds.left, event.clientY - bounds.top];
  }

  /* -------------------------------- data -------------------------------- */

  const landmark = (id) => s.pitch && s.pitch.landmarks.find((row) => row.id === id);
  const pitchLine = (id) => s.pitch && s.pitch.lines.find((row) => row.id === id);
  const pinKey = (pin) => pin.key;
  const pointPin = (id) => s.pins.find((pin) => pin.kind === "point" && pin.ref === id);

  function pinName(pin) {
    const row = pin.kind === "point" ? landmark(pin.ref) : pitchLine(pin.ref);
    return row ? row.name : pin.ref;
  }

  function wirePins() {
    return s.pins.map((pin) => ({ kind: pin.kind, ref: pin.ref, x: pin.x, y: pin.y }));
  }

  function loadPins(rows) {
    s.pins = (rows || []).map((pin) => ({
      ...pin,
      key: pin.kind === "point" ? `p:${pin.ref}` : `l:${pin.ref}:${(pinCounter += 1)}`,
    }));
  }

  /** Points in view, for dragging: pins first, then the pitch's other landmarks. */
  function handles(rect) {
    const out = [];
    for (const pin of s.pins) {
      const [sx, sy] = toScreen(rect, pin.x, pin.y);
      out.push({ pin, key: pinKey(pin), sx, sy, name: pinName(pin), ix: pin.x, iy: pin.y });
    }
    if (!s.H || !s.pitch) return out;
    for (const row of s.pitch.landmarks) {
      if (pointPin(row.id)) continue;
      const p = project(s.H, row.x, row.y);
      if (!p || p[0] < 0 || p[1] < 0 || p[0] >= rect.iw || p[1] >= rect.ih) continue;
      const [sx, sy] = toScreen(rect, p[0], p[1]);
      out.push({ landmark: row, key: `g:${row.id}`, sx, sy, name: row.name, ix: p[0], iy: p[1] });
    }
    return out;
  }

  function hitHandle(rect, sx, sy) {
    let best = null;
    let bestDist = POINT_HIT_PX;
    for (const handle of handles(rect)) {
      const dist = Math.hypot(handle.sx - sx, handle.sy - sy);
      // Pins win ties, so a green point is never lost under a yellow one.
      if (dist < bestDist || (dist === bestDist && handle.pin)) {
        best = handle;
        bestDist = dist;
      }
    }
    return best;
  }

  function hitLine(rect, sx, sy) {
    if (!s.H || !s.pitch) return null;
    let best = null;
    let bestDist = LINE_HIT_PX;
    for (const line of s.pitch.lines) {
      const clipped = clipSegment(s.H, line.a, line.b);
      if (!clipped) continue;
      const a = toScreen(rect, clipped[0][0], clipped[0][1]);
      const b = toScreen(rect, clipped[1][0], clipped[1][1]);
      const dist = segmentDistance(sx, sy, a[0], a[1], b[0], b[1]);
      if (dist < bestDist) {
        best = line;
        bestDist = dist;
      }
    }
    return best;
  }

  /* ------------------------------- solving ------------------------------ */

  function isDirty() {
    return JSON.stringify(wirePins()) !== s.savedPins || s.mode !== s.savedMode;
  }

  /** Ask the server for the camera; one request at a time, newest pins win. */
  async function solve() {
    const session = getSession();
    if (!session || !s.open) return;
    if (solving) {
      solveAgain = true;
      return;
    }
    solving = true;
    const token = openToken;
    try {
      const reply = await getApi().solveCalibration(session.id, {
        pins: wirePins(),
        mode: s.mode,
        start: s.mode === "hand" ? s.camera : null,
        attacking_side: getSide(),
      });
      if (token === openToken && s.open) {
        s.H = reply.field_to_image;
        s.camera = reply.camera;
        s.fit = reply.fit;
        renderPanel();
        requestDraw();
        onChange();
      }
    } catch (error) {
      if (token === openToken) onError("Could not solve the camera", error.message || String(error));
    } finally {
      solving = false;
      if (solveAgain) {
        solveAgain = false;
        solve();
      }
    }
  }

  function remember() {
    s.history.push({ pins: JSON.stringify(s.pins), mode: s.mode });
    if (s.history.length > 80) s.history.shift();
  }

  function undo() {
    const last = s.history.pop();
    if (!last) return;
    s.pins = JSON.parse(last.pins);
    s.mode = last.mode;
    s.selected = null;
    if (s.mode === "hand") chooseNextActive();
    afterEdit();
  }

  function afterEdit() {
    renderPanel();
    requestDraw();
    onChange();
    solve();
  }

  function setMode(mode) {
    if (s.mode === mode) return;
    remember();
    s.mode = mode;
    s.pins = [];
    s.selected = null;
    s.camera = null;
    if (mode === "hand") {
      s.H = null;
      s.active = null;
      chooseNextActive();
    }
    afterEdit();
  }

  /** The end being attacked comes first: that's the end the frame usually shows. */
  function placementOrder() {
    const end = getSide() === "left" ? "l" : "r";
    const names = [
      "box_front_far",
      "box_front_near",
      "six_front_far",
      "six_front_near",
      "spot",
      "box_goal_far",
      "box_goal_near",
      "six_goal_far",
      "six_goal_near",
      "arc_far",
      "arc_near",
      "corner_far",
      "corner_near",
    ];
    return names.map((name) => `${end}_${name}`);
  }

  /** Pick the next landmark to place, after `after` when given. */
  function chooseNextActive(after) {
    const rest = placementOrder().filter((id) => !pointPin(id));
    if (!rest.length) {
      s.active = null;
      return;
    }
    const at = after ? placementOrder().indexOf(after) : -1;
    const later = rest.find((id) => placementOrder().indexOf(id) > at);
    s.active = { kind: "point", ref: later || rest[0] };
  }

  /* ------------------------------- drawing ------------------------------ */

  function requestDraw() {
    if (raf || !s.open) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      draw();
    });
  }

  function strokeWire(g, H, toPixel) {
    if (!s.pitch) return;
    g.beginPath();
    for (const line of s.pitch.lines) {
      const clipped = clipSegment(H, line.a, line.b);
      if (!clipped) continue;
      const a = toPixel(clipped[0][0], clipped[0][1]);
      const b = toPixel(clipped[1][0], clipped[1][1]);
      g.moveTo(a[0], a[1]);
      g.lineTo(b[0], b[1]);
    }
    for (const arc of s.pitch.arcs) {
      let drawing = false;
      for (const point of arc) {
        const p = project(H, point[0], point[1]);
        if (!p) {
          drawing = false;
          continue;
        }
        const q = toPixel(p[0], p[1]);
        if (drawing) g.lineTo(q[0], q[1]);
        else g.moveTo(q[0], q[1]);
        drawing = true;
      }
    }
    g.stroke();
  }

  function drawTag(text, sx, sy, width) {
    ctx.font = "500 12px 'IBM Plex Sans', 'Segoe UI', sans-serif";
    const w = ctx.measureText(text).width + 14;
    const x = Math.min(Math.max(6, sx + 12), width - w - 6);
    const y = sy > 34 ? sy - 32 : sy + 14;
    ctx.fillStyle = "rgba(7, 12, 10, 0.92)";
    ctx.fillRect(x, y, w, 22);
    ctx.fillStyle = "#eaf2ec";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + 7, y + 11.5);
    ctx.textBaseline = "alphabetic";
  }

  function drawLoupe(rect, point) {
    const size = Math.min(LOUPE_PX, rect.cw * 0.32, rect.ch * 0.42);
    const screen = toScreen(rect, point[0], point[1]);
    const x = screen[0] > rect.cw / 2 ? 12 : rect.cw - size - 12;
    const y = 12;
    const scale = rect.scale * LOUPE_ZOOM;
    const originX = x + size / 2 - point[0] * scale;
    const originY = y + size / 2 - point[1] * scale;

    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, size, size);
    ctx.clip();
    ctx.fillStyle = "#050807";
    ctx.fillRect(x, y, size, size);
    if (image.complete && image.naturalWidth) {
      ctx.drawImage(image, originX, originY, rect.iw * scale, rect.ih * scale);
    }
    if (s.H) {
      ctx.strokeStyle = WIRE_DIM;
      ctx.lineWidth = 1;
      strokeWire(ctx, s.H, (u, v) => [originX + u * scale, originY + v * scale]);
    }
    ctx.restore();

    ctx.strokeStyle = WIRE;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
    const cx = x + size / 2;
    const cy = y + size / 2;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 13, cy);
    ctx.lineTo(cx - 4, cy);
    ctx.moveTo(cx + 4, cy);
    ctx.lineTo(cx + 13, cy);
    ctx.moveTo(cx, cy - 13);
    ctx.lineTo(cx, cy - 4);
    ctx.moveTo(cx, cy + 4);
    ctx.lineTo(cx, cy + 13);
    ctx.stroke();
  }

  function draw() {
    if (!s.open) return;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    if (!cw || !ch) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const tw = Math.round(cw * dpr);
    const th = Math.round(ch * dpr);
    if (canvas.width !== tw || canvas.height !== th) {
      canvas.width = tw;
      canvas.height = th;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#050807";
    ctx.fillRect(0, 0, cw, ch);

    clampPan();
    const rect = viewRect();
    if (image.complete && image.naturalWidth) {
      ctx.drawImage(image, rect.x, rect.y, rect.w, rect.h);
    }

    if (s.H) {
      ctx.save();
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = "rgba(0, 0, 0, 0.45)";
      ctx.lineWidth = 3.2;
      strokeWire(ctx, s.H, (u, v) => toScreen(rect, u, v));
      ctx.strokeStyle = WIRE_DIM;
      ctx.lineWidth = 1.5;
      strokeWire(ctx, s.H, (u, v) => toScreen(rect, u, v));
      if (s.hover && s.hover.line) {
        const clipped = clipSegment(s.H, s.hover.line.a, s.hover.line.b);
        if (clipped) {
          const a = toScreen(rect, clipped[0][0], clipped[0][1]);
          const b = toScreen(rect, clipped[1][0], clipped[1][1]);
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b[0], b[1]);
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    const list = handles(rect);
    for (const handle of list) {
      const selected = s.selected === handle.key;
      const hovered = s.hover && s.hover.key === handle.key;
      const radius = selected ? 7.5 : hovered ? 7 : 5.5;
      ctx.beginPath();
      if (handle.pin && handle.pin.kind === "line") {
        ctx.moveTo(handle.sx, handle.sy - radius - 1);
        ctx.lineTo(handle.sx + radius + 1, handle.sy);
        ctx.lineTo(handle.sx, handle.sy + radius + 1);
        ctx.lineTo(handle.sx - radius - 1, handle.sy);
        ctx.closePath();
      } else {
        ctx.arc(handle.sx, handle.sy, radius, 0, Math.PI * 2);
      }
      ctx.fillStyle = handle.pin ? PINNED : "rgba(7, 12, 10, 0.72)";
      ctx.fill();
      ctx.lineWidth = 2;
      const doubtful =
        handle.pin && s.fit && s.fit.worst != null && s.pins[s.fit.worst] === handle.pin;
      ctx.strokeStyle = doubtful ? "#ff4a3d" : handle.pin ? "#ffffff" : WIRE;
      ctx.stroke();
    }

    if (s.drag && s.drag.moved) {
      const pin = s.pins.find((row) => row.key === s.drag.key);
      if (pin) drawLoupe(rect, [pin.x, pin.y]);
    } else if (s.hover && s.hover.name && !s.pan) {
      drawTag(s.hover.name, s.hover.sx, s.hover.sy, cw);
    } else if (s.selected) {
      const handle = list.find((row) => row.key === s.selected);
      if (handle) drawTag(handle.name, handle.sx, handle.sy, cw);
    }

    drawMap();
  }

  /* -------------------------------- map --------------------------------- */

  function mapPoint(x, y) {
    if (!s.pitch) return [0, 0];
    const k = Math.min((MAP_W - 2 * MAP_PAD) / s.pitch.length, (MAP_H - 2 * MAP_PAD) / s.pitch.width);
    const ox = (MAP_W - s.pitch.length * k) / 2;
    const oy = (MAP_H - s.pitch.width * k) / 2;
    return [ox + x * k, oy + y * k];
  }

  function drawMap() {
    if (!mapCtx || !s.pitch || s.mode !== "hand") return;
    const g = mapCtx;
    if (ui.map.width !== MAP_W) ui.map.width = MAP_W;
    if (ui.map.height !== MAP_H) ui.map.height = MAP_H;
    g.fillStyle = "#123521";
    g.fillRect(0, 0, MAP_W, MAP_H);
    g.lineWidth = 2;
    for (const line of s.pitch.lines) {
      const a = mapPoint(line.a[0], line.a[1]);
      const b = mapPoint(line.b[0], line.b[1]);
      const active = s.active && s.active.kind === "line" && s.active.ref === line.id;
      g.strokeStyle = active ? WIRE : "rgba(234, 242, 236, 0.6)";
      g.lineWidth = active ? 5 : 2;
      g.beginPath();
      g.moveTo(a[0], a[1]);
      g.lineTo(b[0], b[1]);
      g.stroke();
    }
    g.strokeStyle = "rgba(234, 242, 236, 0.6)";
    g.lineWidth = 2;
    for (const arc of s.pitch.arcs) {
      g.beginPath();
      arc.forEach((point, index) => {
        const q = mapPoint(point[0], point[1]);
        if (index) g.lineTo(q[0], q[1]);
        else g.moveTo(q[0], q[1]);
      });
      g.stroke();
    }
    for (const row of s.pitch.landmarks) {
      const q = mapPoint(row.x, row.y);
      const placed = Boolean(pointPin(row.id));
      const active = s.active && s.active.kind === "point" && s.active.ref === row.id;
      g.beginPath();
      g.arc(q[0], q[1], active ? 9 : 5.5, 0, Math.PI * 2);
      g.fillStyle = placed ? PINNED : active ? WIRE : "#0a100d";
      g.fill();
      g.lineWidth = 2;
      g.strokeStyle = placed ? "#ffffff" : WIRE;
      g.stroke();
    }
  }

  function onMapPointer(event) {
    if (!s.pitch) return;
    const bounds = ui.map.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * MAP_W;
    const y = ((event.clientY - bounds.top) / bounds.height) * MAP_H;
    let best = null;
    let bestDist = 18;
    for (const row of s.pitch.landmarks) {
      const q = mapPoint(row.x, row.y);
      const dist = Math.hypot(q[0] - x, q[1] - y);
      if (dist < bestDist) {
        best = { kind: "point", ref: row.id };
        bestDist = dist;
      }
    }
    if (!best) {
      bestDist = 12;
      for (const line of s.pitch.lines) {
        const a = mapPoint(line.a[0], line.a[1]);
        const b = mapPoint(line.b[0], line.b[1]);
        const dist = segmentDistance(x, y, a[0], a[1], b[0], b[1]);
        if (dist < bestDist) {
          best = { kind: "line", ref: line.id };
          bestDist = dist;
        }
      }
    }
    if (best) {
      s.active = best;
      renderPanel();
      requestDraw();
    }
  }

  /* -------------------------------- panel ------------------------------- */

  function activeName() {
    if (!s.active) return "";
    const row = s.active.kind === "point" ? landmark(s.active.ref) : pitchLine(s.active.ref);
    return row ? row.name : "";
  }

  function renderPanel() {
    const hand = s.mode === "hand";
    ui.handBox.hidden = !hand;
    if (hand) {
      if (s.active && s.active.kind === "line") {
        ui.prompt.textContent = `Click anywhere on this line in the photo: ${activeName()}`;
      } else if (s.active) {
        ui.prompt.textContent = `Now click this in the photo: ${activeName()}`;
      } else {
        ui.prompt.textContent = "Every point is placed. Drag any of them to fine-tune.";
      }
    }
    ui.skip.disabled = !s.active;
    ui.undo.disabled = !s.history.length;
    ui.reset.disabled = !s.pins.length;

    // Switch starting point: only offered when there's an automatic camera.
    ui.mode.hidden = !s.autoAvailable;
    ui.mode.textContent = hand ? "Use the automatic result" : "Place by hand instead";

    // Continue without a 3D camera: only when no camera is available.
    ui.without3d.hidden = !(s.status === "ready" && !s.autoAvailable && !s.H);

    if (s.status !== "ready" && !hand) {
      ui.help.textContent = "You can start placing points by hand while this runs.";
    } else if (hand && !s.H) {
      ui.help.textContent = "Pick a spot on the map, then click the same spot in the photo.";
    } else {
      ui.help.textContent =
        "Drag a yellow point onto its pitch marking, or drag a yellow line onto the real line. Green points stay where you put them.";
    }
    requestDraw();
  }

  /* -------------------------------- input ------------------------------- */

  function placeActive(point) {
    if (!s.active) return;
    remember();
    if (s.active.kind === "point") {
      const existing = pointPin(s.active.ref);
      const ref = s.active.ref;
      if (existing) {
        existing.x = point[0];
        existing.y = point[1];
      } else {
        s.pins.push({ kind: "point", ref, x: point[0], y: point[1], key: `p:${ref}` });
      }
      s.selected = `p:${ref}`;
      chooseNextActive(ref);
    } else {
      const ref = s.active.ref;
      const key = `l:${ref}:${(pinCounter += 1)}`;
      s.pins.push({ kind: "line", ref, x: point[0], y: point[1], key });
      s.selected = key;
      // A line needs two spots to pin its direction.
      if (s.pins.filter((pin) => pin.kind === "line" && pin.ref === ref).length >= 2) {
        chooseNextActive();
      }
    }
    afterEdit();
  }

  function onPointerDown(event) {
    if (!s.open) return;
    canvas.focus({ preventScroll: true });
    const [sx, sy] = eventPoint(event);
    const rect = viewRect();

    if (event.button === 1 || event.button === 2) {
      s.pan = { sx, sy, panX: s.panX, panY: s.panY };
      canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (event.button !== 0) return;

    const handle = hitHandle(rect, sx, sy);
    const line = handle ? null : hitLine(rect, sx, sy);
    s.drag = {
      key: handle ? handle.key : null,
      handle,
      line,
      startX: sx,
      startY: sy,
      panX: s.panX,
      panY: s.panY,
      offsetX: handle ? handle.sx - sx : 0,
      offsetY: handle ? handle.sy - sy : 0,
      moved: false,
      remembered: false,
    };
    if (handle) s.selected = handle.key;
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
    requestDraw();
  }

  /** The first real movement of a drag turns the grabbed thing into a pin. */
  function beginDragPin(drag) {
    remember();
    drag.remembered = true;
    if (drag.handle && drag.handle.pin) {
      drag.key = drag.handle.pin.key;
      return;
    }
    if (drag.handle && drag.handle.landmark) {
      const ref = drag.handle.landmark.id;
      const key = `p:${ref}`;
      s.pins.push({ kind: "point", ref, x: drag.handle.ix, y: drag.handle.iy, key });
      drag.key = key;
      s.selected = key;
      if (s.active && s.active.ref === ref) chooseNextActive(ref);
      return;
    }
    if (drag.line) {
      const key = `l:${drag.line.id}:${(pinCounter += 1)}`;
      const rect = viewRect();
      const [x, y] = toImage(rect, drag.startX, drag.startY);
      s.pins.push({ kind: "line", ref: drag.line.id, x, y, key });
      drag.key = key;
      s.selected = key;
    }
  }

  function onPointerMove(event) {
    if (!s.open) return;
    const [sx, sy] = eventPoint(event);

    if (s.pan) {
      s.panX = s.pan.panX + sx - s.pan.sx;
      s.panY = s.pan.panY + sy - s.pan.sy;
      requestDraw();
      return;
    }

    const drag = s.drag;
    if (!drag) {
      const rect = viewRect();
      const handle = hitHandle(rect, sx, sy);
      const line = handle ? null : hitLine(rect, sx, sy);
      const next = handle
        ? { key: handle.key, name: handle.name, sx: handle.sx, sy: handle.sy }
        : line
          ? { key: `line:${line.id}`, line, name: line.name, sx, sy }
          : null;
      const changed = (next && next.key) !== (s.hover && s.hover.key);
      s.hover = next;
      canvas.style.cursor = handle || line ? "grab" : s.mode === "hand" && s.active ? "crosshair" : "default";
      if (changed || line) requestDraw();
      return;
    }

    if (!drag.moved && Math.hypot(sx - drag.startX, sy - drag.startY) < DRAG_START_PX) return;
    drag.moved = true;
    s.hover = null;

    if (drag.handle || drag.line) {
      if (!drag.remembered) beginDragPin(drag);
      const rect = viewRect();
      const [x, y] = toImage(rect, sx + drag.offsetX, sy + drag.offsetY);
      const pin = s.pins.find((row) => row.key === drag.key);
      if (pin) {
        pin.x = Math.max(0, Math.min(rect.iw, x));
        pin.y = Math.max(0, Math.min(rect.ih, y));
      }
      canvas.style.cursor = "grabbing";
      requestDraw();
      onChange();
      solve();
      return;
    }

    // Dragging empty space moves a zoomed photo.
    s.panX = drag.panX + sx - drag.startX;
    s.panY = drag.panY + sy - drag.startY;
    requestDraw();
  }

  function onPointerUp(event) {
    if (!s.open) return;
    if (s.pan) {
      s.pan = null;
      requestDraw();
      return;
    }
    const drag = s.drag;
    s.drag = null;
    if (!drag) return;
    if (drag.moved) {
      if (drag.handle || drag.line) afterEdit();
      else requestDraw();
      return;
    }
    // A plain click.
    const placing = s.mode === "hand" && s.active;
    const onPin = drag.handle && drag.handle.pin;
    if (placing && !onPin) {
      const [sx, sy] = eventPoint(event);
      const rect = viewRect();
      const [x, y] = toImage(rect, sx, sy);
      if (x >= 0 && y >= 0 && x <= rect.iw && y <= rect.ih) placeActive([x, y]);
      return;
    }
    if (!drag.handle) s.selected = null;
    requestDraw();
  }

  function onPointerCancel() {
    s.pan = null;
    s.drag = null;
    requestDraw();
  }

  function onLeave() {
    if (s.hover) {
      s.hover = null;
      requestDraw();
    }
  }

  function onWheel(event) {
    if (!s.open) return;
    event.preventDefault();
    const [sx, sy] = eventPoint(event);
    const before = viewRect();
    const [ix, iy] = toImage(before, sx, sy);
    s.zoom = Math.max(1, Math.min(MAX_ZOOM, s.zoom * Math.exp(-event.deltaY * 0.0016)));
    const after = viewRect();
    s.panX += sx - (after.x + ix * after.scale);
    s.panY += sy - (after.y + iy * after.scale);
    requestDraw();
  }

  function onDoubleClick(event) {
    if (!s.open) return;
    const [sx, sy] = eventPoint(event);
    if (hitHandle(viewRect(), sx, sy)) return;
    s.zoom = 1;
    s.panX = 0;
    s.panY = 0;
    requestDraw();
  }

  /** Keyboard while the step is open.  Returns true when the key was used. */
  function handleKey(event) {
    if (!s.open) return false;
    const key = event.key;
    if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "z") {
      undo();
      return true;
    }
    if (!s.selected) return false;
    const step = event.shiftKey ? 5 : 1;
    const delta = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    }[key];
    if (delta) {
      let pin = s.pins.find((row) => row.key === s.selected);
      if (!pin) {
        const handle = handles(viewRect()).find((row) => row.key === s.selected);
        if (!handle || !handle.landmark) return false;
        remember();
        const ref = handle.landmark.id;
        pin = { kind: "point", ref, x: handle.ix, y: handle.iy, key: `p:${ref}` };
        s.pins.push(pin);
        s.selected = pin.key;
      } else {
        remember();
      }
      pin.x += delta[0];
      pin.y += delta[1];
      afterEdit();
      return true;
    }
    if (key === "Delete" || key === "Backspace") {
      const index = s.pins.findIndex((row) => row.key === s.selected);
      if (index < 0) return false;
      remember();
      s.pins.splice(index, 1);
      s.selected = null;
      afterEdit();
      return true;
    }
    return false;
  }

  /* ---------------------------- open and close -------------------------- */

  function applyPayload(payload) {
    s.status = payload.status;
    s.source = payload.source;
    s.autoAvailable = Boolean(payload.auto_3d);
    s.pitch = payload.pitch || s.pitch;
    return payload;
  }

  /** While automatic calibration runs: poll, then adopt it unless the user took over. */
  function poll(token) {
    const session = getSession();
    if (!session || token !== openToken || !s.open) return;
    pollTimer = setTimeout(async () => {
      try {
        const payload = await getApi().getCalibration(session.id);
        if (token !== openToken || !s.open) return;
        applyPayload(payload);
        if (payload.status !== "ready") {
          poll(token);
          return;
        }
        if (!s.pins.length && s.mode === "edit") {
          s.mode = payload.mode;
          s.savedMode = payload.mode;
          s.H = payload.field_to_image;
          s.camera = payload.camera;
          s.fit = payload.fit;
          if (s.mode === "hand") chooseNextActive();
        }
        renderPanel();
        requestDraw();
        onChange();
      } catch (error) {
        if (token === openToken) onError("Calibration unavailable", error.message || String(error));
      }
    }, POLL_MS);
  }

  /** Open the step and load the current calibration. */
  async function open() {
    const session = getSession();
    if (!session) return;
    openToken += 1;
    const token = openToken;
    clearTimeout(pollTimer);
    s.open = true;
    s.zoom = 1;
    s.panX = 0;
    s.panY = 0;
    s.history = [];
    s.selected = null;
    s.hover = null;
    s.drag = null;
    s.pan = null;
    s.active = null;
    s.fit = { tone: "idle", label: "LOADING", message: "Opening calibration…", can_continue: false };
    onChange();
    requestDraw();
    try {
      const payload = await getApi().getCalibration(session.id);
      if (token !== openToken || !s.open) return;
      applyPayload(payload);
      s.mode = payload.mode;
      s.savedMode = payload.mode;
      s.H = payload.field_to_image;
      s.camera = payload.camera;
      s.fit = payload.fit;
      loadPins(payload.pins);
      s.savedPins = JSON.stringify(wirePins());
      if (s.mode === "hand") chooseNextActive();
      renderPanel();
      requestDraw();
      onChange();
      if (payload.status !== "ready") poll(token);
    } catch (error) {
      if (token === openToken) onError("Calibration unavailable", error.message || String(error));
    }
  }

  /** Close the step (nothing is sent to the server). */
  function close() {
    openToken += 1;
    s.open = false;
    clearTimeout(pollTimer);
    cancelAnimationFrame(raf);
    raf = 0;
  }

  /** Save what's on screen.  Returns true when the server's calibration changed. */
  async function commit() {
    const session = getSession();
    if (!session || !isDirty()) return false;
    if (s.mode !== "hand" && !s.pins.length) {
      await getApi().resetCalibration(session.id);
    } else {
      await getApi().applyCalibration(session.id, {
        pins: wirePins(),
        mode: s.mode,
        attacking_side: getSide(),
      });
    }
    s.savedPins = JSON.stringify(wirePins());
    s.savedMode = s.mode;
    return true;
  }

  /** Drop any hand work so Analyze uses the automatic result, 3D or not. */
  async function discard() {
    const session = getSession();
    if (session) await getApi().resetCalibration(session.id);
    s.pins = [];
    s.savedPins = "[]";
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerCancel);
  canvas.addEventListener("pointerleave", onLeave);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("dblclick", onDoubleClick);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  if (ui.map) ui.map.addEventListener("pointerdown", onMapPointer);
  ui.undo.addEventListener("click", undo);
  ui.reset.addEventListener("click", () => {
    if (!s.pins.length) return;
    remember();
    s.pins = [];
    s.selected = null;
    if (s.mode === "hand") {
      s.H = null;
      s.camera = null;
      s.active = null;
      chooseNextActive();
    }
    afterEdit();
  });
  ui.skip.addEventListener("click", () => {
    if (!s.active) return;
    if (s.active.kind === "point") chooseNextActive(s.active.ref);
    else chooseNextActive();
    renderPanel();
    requestDraw();
  });
  ui.mode.addEventListener("click", () => setMode(s.mode === "hand" ? "edit" : "hand"));
  image.addEventListener("load", requestDraw);

  return {
    open,
    close,
    commit,
    discard,
    handleKey,
    redraw: requestDraw,
    /** What the app needs for its status block and primary button. */
    info() {
      return {
        open: s.open,
        status: s.status,
        fit: s.fit,
        mode: s.mode,
        dirty: isDirty(),
        hasCamera: Boolean(s.H),
        canContinue: Boolean(s.fit && s.fit.can_continue && s.status === "ready"),
      };
    },
    /** Test hook: where a landmark currently draws, in image pixels. */
    debug() {
      return { state: s, project };
    },
  };
}
