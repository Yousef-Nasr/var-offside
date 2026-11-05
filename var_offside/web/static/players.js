/**
 * Player markers on the photo: a ring on the ground and a label above the head.
 *
 * Team colours are fixed: Team 1 is always red and Team 2 always blue, whatever the
 * kits look like. Every position here is in screen pixels.
 */

export const TEAM_COLORS = { 1: "#ff3347", 2: "#2e7dff" };
const REFEREE_COLOR = "#9aa8a0";
const NO_TEAM_COLOR = "#c9d3cc";
const LINE_COLOR = "#38d6ff";
const OFFSIDE_COLOR = "#ff3347";

/** Marker colour for a player. */
export function markerColor(player) {
  if (player.role === "referee") return REFEREE_COLOR;
  return TEAM_COLORS[player.team] || NO_TEAM_COLOR;
}

/** One half of an ellipse: from the left end, round the front or the back. */
function halfEllipse(ctx, cx, cy, rx, ry, front) {
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, Math.PI, 0, front);
}

/**
 * Draw the ring under a player's feet.
 * @param {CanvasRenderingContext2D} ctx
 * @param {{x1:number,y1:number,x2:number,y2:number}} box Player box on screen.
 * @param {object} style
 * @param {string} style.color Team colour.
 * @param {boolean} [style.keeper] Dashed ring for a goalkeeper.
 * @param {boolean} [style.hovered] Brighter ring.
 * @param {"offside"|"defender"|null} [style.emphasis] Extra ring for the two deciding players.
 * @param {number} [style.beat] 0 to 1, for the slow pulse of the extra ring.
 */
export function drawPlayerRing(ctx, box, style) {
  const width = Math.max(box.x2 - box.x1, 8);
  const rx = Math.max(width * 0.62, 9);
  const ry = Math.max(rx * 0.3, 3);
  const cx = (box.x1 + box.x2) / 2;
  const cy = box.y2 - ry * 0.25;

  ctx.save();
  ctx.lineCap = "round";

  // Soft pool of colour on the grass.
  const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, rx);
  glow.addColorStop(0, withAlpha(style.color, style.hovered ? 0.5 : 0.34));
  glow.addColorStop(0.75, withAlpha(style.color, style.hovered ? 0.3 : 0.18));
  glow.addColorStop(1, withAlpha(style.color, 0.04));
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(1, ry / rx);
  ctx.translate(-cx, -cy);
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(cx, cy, rx, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  if (style.keeper) ctx.setLineDash([rx * 0.32, rx * 0.2]);

  // The far half sits behind the player, so it stays quiet.
  halfEllipse(ctx, cx, cy, rx, ry, false);
  ctx.strokeStyle = withAlpha(style.color, 0.6);
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // The near half carries the colour, with a dark edge so it holds on bright grass.
  halfEllipse(ctx, cx, cy, rx, ry, true);
  ctx.strokeStyle = "rgba(4, 8, 6, 0.55)";
  ctx.lineWidth = 4.5;
  ctx.stroke();
  ctx.strokeStyle = style.color;
  ctx.lineWidth = style.hovered ? 3.2 : 2.6;
  ctx.stroke();
  ctx.setLineDash([]);

  if (style.emphasis) {
    const beat = style.beat == null ? 0.5 : style.beat;
    const grow = style.emphasis === "offside" ? 1.22 + beat * 0.14 : 1.22;
    ctx.globalAlpha = style.emphasis === "offside" ? 0.55 + beat * 0.45 : 0.95;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx * grow, ry * grow, 0, 0, Math.PI * 2);
    ctx.strokeStyle = style.emphasis === "offside" ? "#ffffff" : LINE_COLOR;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

/** Corner brackets around a hovered player, marking what a click will open. */
export function drawHoverBrackets(ctx, box, color) {
  const w = box.x2 - box.x1;
  const h = box.y2 - box.y1;
  const arm = Math.min(9, w * 0.3, h * 0.3);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (const [x, y, dx, dy] of [
    [box.x1, box.y1, 1, 1],
    [box.x2, box.y1, -1, 1],
    [box.x1, box.y2, 1, -1],
    [box.x2, box.y2, -1, -1],
  ]) {
    ctx.moveTo(x + dx * arm, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + dy * arm);
  }
  ctx.stroke();
  ctx.restore();
}

/** Parts of a label, left to right. */
function labelParts(player, emphasis) {
  const parts = [{ text: player.id == null ? "NEW" : String(player.id), kind: "number" }];
  if (player.role === "goalkeeper") parts.push({ text: "GK", kind: "plain" });
  if (player.role === "referee") parts.push({ text: "REF", kind: "plain" });
  if (emphasis === "defender") parts.push({ text: "LAST DEFENDER", kind: "line" });
  if (emphasis === "offside") parts.push({ text: "OFFSIDE", kind: "offside" });
  return parts;
}

/**
 * Lay out and draw the labels above the players.  Deciding players claim their
 * place first; the others move up or drop out so that nothing overlaps.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array<{player:object, box:object, color:string, emphasis:string|null}>} items
 */
export function drawPlayerLabels(ctx, items) {
  const HEIGHT = 18;
  const PAD = 6;
  const placed = [];
  const font = "700 13px 'Barlow Condensed', 'Arial Narrow', sans-serif";

  ctx.save();
  ctx.font = font;
  ctx.textBaseline = "middle";

  const ordered = [...items].sort((a, b) => Number(Boolean(b.emphasis)) - Number(Boolean(a.emphasis)));
  for (const item of ordered) {
    const parts = labelParts(item.player, item.emphasis).map((part) => ({
      ...part,
      width: ctx.measureText(part.text).width + PAD * 2,
    }));
    const total = parts.reduce((sum, part) => sum + part.width, 0);
    const cx = (item.box.x1 + item.box.x2) / 2;
    const x = Math.max(4, Math.min(cx - total / 2, ctx.canvas.clientWidth - total - 4));
    let y = Math.max(4, item.box.y1 - HEIGHT - 7);

    let tries = item.emphasis ? 6 : 2;
    const hits = () =>
      placed.some(
        (rect) => x < rect.x + rect.w && x + total > rect.x && y < rect.y + rect.h && y + HEIGHT > rect.y,
      );
    while (hits() && tries > 0) {
      y = Math.max(4, y - HEIGHT - 2);
      tries -= 1;
    }
    if (hits() && !item.emphasis) continue;
    placed.push({ x, y, w: total, h: HEIGHT });

    // Pointer down to the head.
    const tip = Math.max(x + 6, Math.min(cx, x + total - 6));
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.moveTo(tip - 4, y + HEIGHT - 0.5);
    ctx.lineTo(tip + 4, y + HEIGHT - 0.5);
    ctx.lineTo(tip, y + HEIGHT + 5);
    ctx.closePath();
    ctx.fill();

    // Plate.
    ctx.save();
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, total, HEIGHT, 4);
    else ctx.rect(x, y, total, HEIGHT);
    ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
    ctx.shadowBlur = 6;
    ctx.shadowOffsetY = 1;
    ctx.fillStyle = "rgba(7, 12, 10, 0.92)";
    ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.clip();

    let at = x;
    for (const part of parts) {
      if (part.kind === "number") ctx.fillStyle = item.color;
      else if (part.kind === "offside") ctx.fillStyle = OFFSIDE_COLOR;
      else if (part.kind === "line") ctx.fillStyle = LINE_COLOR;
      else ctx.fillStyle = "rgba(7, 12, 10, 0.92)";
      ctx.fillRect(at, y, part.width, HEIGHT);
      ctx.fillStyle = part.kind === "line" ? "#04141a" : "#ffffff";
      ctx.fillText(part.text, at + PAD, y + HEIGHT / 2 + 0.5);
      at += part.width;
    }
    ctx.restore();
  }
  ctx.restore();
}

/** `#rrggbb` to `rgba()`. */
function withAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
