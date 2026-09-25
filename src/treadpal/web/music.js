"use strict";

// Live music visuals: spectrum/chroma frames from the agent, the beat grid
// from the server, a phase-locked beat clock, and the subtle ways the rest of
// the UI moves with the music.

const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
// Lane lightness (hue and saturation come from each lane's own note)
const LANE_LIGHT = { vocals: 82, other: 72, bass: 62, drums: 74, high: 80, mid: 70 };
const NOTE_SAT = 42; // Saturation of a lane with a clear note; fades to grey without one
const DEFAULT_HUE = 250; // Resting lavender when nothing is playing
const noteHue = (n) => n * 30; // Pitch classes are cyclic, so is hue: C red … B rose
const SVGNS = "http://www.w3.org/2000/svg";

const M = {
  frame: null,
  frameAt: -Infinity,
  soundAt: -Infinity, // Last frame that wasn't silence
  zones: [12, 20, 12],
  // Stem frames from the server's separator (vocals/other/bass/drums)
  stemFrame: null,
  stemAt: -Infinity,
  // Visualiser lanes, top to bottom: stems when separated, else the agent's frequency zones
  laneKey: null,
  laneNames: [],
  lanes: [], // Smoothed energy per lane
  laneSpec: [], // Smoothed spectrum per lane, low to high
  lanePeaks: [],
  laneNotes: [], // Per-lane note tracker (stems only): { chroma, clarity, note, hue, sat }
  chroma: new Array(12).fill(0),
  note: null, // Dominant pitch class of the smoothed chroma
  clarity: 0,
  hx: Math.cos((DEFAULT_HUE * Math.PI) / 180),
  hy: Math.sin((DEFAULT_HUE * Math.PI) / 180),
  hue: DEFAULT_HUE,
  // Beat clock: continuous beat count, integers fall on beats (display offset included)
  phase: 0,
  period: null,
  gridAt: -Infinity,
  conf: 0,
  harmonic: null,
  bar: null,
  barOffset: 0, // floor(phase) - barOffset ≡ 0 (mod bar) on downbeats
  onsetEnv: 0,
  pulse: 0,
  cssCache: {},
  // Visualiser history
  hist: [], // { t, b: lane energies, h: lane hues, s: lane saturations, seq }
  histAt: 0,
  histSeq: 0, // Running sample number, so gradient stops stay pinned to their samples
  beats: [], // { t, down }
  lastBeat: null,
  notes: [], // { t, note } when the dominant note changes
  noteCand: null,
  noteSince: 0,
};
const HIST_S = 6;

const vizLive = () => now() - M.frameAt < 1500;
const stemsLive = () => now() - M.stemAt < 1000;
// The server sends a grid ~2.5 times a second while music plays, none in silence
const gridLive = () => M.period != null && now() - M.gridAt < 4000;

/**
 * Music is actually sounding now, not just a source connected: the agent's
 * frames carry sound, or (for BPM-only sources) the beat grid is fresh.
 */
function musicPlaying() {
  if (vizLive()) return now() - M.soundAt < 1500;
  return gridLive() || (musicLive() && S.music?.age_s != null && S.music.age_s < 5);
}

/** Beat period in seconds while music plays: the server's grid, else the polled BPM. */
function beatPeriod() {
  if (!musicPlaying()) return null;
  if (gridLive()) return M.period;
  const bpm = S.music?.detected_bpm;
  return bpm ? 60 / bpm : null;
}

const beatInBar = () => {
  const bar = M.bar ?? 4;
  return (((Math.floor(M.phase) - M.barOffset) % bar) + bar) % bar;
};

// ---------- stream ----------

function connectViz() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/viz`);
  ws.onmessage = (e) => {
    let m;
    try {
      m = JSON.parse(e.data);
    } catch {
      return;
    }
    if (m.type === "viz") onVizFrame(m);
    else if (m.type === "stems") onStemFrame(m);
    else if (m.type === "beat") onBeatGrid(m);
    else if (m.type === "viz_end") {
      M.frameAt = -Infinity;
      M.stemAt = -Infinity;
      M.gridAt = -Infinity;
    }
  };
  ws.onclose = () => setTimeout(connectViz, 2000);
}

function onVizFrame(m) {
  M.frameAt = now();
  M.frame = m.silent ? null : m;
  if (!m.silent) M.soundAt = M.frameAt;
  if (m.zones) M.zones = m.zones;
  // With stems, kicks come from the drum stem instead (no bass notes mistaken for kicks)
  if (m.onset > 0 && !stemsLive()) onOnset(m.onset);
}

function onStemFrame(m) {
  M.stemAt = now();
  M.stemFrame = m.silent ? null : m;
  if (!m.silent) M.soundAt = M.stemAt;
  if (m.onset > 0) onOnset(m.onset, m.lag ?? 0);
}

/**
 * Lock the beat clock to the server's grid. The grid says "the last beat was
 * `age` seconds ago", so no clock sync is needed. Small errors are eased out;
 * a confident, large error (new song, first lock) snaps.
 */
function onBeatGrid(m) {
  const P = m.period;
  const offsetBeats = prefs.beatOffsetMs / 1000 / P;
  const target = m.age / P + offsetBeats; // Beats since the grid's last beat
  if (!gridLive()) {
    M.phase = Math.floor(M.phase) + target; // Keep the beat count, take the fraction
  } else {
    let err = target - M.phase;
    err -= Math.round(err);
    M.phase += Math.abs(err) > 0.2 && m.conf > 0.6 ? err : err * 0.5;
  }
  if (m.bar && m.down_age != null) {
    // Which beat of the bar the grid's last beat was, and so where downbeats fall
    const beatsSinceDown = Math.round((m.down_age - m.age) / P);
    const lastBeatIdx = Math.round(M.phase - target);
    M.barOffset = lastBeatIdx - beatsSinceDown;
    M.bar = m.bar;
  }
  M.period = P;
  M.gridAt = now();
  M.conf = m.conf;
  M.harmonic = m.harmonic ?? M.harmonic;
}

/**
 * Onsets have tight timing; let them gently pull the clock onto the beat.
 * `lag` is how long before now the onset actually sounded (separation time).
 */
function onOnset(strength, lag = 0) {
  M.onsetEnv = Math.max(M.onsetEnv, strength);
  if (!gridLive()) return;
  const raw = M.phase - (lag + prefs.beatOffsetMs / 1000) / M.period;
  const frac = raw - Math.round(raw);
  if (Math.abs(frac) < 0.15) M.phase -= frac * 0.12;
}

function shiftBeatOffset(deltaMs) {
  const P = beatPeriod();
  if (P) M.phase += deltaMs / 1000 / P;
}

// ---------- per-frame ----------

function ease(cur, target, dt, attack, release) {
  return cur + (target - cur) * (1 - Math.exp(-dt / (target > cur ? attack : release)));
}

const noteTracker = (hue) => ({
  chroma: new Array(12).fill(0),
  clarity: 0,
  note: null,
  hx: Math.cos((hue * Math.PI) / 180),
  hy: Math.sin((hue * Math.PI) / 180),
  hue,
  sat: 0,
});

/**
 * Follow a chroma: its dominant note, how clearly it stands out, and a hue that
 * glides around the wheel toward that note (vector average, so B→C goes the
 * short way). Without a clear note the hue drifts to `idleHue`, or holds if null.
 */
function trackNote(tr, chroma, dt, idleHue) {
  // The chroma already spans ~190 ms of audio; a little easing just steadies it
  for (let i = 0; i < 12; i++) tr.chroma[i] = ease(tr.chroma[i], chroma?.[i] ?? 0, dt, 0.1, 0.15);
  let best = 0;
  let sum = 0;
  for (let i = 0; i < 12; i++) {
    sum += tr.chroma[i];
    if (tr.chroma[i] > tr.chroma[best]) best = i;
  }
  const top = tr.chroma[best];
  // 1 for a lone note; ~0.35 for noise, where every pitch class is about as loud
  tr.clarity = top > 0.05 ? clamp((top - (sum - top) / 11) / top, 0, 1) : 0;
  tr.note = top > 0.05 ? best : null;
  const clear = chroma && tr.note != null && tr.clarity > 0.2;
  // Colour only for a clear note; noisy sound (most drums) stays grey
  tr.sat = ease(tr.sat, clear ? NOTE_SAT * clamp((tr.clarity - 0.45) / 0.3, 0, 1) : 0, dt, 0.15, 0.4);
  if (!clear && idleHue == null) return;
  const th = ((clear ? noteHue(tr.note) : idleHue) * Math.PI) / 180;
  const k = 1 - Math.exp(-dt / (clear ? 0.25 : 3));
  tr.hx += (Math.cos(th) - tr.hx) * k;
  tr.hy += (Math.sin(th) - tr.hy) * k;
  tr.hue = ((Math.atan2(tr.hy, tr.hx) * 180) / Math.PI + 360) % 360;
}

// A lane's colour: its own note when separated, else the overall note
const laneHue = (z) => M.laneNotes[z]?.hue ?? M.hue;
const laneSat = (z) => M.laneNotes[z]?.sat ?? NOTE_SAT;

/**
 * What the lanes should show now, top to bottom: the separated stems when the
 * server is splitting the music, else the agent's high/mid/bass frequency zones.
 */
function laneTargets() {
  if (stemsLive()) {
    const f = M.stemFrame;
    const names = f?.names ?? M.laneNames;
    const nb = f?.bands ?? 16;
    return {
      key: `stems:${names.join()}`,
      names,
      level: names.map((_, i) => f?.level?.[i] ?? 0),
      spec: names.map((_, i) => Array.from({ length: nb }, (_, j) => f?.spec?.[i * nb + j] ?? 0)),
      // Per-stem notes (older servers don't send them: lanes then share the overall note)
      chroma: f?.stem_chroma ? names.map((_, i) => f.stem_chroma.slice(i * 12, i * 12 + 12)) : null,
    };
  }
  const f = vizLive() ? M.frame : null;
  let off = 0;
  const zoneSpec = M.zones.map((count) => {
    const s = Array.from({ length: count }, (_, j) => f?.spec?.[off + j] ?? 0);
    off += count;
    return s;
  });
  return {
    key: "zones",
    names: ["high", "mid", "bass"],
    level: [2, 1, 0].map((z) => f?.bands?.[z] ?? 0),
    spec: [2, 1, 0].map((z) => zoneSpec[z]),
  };
}

function updateMusic(dt) {
  const f = vizLive() ? M.frame : null;

  const L = laneTargets();
  if (L.key !== M.laneKey) {
    // Different lanes (stems came or went): start the picture afresh
    M.laneKey = L.key;
    M.laneNames = L.names;
    M.lanes = L.level.map(() => 0);
    M.laneSpec = L.spec.map((s) => s.map(() => 0));
    M.lanePeaks = L.spec.map((s) => s.map(() => 0));
    M.laneNotes = [];
    M.hist = [];
  }
  // Light smoothing only: frames arrive ~43/s, so ~80 ms of release just removes flicker
  L.level.forEach((v, i) => (M.lanes[i] = ease(M.lanes[i], v, dt, 0.015, 0.08)));
  L.spec.forEach((s, i) =>
    s.forEach((v, j) => {
      const cur = (M.laneSpec[i][j] = ease(M.laneSpec[i][j] ?? 0, v, dt, 0.015, 0.07));
      M.lanePeaks[i][j] = Math.max(cur, (M.lanePeaks[i][j] ?? 0) - dt * 1.2);
    }),
  );

  // The overall note (from the pitched stems when separated: drums smear every
  // pitch class) tints the whole UI; each stem lane follows its own note
  const cf = stemsLive() ? M.stemFrame : f;
  trackNote(M, cf?.chroma, dt, DEFAULT_HUE);
  if (L.chroma && M.laneNotes.length !== L.names.length) M.laneNotes = L.names.map(() => noteTracker(M.hue));
  M.laneNotes.forEach((tr, i) => trackNote(tr, L.chroma?.[i], dt, null));

  // Beat clock
  const P = beatPeriod();
  if (P) M.phase += dt / P;
  const beat = Math.floor(M.phase);
  const frac = M.phase - beat;
  const t = now();
  const live = vizLive() || stemsLive();
  if (P && live && M.lastBeat != null && beat > M.lastBeat) {
    M.beats.push({ t, down: M.bar != null && beatInBar() === 0 });
  }
  M.lastBeat = beat;

  // Visualiser history, ~40 samples/s
  if (live && t - M.histAt >= 25) {
    M.histAt = t;
    M.hist.push({ t, b: [...M.lanes], h: M.laneNames.map((_, z) => laneHue(z)), s: M.laneNames.map((_, z) => laneSat(z)), seq: M.histSeq++ });
    // Mark a chord change once the new dominant note has held for a quarter second
    const n = M.clarity > 0.25 ? M.note : null;
    if (n !== M.noteCand) {
      M.noteCand = n;
      M.noteSince = t;
    } else if (n != null && t - M.noteSince > 250 && M.notes.at(-1)?.note !== n) {
      M.notes.push({ t: M.noteSince, note: n });
    }
  }
  const cutoff = t - HIST_S * 1000;
  while (M.hist.length && M.hist[0].t < cutoff) M.hist.shift();
  while (M.beats.length && M.beats[0].t < cutoff) M.beats.shift();
  while (M.notes.length > 1 && M.notes[1].t < cutoff) M.notes.shift();
  const accent = M.bar && beatInBar() === 0 ? 1 : 0.75;
  const tick = P ? Math.exp(-frac * 6) * accent : 0;
  M.onsetEnv *= Math.exp(-dt / 0.12);
  M.pulse = Math.max(tick * 0.85, M.onsetEnv * 0.6);

  const on = P != null || live;
  const bass = M.lanes[M.laneNames.indexOf("bass")] ?? 0;
  setVar("--note-h", M.hue.toFixed(0));
  setVar("--pulse", (on ? M.pulse : 0).toFixed(2));
  setVar("--glow", (on ? 0.03 + bass * 0.08 + M.pulse * 0.05 : 0).toFixed(3));
}

function setVar(name, value) {
  if (M.cssCache[name] !== value) {
    M.cssCache[name] = value;
    document.documentElement.style.setProperty(name, value);
  }
}

// ---------- visualiser ----------

let vizCtx = null;

function drawViz() {
  const box = document.getElementById("music");
  if (box.classList.contains("idle") || !(vizLive() || stemsLive()) || !M.laneNames.length) {
    box.classList.remove("viz-on");
    return;
  }
  box.classList.add("viz-on");
  const c = document.getElementById("viz-canvas");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth;
  const h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    vizCtx = null;
  }
  if (!vizCtx) {
    vizCtx = c.getContext("2d");
    vizCtx.scale(dpr, dpr);
  }
  const ctx = vizCtx;
  ctx.clearRect(0, 0, w, h);

  // Layout: history flows left out of the live spectrum on the right. Lanes
  // top to bottom: vocals, other, bass, drums (or high, mid, bass), like a score.
  const top = 18;
  const nl = M.laneNames.length;
  const laneH = (h - top) / nl;
  const laneOf = (z) => top + z * laneH; // Top y of lane z
  const light = M.laneNames.map((n) => LANE_LIGHT[n] ?? 72);
  const hw = Math.round(w * 0.76);
  const sx0 = hw + 16;
  const sw = w - sx0;
  const t = now();
  const xAt = (ts) => hw - ((t - ts) / (HIST_S * 1000)) * hw;

  // Beats: faint lines through the lanes, stronger on the bar line
  for (const bt of M.beats) {
    const x = Math.round(xAt(bt.t)) + 0.5;
    const age = (t - bt.t) / (HIST_S * 1000);
    ctx.fillStyle = `rgba(255,255,255,${(bt.down ? 0.13 : 0.055) * (1 - age)})`;
    ctx.fillRect(x - 0.5, top, 1, h - top);
  }

  // Ribbons: thickness is the lane's energy, colour the note sounding at the time
  const hist = M.hist;
  if (hist.length > 2) {
    for (let z = 0; z < nl; z++) {
      const L = light[z];
      const cy = laneOf(z) + laneH / 2;
      const half = (v) => 0.5 + v * laneH * 0.42; // Linear in dB: equal steps look equal
      ctx.beginPath();
      ctx.moveTo(xAt(hist[0].t), cy - half(hist[0].b[z]));
      for (let i = 1; i < hist.length; i++) {
        const a = hist[i - 1];
        const b = hist[i];
        ctx.quadraticCurveTo(xAt(a.t), cy - half(a.b[z]), (xAt(a.t) + xAt(b.t)) / 2, cy - (half(a.b[z]) + half(b.b[z])) / 2);
      }
      const lastS = hist[hist.length - 1];
      ctx.lineTo(xAt(lastS.t), cy - half(lastS.b[z]));
      ctx.lineTo(xAt(lastS.t), cy + half(lastS.b[z]));
      for (let i = hist.length - 2; i >= 0; i--) {
        const a = hist[i + 1];
        const b = hist[i];
        ctx.quadraticCurveTo(xAt(a.t), cy + half(a.b[z]), (xAt(a.t) + xAt(b.t)) / 2, cy + (half(a.b[z]) + half(b.b[z])) / 2);
      }
      ctx.closePath();
      const g = ctx.createLinearGradient(0, 0, hw, 0);
      // Colour stops ride with their samples (every 10th, by sequence number),
      // so the colours scroll smoothly instead of re-sampling each frame
      for (const sm of hist) {
        if (sm.seq % 10 !== 0) continue;
        const pos = clamp(xAt(sm.t) / hw, 0, 1);
        g.addColorStop(pos, `hsla(${sm.h[z].toFixed(1)}, ${sm.s[z].toFixed(0)}%, ${L}%, ${0.04 + 0.36 * pos * pos})`);
      }
      g.addColorStop(1, `hsla(${lastS.h[z].toFixed(1)}, ${lastS.s[z].toFixed(0)}%, ${L}%, 0.4)`);
      ctx.fillStyle = g;
      ctx.fill();
    }
  }

  // Chord changes: the new note's name, in its own colour, where it began
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "left";
  for (const n of M.notes) {
    const x = Math.max(0, xAt(n.t));
    const age = (t - n.t) / (HIST_S * 1000);
    ctx.fillStyle = `hsla(${noteHue(n.note)}, 55%, 74%, ${0.85 * (1 - Math.max(0, age))})`;
    ctx.fillText(NOTE_NAMES[n.note], x + 3, 11);
    ctx.fillRect(Math.round(x), 2, 1, 11);
  }

  // Lane names at the left, brightening with their energy
  ctx.font = "10px system-ui, sans-serif";
  for (let z = 0; z < nl; z++) {
    ctx.fillStyle = `rgba(255,255,255,${0.18 + 0.4 * M.lanes[z]})`;
    // Each stem's own note beside its name, while it has a clear one
    const ln = M.laneNotes[z];
    const noteName = ln?.note != null && ln.sat > NOTE_SAT * 0.3 ? ` · ${NOTE_NAMES[ln.note]}` : "";
    ctx.fillText(M.laneNames[z] + noteName, 0, laneOf(z) + 11);
  }

  // Now: each lane's live spectrum, frequency rising upward
  ctx.fillStyle = "rgba(255,255,255,0.07)";
  ctx.fillRect(hw + 7, top, 1, h - top);
  M.laneSpec.forEach((spec, z) => {
    const L = light[z];
    const hue = laneHue(z).toFixed(0);
    const sat = laneSat(z).toFixed(0);
    const count = spec.length;
    if (!count) return;
    const y0 = laneOf(z) + laneH; // Bottom of lane
    const pts = [];
    for (let j = 0; j < count; j++) {
      pts.push([sx0 + spec[j] * sw, y0 - ((j + 0.5) / count) * laneH, M.lanePeaks[z][j] ?? 0]);
    }
    ctx.beginPath();
    ctx.moveTo(sx0, y0);
    ctx.lineTo(pts[0][0], y0);
    for (let k = 0; k < pts.length - 1; k++) {
      const mx = (pts[k][0] + pts[k + 1][0]) / 2;
      const my = (pts[k][1] + pts[k + 1][1]) / 2;
      ctx.quadraticCurveTo(pts[k][0], pts[k][1], mx, my);
    }
    ctx.lineTo(pts[pts.length - 1][0], y0 - laneH);
    ctx.lineTo(sx0, y0 - laneH);
    ctx.closePath();
    ctx.fillStyle = `hsla(${hue}, ${sat}%, ${L}%, ${0.22 + 0.18 * M.pulse})`;
    ctx.fill();
    ctx.fillStyle = `hsla(${hue}, ${sat}%, ${L + 8}%, 0.45)`;
    for (const [, y, pk] of pts) {
      if (pk > 0.03) ctx.fillRect(sx0 + pk * sw, y - 0.75, 1.5, 1.5);
    }
  });

  // Chroma: each note in its own hue; the dominant one stands forward
  const bars = document.querySelectorAll("#chroma span");
  bars.forEach((el, pc) => {
    el.style.height = `${Math.max(6, M.chroma[pc] * 100)}%`;
    el.style.opacity = pc === M.note ? 0.95 : 0.25 + 0.35 * M.chroma[pc];
  });
  const name = document.getElementById("m-note");
  const label = M.note != null && M.clarity > 0.15 ? NOTE_NAMES[M.note] : "–";
  if (name.textContent !== label) name.textContent = label;
}

function buildChroma() {
  const box = document.getElementById("chroma");
  NOTE_NAMES.forEach((n, pc) => {
    const s = document.createElement("span");
    s.title = n;
    s.style.background = `hsl(${noteHue(pc)} 50% 68%)`;
    box.append(s);
  });
}

// ---------- footsteps on the belt ----------

const steps = [];

/** Move belt footprints with the belt and fade them; the walker's gait stamps them (spawnStep). */
function updateFootsteps(dt, beltPx) {
  for (let i = steps.length - 1; i >= 0; i--) {
    const s = steps[i];
    s.x -= beltPx;
    s.age += dt;
    if (s.age > 1.5 || s.x < -G.r) {
      s.g.remove();
      steps.splice(i, 1);
      continue;
    }
    const fade = Math.max(0, 1 - s.age / 1.5) ** 1.5;
    s.g.setAttribute("transform", `translate(${s.x.toFixed(1)} ${-G.r - 4.5})`);
    s.sole.setAttribute("opacity", (fade * (s.lead ? 0.9 : 0.6)).toFixed(3));
    const t = Math.min(1, s.age / 0.45);
    attrs(s.ripple, { rx: 4 + 18 * t, ry: 1.5 + 3.5 * t, opacity: ((1 - t) * 0.45).toFixed(3) });
  }
}

function spawnStep(lead) {
  const g = document.createElementNS(SVGNS, "g");
  g.setAttribute("class", "step-mark");
  const ripple = document.createElementNS(SVGNS, "ellipse");
  const sole = document.createElementNS(SVGNS, "rect");
  attrs(sole, { x: -9, y: -1.5, width: 18, height: 3, rx: 1.5 });
  g.append(ripple, sole);
  document.getElementById("steps").append(g);
  // Alternate feet land a little apart along the belt
  steps.push({ g, sole, ripple, lead, x: G.L * 0.6 + (lead ? 14 : -8), age: 0 });
}

document.addEventListener("DOMContentLoaded", buildChroma);
