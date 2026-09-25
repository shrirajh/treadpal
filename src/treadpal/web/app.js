"use strict";

// ---------- constants ----------

const KMH_PER_MPH = 1.60934;
// Stage geometry (SVG units). Deck pivots on the rear roller at (px, py).
const G = { px: 140, py: 340, L: 560, r: 16, R: 606, floorY: 400 };
const ROLLER_R = 11;
const PX_PER_M = 90;          // belt animation scale: calm, not literal
const PX_PER_KMH = 60;        // belt drag sensitivity
const MAX_VIS_DEG = 14;       // steepest the drawing gets, whatever the treadmill's range
const HOLD_LOCAL_MS = 3000;   // trust our own command over the polled target for this long
const SMOOTH_S = 0.45;        // display easing toward reported values
const DEFAULT_SPEED_RANGE = { min: 0.8, max: 12, step: 0.1 };
const DEFAULT_INCLINE_RANGE = { min: 0, max: 15, step: 0.5 };
const DEFAULT_PRESETS = [
  { name: "Stroll", speed: 3.2, incline: 0 },
  { name: "Walk", speed: 4.5, incline: 0 },
  { name: "Brisk", speed: 5.6, incline: 1 },
  { name: "Hill", speed: 4.8, incline: 5 },
  { name: "Climb", speed: 4.0, incline: 10 },
];

const $ = (sel) => document.querySelector(sel);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const now = () => performance.now();

// ---------- preferences ----------

const prefs = Object.assign(
  { unit: null, slowKmh: 3.2, presets: null, beatOffsetMs: 0 },
  JSON.parse(localStorage.getItem("treadpal.prefs") || "{}"),
);
const savePrefs = () => localStorage.setItem("treadpal.prefs", JSON.stringify(prefs));
const presets = () => prefs.presets ?? DEFAULT_PRESETS;

// ---------- state ----------

const S = {
  status: null,
  music: null,
  serverUp: true,
  disp: { speed: 0, incline: 0, belt: 0, roller: 0 },
  local: { speed: null, speedAt: -Infinity, incline: null, inclineAt: -Infinity },
  drag: null, // { kind: "speed" | "incline", value, ... }
  bandEditUntil: 0,
};

// ---------- units & ranges ----------

const unit = () => prefs.unit ?? (S.status?.prefers_mph ? "mph" : "kmh");
const unitLabel = () => (unit() === "mph" ? "mph" : "km/h");
const toU = (kmh) => (unit() === "mph" ? kmh / KMH_PER_MPH : kmh);
const fromU = (u) => (unit() === "mph" ? u * KMH_PER_MPH : u);
// Formatters never print NaN: a missing or bad value shows as a dash
const fin = (fn) => (x, ...r) => (Number.isFinite(x) ? fn(x, ...r) : "–");
const fmtSpeed = fin((kmh) => toU(kmh).toFixed(1));
const fmtIncline = fin((p) => (Math.abs(p) < 0.05 ? 0 : p).toFixed(1));
const fmtPct = fin((p) => `${+p.toFixed(1)}%`);

const speedRange = () => S.status?.speed_range ?? DEFAULT_SPEED_RANGE;
const inclineRange = () => S.status?.incline_range ?? DEFAULT_INCLINE_RANGE;
const inclineStep = () => Math.max(inclineRange().step || 0.5, 0.5);

const speedResolution = () => S.status?.speed_resolution_kmh ?? 0.01;

/**
 * Clamp to the treadmill's range, snap to 0.1 in the display unit, then to a
 * speed the treadmill can actually run (e.g. 0.1 mph steps shown in km/h).
 */
function snapSpeed(kmh) {
  const r = speedRange();
  const lo = Math.ceil(toU(r.min) * 10 - 1e-6) / 10;
  const hi = Math.floor(toU(r.max) * 10 + 1e-6) / 10;
  const v = fromU(clamp(Math.round(toU(kmh) * 10) / 10, lo, hi));
  const res = speedResolution();
  return clamp(Math.round(v / res) * res, r.min, r.max);
}

function snapIncline(p) {
  const r = inclineRange();
  const s = inclineStep();
  return clamp(Math.round(p / s) * s, r.min, r.max);
}

// ---------- actual vs target ----------

const connected = () => S.serverUp && !!S.status?.connected;
const machineState = () => S.status?.machine_state ?? null;
const actualSpeed = () => S.status?.last_data?.speed_kmh ?? 0;
const actualIncline = () => S.status?.last_data?.incline_pct ?? 0;

function targetSpeed() {
  if (S.drag?.kind === "speed") return S.drag.value;
  if (S.local.speed != null && now() - S.local.speedAt < HOLD_LOCAL_MS) return S.local.speed;
  return S.status?.target_speed_kmh ?? null;
}

function targetIncline() {
  if (S.drag?.kind === "incline") return S.drag.value;
  if (S.local.incline != null && now() - S.local.inclineAt < HOLD_LOCAL_MS) return S.local.incline;
  return S.status?.target_incline_pct ?? null;
}

// ---------- API ----------

async function api(path, body, method = "POST") {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const detail = (await res.json()).detail;
      if (detail) msg = typeof detail === "string" ? detail : "Invalid value";
    } catch { /* keep statusText */ }
    throw new Error(msg);
  }
  return res.json();
}

const timers = {};

/** Coalesce rapid changes (e.g. repeated + taps) into one command. */
function queue(key, fn, delay) {
  clearTimeout(timers[key]);
  timers[key] = setTimeout(async () => {
    S.local[key + "At"] = now();
    try {
      await fn();
    } catch (e) {
      S.local[key] = null;
      toast(e.message, { error: true });
    }
  }, delay);
}

function setSpeed(kmh, delay = 180) {
  const v = snapSpeed(kmh);
  S.local.speed = v;
  S.local.speedAt = now();
  takeOverFromMusic();
  queue("speed", () => api("/api/control/set_speed", { value: +v.toFixed(3) }), delay);
  renderPresetState();
}

function setIncline(p, delay = 180) {
  const v = snapIncline(p);
  S.local.incline = v;
  S.local.inclineAt = now();
  queue("incline", () => api("/api/control/set_incline", { value: +v.toFixed(1) }), delay);
  renderPresetState();
}

function nudgeSpeed(dir) {
  const base = targetSpeed() ?? actualSpeed();
  // At least one treadmill step, so every press changes something
  setSpeed(base + dir * Math.max(fromU(0.1), speedResolution()));
}

function nudgeIncline(dir) {
  const base = targetIncline() ?? actualIncline();
  setIncline(base + dir * inclineStep());
}

function slowDown() {
  const base = targetSpeed() ?? actualSpeed();
  if (base <= prefs.slowKmh + 0.01) {
    toast("Already at an easy pace");
    return;
  }
  setSpeed(prefs.slowKmh, 0);
}

const goFlat = () => setIncline(0, 0);

async function lifecycle(act) {
  try {
    await api(`/api/control/${act}`);
    if (S.status) S.status.machine_state = { start: "running", pause: "paused", stop: "stopped" }[act];
    if (act === "stop") S.local.speed = null;
    renderLifecycle();
  } catch (e) {
    toast(e.message, { error: true });
  }
}

// ---------- music ----------

const musicLive = () => {
  const m = S.music;
  return !!m && (m.active || (m.age_s != null && m.age_s < 20));
};
const musicSteering = () => musicLive() && !S.music.paused;

/** Manual speed input wins over the music; say so and offer the way back. */
function takeOverFromMusic() {
  if (!musicSteering()) return;
  setFollow(false, { quiet: true });
  toast("Music paused — you're steering", { action: "Resume", onAction: () => setFollow(true) });
}

async function setFollow(on, { quiet = false } = {}) {
  if (S.music) S.music.paused = !on;
  if (on) S.local.speed = null; // let the music's target show through
  renderMusic();
  try {
    await api(on ? "/api/bpm/resume" : "/api/bpm/pause");
  } catch (e) {
    if (!quiet) toast(e.message, { error: true });
  }
}

async function shiftHarmonic(path) {
  try {
    await api(`/api/bpm/harmonic/${path}`);
    pollMusicOnce();
  } catch (e) {
    toast(e.message, { error: true });
  }
}

const HARMONIC_LABELS = { 0.25: "¼×", 0.5: "½×", 0.75: "¾×", 1: "1×", 1.5: "1½×", 2: "2×", 3: "3×", 4: "4×" };
const fmtHarmonic = (h) => HARMONIC_LABELS[h] ?? `${h}×`;

// ---------- stage geometry ----------

let exag = 1.5;
const visRad = (pct) => Math.atan(pct / 100) * exag;
const pctFromRad = (a) => Math.tan(a / exag) * 100;
const polar = (a, rad) => [G.px + rad * Math.cos(a), G.py - rad * Math.sin(a)];
const toWorld = (x, y, a) => [
  G.px + x * Math.cos(a) + y * Math.sin(a),
  G.py - x * Math.sin(a) + y * Math.cos(a),
];
const attrs = (el, o) => { for (const k in o) el.setAttribute(k, o[k]); };

function buildStatic() {
  const { L, r, px, floorY } = G;
  const loop = `M 0 ${-r} L ${L} ${-r} A ${r} ${r} 0 0 1 ${L} ${r} L 0 ${r} A ${r} ${r} 0 0 1 0 ${-r} Z`;
  for (const id of ["#belt-base", "#belt-slats", "#ghost-belt"]) $(id).setAttribute("d", loop);
  attrs($("#ghost-axis"), { x1: L + r, y1: 0, x2: G.R - 12, y2: 0 });
  attrs($("#deck-frame"), { x: -r - 8, y: r + 5, width: L + 2 * r + 16, height: 14 });
  $("#upright").setAttribute("d", `M ${L - 26} ${r + 8} L ${L - 92} -188`);
  attrs($("#handlebar"), { x1: L - 76, y1: -144, x2: L - 206, y2: -140 });
  $("#console").setAttribute("transform", `rotate(-14 ${L - 112} -198)`);
  attrs($("#console-body"), { x: L - 172, y: -214, width: 120, height: 32 });
  attrs($("#console-screen"), { x: L - 164, y: -207, width: 104, height: 18 });
  attrs($("#console-text"), { x: L - 112, y: -194, "text-anchor": "middle" });
  $("#roller-front").setAttribute("transform", `translate(${L} 0)`);
  attrs($("#belt-grab"), { x: -r, y: -r - 30, width: L + 2 * r, height: 2 * r + 38 });
  attrs($("#hint-belt"), { x: L / 2, y: -r - 16, "text-anchor": "middle" });
  $("#rear-foot").setAttribute(
    "d",
    `M ${px - 12} ${G.py + r + 14} L ${px - 22} ${floorY} L ${px + 22} ${floorY} L ${px + 12} ${G.py + r + 14} Z`,
  );
  attrs($(".floor"), { y1: floorY, y2: floorY });
  attrs($("#leg-sleeve"), { x: px + L - 60 - 7, y: floorY - 22 });
}

let scaleKey = "";

function buildScale() {
  const r = inclineRange();
  const key = `${r.min}:${r.max}`;
  if (key === scaleKey) return;
  scaleKey = key;

  const maxAbs = Math.max(Math.abs(r.min), Math.abs(r.max), 1);
  exag = Math.min(1.6, MAX_VIS_DEG / ((Math.atan(maxAbs / 100) * 180) / Math.PI));

  const g = $("#scale");
  g.replaceChildren();
  const ns = "http://www.w3.org/2000/svg";
  const lo = Math.ceil(r.min);
  const hi = Math.floor(r.max);
  const major = hi - lo > 20 ? 10 : 5;
  for (let p = lo; p <= hi; p++) {
    const a = visRad(p);
    const isMajor = p % major === 0;
    const [x1, y1] = polar(a, G.R + 22);
    const [x2, y2] = polar(a, G.R + (isMajor ? 34 : 28));
    const tick = document.createElementNS(ns, "line");
    attrs(tick, { x1, y1, x2, y2, class: `scale-tick${isMajor ? " major" : ""}` });
    g.append(tick);
    if (isMajor || p === hi) {
      const [tx, ty] = polar(a, G.R + 44);
      const label = document.createElementNS(ns, "text");
      attrs(label, { x: tx, y: ty, class: "scale-label", "dominant-baseline": "middle" });
      label.textContent = p === hi ? `${p}%` : `${p}`;
      g.append(label);
    }
  }
}

// ---------- rendering ----------

const textCache = new Map();
function setText(el, text) {
  if (textCache.get(el) !== text) {
    textCache.set(el, text);
    el.textContent = text;
  }
}

const els = {};
function cacheEls() {
  for (const id of [
    "deck", "belt-slats", "roller-rear", "roller-front", "leg-rod", "arc-actual", "ghost",
    "handle", "target-notch", "hint-tilt", "console-text", "speed-num", "incline-num",
    "speed-target", "incline-target", "speed-track", "incline-track", "stage-wrap",
  ]) els[id] = document.getElementById(id);
}

function renderStage() {
  const { L, r } = G;
  const aA = visRad(S.disp.incline);
  const deg = (aA * 180) / Math.PI;
  els.deck.setAttribute("transform", `translate(${G.px} ${G.py}) rotate(${-deg})`);

  els["belt-slats"].style.strokeDashoffset = S.disp.belt;
  const rot = (S.disp.roller * 180) / Math.PI;
  els["roller-rear"].setAttribute("transform", `rotate(${rot})`);
  els["roller-front"].setAttribute("transform", `translate(${L} 0) rotate(${rot})`);

  const [lx, ly] = toWorld(L - 60, r + 19, aA);
  attrs(els["leg-rod"], { x1: G.px + L - 60, y1: G.floorY - 4, x2: lx, y2: ly });

  const ra = G.R + 14;
  if (Math.abs(aA) > 1e-4) {
    const [x0, y0] = polar(0, ra);
    const [x1, y1] = polar(aA, ra);
    els["arc-actual"].setAttribute("d", `M ${x0} ${y0} A ${ra} ${ra} 0 0 ${aA > 0 ? 0 : 1} ${x1} ${y1}`);
  } else {
    els["arc-actual"].setAttribute("d", "");
  }

  const t = targetIncline();
  const aT = visRad(t ?? S.disp.incline);
  const degT = (aT * 180) / Math.PI;
  const apart = t != null && Math.abs(t - S.disp.incline) > 0.15;
  els.ghost.setAttribute("transform", `translate(${G.px} ${G.py}) rotate(${-degT})`);
  els.ghost.classList.toggle("on", apart || S.drag?.kind === "incline");
  const [hx, hy] = polar(aT, G.R);
  els.handle.setAttribute("transform", `translate(${hx} ${hy})`);
  const [nx, ny] = polar(aT, ra + 1);
  els["target-notch"].setAttribute("transform", `translate(${nx} ${ny}) rotate(${-degT})`);
  els["target-notch"].classList.toggle("on", t != null);
  attrs(els["hint-tilt"], { x: hx + 6, y: hy + 34 });

  setText(els["console-text"], `${fmtSpeed(S.disp.speed)} ${unitLabel()}   ${fmtIncline(S.disp.incline)}%`);
}

function renderReadouts() {
  setText(els["speed-num"], fmtSpeed(S.disp.speed));
  setText(els["incline-num"], fmtIncline(S.disp.incline));
  // Treadmills may report speed truncated to 0.1, so allow just under one step
  renderTarget("speed", targetSpeed(), actualSpeed(), 0.11, fmtSpeed);
  renderTarget("incline", targetIncline(), actualIncline(), 0.15, (p) => `${fmtIncline(p)}%`);
  renderTrack("speed", speedRange(), S.disp.speed, targetSpeed());
  renderTrack("incline", inclineRange(), S.disp.incline, targetIncline());
}

function renderTarget(kind, t, a, tol, fmt) {
  const el = els[`${kind}-target`];
  const ms = machineState();
  let cls = "";
  let text = "—";
  if (kind === "speed" && ms === "stopped" && S.drag?.kind !== "speed") {
    text = "stopped";
  } else if (t == null) {
    text = "—";
  } else if (kind === "speed" && ms === "paused") {
    cls = "moving";
    text = `resumes at ${fmt(t)}`;
  } else if (Math.abs(t - a) > tol || S.drag?.kind === kind) {
    cls = "moving";
    text = `target ${t > a ? "↑" : "↓"} ${fmt(t)}`;
  } else {
    cls = "steady";
    text = "on target";
  }
  if (el.dataset.cls !== cls) {
    el.dataset.cls = cls;
    el.className = `r-target ${cls}`;
  }
  setText(el.querySelector(".t-text"), text);
}

function renderTrack(kind, r, value, target) {
  const el = els[`${kind}-track`];
  const frac = (v) => clamp((v - r.min) / (r.max - r.min), 0, 1) * 100;
  el.querySelector(".t-fill").style.width = `${frac(value)}%`;
  const tick = el.querySelector(".t-tick");
  tick.classList.toggle("on", target != null);
  if (target != null) tick.style.left = `${frac(target)}%`;
  if (kind === "speed") {
    const band = el.querySelector(".t-band");
    const m = S.music;
    band.classList.toggle("on", musicLive());
    if (m) {
      band.style.left = `${frac(m.min_speed_kmh)}%`;
      band.style.width = `${frac(m.max_speed_kmh) - frac(m.min_speed_kmh)}%`;
    }
  }
}

function renderLifecycle() {
  const ms = machineState();
  for (const b of document.querySelectorAll("#lifecycle button")) {
    b.classList.toggle("on", b.dataset.state === ms);
  }
}

function renderUnits() {
  for (const el of document.querySelectorAll(".u-speed")) el.textContent = unitLabel();
  for (const b of document.querySelectorAll("#unit-toggle button")) b.classList.toggle("on", b.dataset.unit === unit());
  const r = speedRange();
  $("#speed-min").textContent = toU(r.min).toFixed(1);
  $("#speed-max").textContent = toU(r.max).toFixed(1);
  const ir = inclineRange();
  $("#incline-min").textContent = `${ir.min}%`;
  $("#incline-max").textContent = `${ir.max}%`;
  $("#slow-sub").textContent = `to ${fmtSpeed(prefs.slowKmh)} ${unitLabel()}`;
  $("#slow-pace").value = toU(prefs.slowKmh).toFixed(1);
  $("#st-dist-u").textContent = unit() === "mph" ? "mi" : "km";
  $("#st-pace-u").textContent = unit() === "mph" ? "/mi" : "/km";
  renderPresets();
  renderMusic();
}

function renderPresets() {
  const box = $("#presets");
  box.replaceChildren();
  presets().forEach((p, i) => {
    const b = document.createElement("button");
    b.className = "preset";
    b.dataset.index = i;
    const values = `${fmtSpeed(p.speed)} ${unitLabel()} · ${fmtPct(p.incline)}`;
    b.innerHTML = p.name
      ? `<b></b><small>${values}</small>`
      : `<b>${fmtSpeed(p.speed)} ${unitLabel()}</b><small>${fmtPct(p.incline)} incline</small>`;
    if (p.name) b.querySelector("b").textContent = p.name;
    b.addEventListener("click", () => {
      setSpeed(p.speed, 0);
      setIncline(p.incline, 0);
    });
    const x = document.createElement("span");
    x.className = "x";
    x.textContent = "×";
    x.title = "Remove preset";
    x.addEventListener("click", (e) => {
      e.stopPropagation();
      prefs.presets = presets().filter((_, j) => j !== i);
      savePrefs();
      renderPresets();
    });
    b.append(x);
    box.append(b);
  });
  const add = document.createElement("button");
  add.className = "preset add";
  add.title = "Save current speed and incline as a preset";
  add.textContent = "+";
  add.addEventListener("click", () => {
    const speed = targetSpeed() ?? actualSpeed();
    const incline = targetIncline() ?? actualIncline();
    if (speed <= 0) {
      toast("Set a speed first, then save it");
      return;
    }
    prefs.presets = [...presets(), { name: null, speed: snapSpeed(speed), incline: snapIncline(incline) }];
    savePrefs();
    renderPresets();
    toast("Preset saved");
  });
  box.append(add);
  renderPresetState();
}

function renderPresetState() {
  const ts = targetSpeed();
  const ti = targetIncline();
  for (const b of document.querySelectorAll(".preset[data-index]")) {
    const p = presets()[+b.dataset.index];
    const on = ts != null && ti != null
      && Math.abs(toU(ts) - toU(p.speed)) < 0.05 && Math.abs(ti - p.incline) < 0.05;
    b.classList.toggle("on", on);
  }
}

function fmtDuration(s) {
  if (!Number.isFinite(s)) return "–";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

function renderStats() {
  const d = S.status?.last_data;
  const tot = sessionTotals();
  $("#st-time").textContent = fmtDuration(Math.max(0, tot.e));
  const dist = Math.max(0, tot.d) / (unit() === "mph" ? 1609.34 : 1000);
  $("#st-dist").textContent = dist.toFixed(2);
  const u = toU(d?.speed_kmh ?? 0);
  if (u >= 0.5) {
    const pace = 60 / u;
    $("#st-pace").textContent = `${Math.floor(pace)}:${String(Math.round((pace % 1) * 60) % 60).padStart(2, "0")}`;
  } else {
    $("#st-pace").textContent = "–";
  }
  $("#st-kcal").textContent = Math.max(0, tot.k);
  const hr = d?.heart_rate_bpm;
  $("#st-hr-wrap").hidden = hr == null;
  if (hr != null) $("#st-hr").textContent = hr;
}

function renderConnection() {
  document.body.classList.toggle("offline", !connected());
  document.body.classList.toggle("server-down", !S.serverUp);
  document.body.classList.toggle("estimated", !!S.status?.motion_estimated);
  let text = "Searching for treadmill…";
  if (!S.serverUp) text = "Server offline";
  else if (S.status?.connected) text = S.status.device_name ?? "Treadmill";
  $("#conn-text").textContent = text;
  $("#overlay-text").textContent = S.serverUp ? "Looking for your treadmill…" : "Can't reach the TreadPal server";
}

function renderMusic() {
  const m = S.music;
  const live = musicLive() || vizLive();
  const box = $("#music");
  box.classList.toggle("idle", !live);
  box.classList.toggle("live", live);
  box.classList.toggle("paused", !!m?.paused);
  $("#music-tag").classList.toggle("on", musicSteering());

  let state = "no source";
  if (live) {
    state = m?.paused ? "paused" : !musicPlaying() ? "quiet" : m?.detected_bpm ? "steering" : "listening";
  }
  $("#m-state").textContent = state;
  $("#m-follow").checked = !m?.paused;
  if (!m) return;

  const bpm = m.detected_bpm;
  $("#m-bpm").textContent = bpm ? Math.round(bpm) : "–";
  $("#m-h").textContent = m.selected_harmonic ? fmtHarmonic(m.selected_harmonic) : "–";
  $("#m-cad").textContent = m.effective_cadence ? `${Math.round(m.effective_cadence)} steps/min` : "";
  $("#m-auto").hidden = !m.harmonic_override;
  $("#m-want").textContent = m.commanded_speed_kmh ? fmtSpeed(m.commanded_speed_kmh) : "–";

  if (now() > S.bandEditUntil) {
    const r = speedRange();
    for (const input of [$("#m-band-lo"), $("#m-band-hi")]) {
      attrs(input, { min: toU(r.min).toFixed(1), max: toU(r.max).toFixed(1), step: 0.1 });
    }
    $("#m-band-lo").value = toU(m.min_speed_kmh).toFixed(1);
    $("#m-band-hi").value = toU(m.max_speed_kmh).toFixed(1);
  }
  renderBand();
}

function renderBand() {
  const lo = $("#m-band-lo");
  const hi = $("#m-band-hi");
  const min = +lo.min;
  const span = +lo.max - min || 1;
  const fill = $("#m-band .dual-fill");
  fill.style.left = `${((+lo.value - min) / span) * 100}%`;
  fill.style.width = `${((+hi.value - +lo.value) / span) * 100}%`;
  $("#m-band-text").textContent = `${(+lo.value).toFixed(1)} – ${(+hi.value).toFixed(1)} ${unitLabel()}`;
}

// ---------- animation loop ----------

let lastT = now();
function frame(t) {
  const dt = Math.min(0.1, (t - lastT) / 1000);
  lastT = t;
  const k = 1 - Math.exp(-dt / SMOOTH_S);
  S.disp.speed += (actualSpeed() - S.disp.speed) * k;
  S.disp.incline += (actualIncline() - S.disp.incline) * k;
  const beltPx = (S.disp.speed / 3.6) * PX_PER_M * dt;
  advanceBelt(beltPx);
  updateMusic(dt);
  renderStage();
  updateFootsteps(dt, beltPx);
  renderReadouts();
  drawViz();
  updateJourney(dt);
  requestAnimationFrame(frame);
}

/** Move the belt surface toward the rear by `px`; rollers turn with it. */
function advanceBelt(px) {
  S.disp.belt = (S.disp.belt + px) % 1200;
  S.disp.roller = (S.disp.roller - px / ROLLER_R) % (2 * Math.PI);
}

// ---------- polling ----------

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    S.status = await res.json();
    S.serverUp = true;
  } catch {
    S.serverUp = false;
  }
  buildScale();
  renderConnection();
  renderLifecycle();
  updateSession();
  renderStats();
  renderPresetState();
  // Unit preference and ranges come from the server; relabel only when they change
  const key = `${unit()}|${JSON.stringify(speedRange())}|${JSON.stringify(inclineRange())}`;
  if (key !== unitsKey) {
    unitsKey = key;
    renderUnits();
  }
  setTimeout(pollStatus, 500);
}
let unitsKey = "";

async function pollMusicOnce() {
  try {
    S.music = await (await fetch("/api/bpm/status")).json();
  } catch { /* server down; status poll reports it */ }
  renderMusic();
}

async function pollMusic() {
  await pollMusicOnce();
  setTimeout(pollMusic, 1000);
}

// ---------- interaction ----------

function svgPoint(e) {
  const svg = $("#stage");
  const p = svg.createSVGPoint();
  p.x = e.clientX;
  p.y = e.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());
}

const angleOf = (p) => Math.atan2(G.py - p.y, p.x - G.px);

function beginDrag(e, drag) {
  if (!connected()) return;
  e.preventDefault();
  S.drag = drag;
  $("#stage").setPointerCapture(e.pointerId);
  els["stage-wrap"].classList.add(`dragging-${drag.kind}`);
}

function commitDrag() {
  const d = S.drag;
  if (!d) return;
  S.drag = null;
  els["stage-wrap"].classList.remove("dragging-speed", "dragging-incline");
  if (d.kind === "speed") setSpeed(d.value, 0);
  else setIncline(d.value, 0);
}

function bindStage() {
  const svg = $("#stage");
  const wrap = els["stage-wrap"];

  const belt = $("#belt-grab");
  belt.addEventListener("pointerenter", () => wrap.classList.add("hover-belt"));
  belt.addEventListener("pointerleave", () => wrap.classList.remove("hover-belt"));
  belt.addEventListener("pointerdown", (e) => {
    const p = svgPoint(e);
    const start = targetSpeed() ?? actualSpeed();
    beginDrag(e, { kind: "speed", start, value: start, x0: p.x, y0: p.y, along: 0 });
  });

  for (const el of document.querySelectorAll(".grab-tilt")) {
    el.addEventListener("pointerenter", () => wrap.classList.add("hover-tilt"));
    el.addEventListener("pointerleave", () => wrap.classList.remove("hover-tilt"));
    el.addEventListener("pointerdown", (e) => {
      const start = targetIncline() ?? actualIncline();
      beginDrag(e, { kind: "incline", value: start, a0: angleOf(svgPoint(e)), start: visRad(start) });
    });
  }

  svg.addEventListener("pointermove", (e) => {
    const d = S.drag;
    if (!d || d.track) return;
    const p = svgPoint(e);
    if (d.kind === "speed") {
      // Distance along the deck; pulling the belt rearward (left) speeds it up
      const a = visRad(S.disp.incline);
      const along = (p.x - d.x0) * Math.cos(a) - (p.y - d.y0) * Math.sin(a);
      advanceBelt(-(along - d.along));
      d.along = along;
      d.value = snapSpeed(d.start - along / PX_PER_KMH);
    } else {
      const a = clamp(d.start + angleOf(p) - d.a0, -1.2, 1.2);
      d.value = snapIncline(pctFromRad(a));
    }
  });
  svg.addEventListener("pointerup", commitDrag);
  svg.addEventListener("pointercancel", commitDrag);

  // Fine adjustment with the wheel over the belt or the tilt handle
  belt.addEventListener("wheel", (e) => { e.preventDefault(); nudgeSpeed(e.deltaY < 0 ? 1 : -1); }, { passive: false });
  $("#handle").addEventListener("wheel", (e) => { e.preventDefault(); nudgeIncline(e.deltaY < 0 ? 1 : -1); }, { passive: false });
}

function bindTrack(kind) {
  const el = els[`${kind}-track`];
  const valueAt = (e) => {
    const b = el.getBoundingClientRect();
    const r = kind === "speed" ? speedRange() : inclineRange();
    const raw = r.min + clamp((e.clientX - b.left) / b.width, 0, 1) * (r.max - r.min);
    return kind === "speed" ? snapSpeed(raw) : snapIncline(raw);
  };
  el.addEventListener("pointerdown", (e) => {
    if (!connected()) return;
    el.setPointerCapture(e.pointerId);
    S.drag = { kind, value: valueAt(e), track: true };
  });
  el.addEventListener("pointermove", (e) => {
    if (S.drag?.track && S.drag.kind === kind) S.drag.value = valueAt(e);
  });
  el.addEventListener("pointerup", commitDrag);
  el.addEventListener("pointercancel", commitDrag);
}

function beginEdit(kind) {
  const btn = $(`#${kind}-num`);
  const current = kind === "speed" ? toU(targetSpeed() ?? actualSpeed()) : targetIncline() ?? actualIncline();
  const input = document.createElement("input");
  input.type = "number";
  input.className = "num-input";
  input.step = kind === "speed" ? "0.1" : String(inclineStep());
  input.value = current.toFixed(1);
  btn.hidden = true;
  btn.after(input);
  input.focus();
  input.select();

  let done = false;
  const finish = (commit) => {
    if (done) return;
    done = true;
    const v = parseFloat(input.value);
    input.remove();
    btn.hidden = false;
    if (commit && Number.isFinite(v)) {
      if (kind === "speed") setSpeed(fromU(v), 0);
      else setIncline(v, 0);
    }
  };
  input.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") finish(true);
    if (e.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

const ACTIONS = {
  "speed-up": () => nudgeSpeed(1),
  "speed-down": () => nudgeSpeed(-1),
  "incline-up": () => nudgeIncline(1),
  "incline-down": () => nudgeIncline(-1),
  slow: slowDown,
  flat: goFlat,
  start: () => lifecycle("start"),
  pause: () => lifecycle("pause"),
  stop: () => lifecycle("stop"),
};

function bindControls() {
  for (const b of document.querySelectorAll("[data-act]")) {
    b.addEventListener("click", () => ACTIONS[b.dataset.act]());
  }

  // Hold +/- to repeat
  for (const b of document.querySelectorAll(".step")) {
    let hold;
    const stop = () => clearTimeout(hold);
    b.addEventListener("pointerdown", () => {
      const repeat = (delay) => {
        hold = setTimeout(() => { ACTIONS[b.dataset.act](); repeat(90); }, delay);
      };
      repeat(450);
    });
    for (const ev of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(ev, stop);
  }

  $("#speed-num").addEventListener("click", () => connected() && beginEdit("speed"));
  $("#incline-num").addEventListener("click", () => connected() && beginEdit("incline"));

  for (const b of document.querySelectorAll("#unit-toggle button")) {
    b.addEventListener("click", () => {
      prefs.unit = b.dataset.unit;
      savePrefs();
      renderUnits();
    });
  }

  $("#settings-btn").addEventListener("click", () => { $("#settings").hidden = !$("#settings").hidden; });
  $("#slow-pace").addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    if (Number.isFinite(v) && v > 0) {
      prefs.slowKmh = fromU(v);
      savePrefs();
    }
    renderUnits();
  });
  $("#beat-offset").value = prefs.beatOffsetMs;
  $("#beat-offset-val").textContent = `${prefs.beatOffsetMs} ms`;
  $("#beat-offset").addEventListener("input", (e) => {
    const v = +e.target.value;
    shiftBeatOffset(v - prefs.beatOffsetMs);
    prefs.beatOffsetMs = v;
    $("#beat-offset-val").textContent = `${v} ms`;
    savePrefs();
  });
  $("#reset-presets").addEventListener("click", () => {
    prefs.presets = null;
    savePrefs();
    renderPresets();
  });

  // Music
  $("#m-follow").addEventListener("change", (e) => setFollow(e.target.checked));
  $("#m-h-up").addEventListener("click", () => shiftHarmonic("up"));
  $("#m-h-down").addEventListener("click", () => shiftHarmonic("down"));
  $("#m-auto").addEventListener("click", () => shiftHarmonic("reset"));
  const lo = $("#m-band-lo");
  const hi = $("#m-band-hi");
  const onBandInput = (e) => {
    S.bandEditUntil = now() + 2000;
    if (+hi.value - +lo.value < 0.2) {
      if (e.target === lo) lo.value = (+hi.value - 0.2).toFixed(1);
      else hi.value = (+lo.value + 0.2).toFixed(1);
    }
    renderBand();
  };
  const onBandChange = async () => {
    S.bandEditUntil = now() + 2000;
    const body = { min_speed_kmh: +fromU(+lo.value).toFixed(3), max_speed_kmh: +fromU(+hi.value).toFixed(3) };
    if (S.music) Object.assign(S.music, body);
    try {
      await api("/api/bpm/config", body, "PUT");
    } catch (e) {
      toast(e.message, { error: true });
    }
  };
  for (const input of [lo, hi]) {
    input.addEventListener("input", onBandInput);
    input.addEventListener("change", onBandChange);
  }

  document.addEventListener("keydown", (e) => {
    if (e.target.closest?.("input, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    const handlers = {
      ArrowUp: () => (e.shiftKey ? nudgeIncline(1) : nudgeSpeed(1)),
      ArrowDown: () => (e.shiftKey ? nudgeIncline(-1) : nudgeSpeed(-1)),
      PageUp: () => nudgeIncline(1),
      PageDown: () => nudgeIncline(-1),
      s: slowDown,
      f: goFlat,
      " ": () => lifecycle(machineState() === "running" ? "pause" : "start"),
      Escape: () => lifecycle("stop"),
    };
    const h = handlers[e.key.length === 1 ? e.key.toLowerCase() : e.key];
    if (!h) return;
    e.preventDefault();
    // Keep Space from also "clicking" whichever button has focus
    if (document.activeElement instanceof HTMLButtonElement) document.activeElement.blur();
    h();
  });
}

// ---------- toast ----------

let toastTimer;
function toast(text, { error = false, action = null, onAction = null, ms = 3500 } = {}) {
  const el = $("#toast");
  const btn = $("#toast-action");
  $("#toast-text").textContent = text;
  el.classList.toggle("error", error);
  btn.hidden = !action;
  btn.textContent = action ?? "";
  btn.onclick = () => {
    el.classList.remove("show");
    onAction?.();
  };
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), action ? ms + 2500 : ms);
}

// ---------- boot ----------

// After music.js / journey.js have loaded
document.addEventListener("DOMContentLoaded", () => {
  cacheEls();
  buildStatic();
  buildScale();
  bindStage();
  bindTrack("speed");
  bindTrack("incline");
  bindControls();
  initJourney();
  bindSessions();
  renderUnits();
  initSessions().then(pollStatus);
  pollMusic();
  connectViz();
  requestAnimationFrame(frame);
});
