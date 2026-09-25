"use strict";

// Workout sessions in IndexedDB, surviving treadmill counter resets.
//
// The treadmill zeroes distance/time/energy whenever it reconnects (e.g. the
// server restarts). A session is an offset plus the treadmill's current
// counters ("segment"); when the counters reset within SESSION_GAP_MS of the
// last activity, the user can fold the old totals into the offset and carry on.

const SESSION_GAP_MS = 10 * 60 * 1000;
const SAVE_EVERY_MS = 2000;
const ZERO = { d: 0, e: 0, k: 0 };

const Sess = {
  db: null,
  ready: false,
  prev: null, // Most recent stored session, at load
  cur: null, // { id, start, last, offset, seg }
  pending: null, // Session awaiting "continue or start fresh"
  savedAt: 0,
};

const add = (a, b) => ({ d: a.d + b.d, e: a.e + b.e, k: a.k + b.k });
const countersOf = (ld) => ({ d: ld.distance_m, e: ld.elapsed_time_s, k: ld.calories_kcal });
/** Counters went backwards: the treadmill started over. */
const isReset = (seg, rep) => rep.e + 3 < seg.e || rep.d + 15 < seg.d;

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("treadpal", 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      db.createObjectStore("sessions", { keyPath: "id", autoIncrement: true });
      db.createObjectStore("samples", { autoIncrement: true }).createIndex("session", "session");
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function store(name, mode, fn) {
  if (!Sess.db) return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const tx = Sess.db.transaction(name, mode);
    const req = fn(tx.objectStore(name));
    tx.oncomplete = () => resolve(req?.result ?? null);
    tx.onerror = () => reject(tx.error);
  });
}

async function initSessions() {
  try {
    Sess.db = await openDb();
    // Newest session: last key in the store
    Sess.prev = await new Promise((resolve) => {
      const req = Sess.db.transaction("sessions").objectStore("sessions").openCursor(null, "prev");
      req.onsuccess = () => resolve(req.result?.value ?? null);
      req.onerror = () => resolve(null);
    });
  } catch {
    Sess.db = null; // Private browsing etc.: sessions still work, just not across reloads
  }
  Sess.ready = true;
}

/** Session totals, or the treadmill's raw counters when there's no session (yet). */
function sessionTotals() {
  const ld = S.status?.last_data;
  const rep = ld ? countersOf(ld) : ZERO;
  return Sess.cur ? add(Sess.cur.offset, rep) : rep;
}

function startSession(rep, { fromNow = false } = {}) {
  const t = Date.now();
  // fromNow: count from this moment even though the treadmill's counters aren't zero
  Sess.cur = { start: t, last: t, offset: fromNow ? { d: -rep.d, e: -rep.e, k: -rep.k } : { ...ZERO }, seg: rep };
  Sess.savedAt = 0;
  return saveSession();
}

async function saveSession() {
  const s = Sess.cur;
  if (!s) return;
  const id = await store("sessions", "readwrite", (st) => st.put(s)).catch(() => null);
  if (s.id == null && id != null) s.id = id;
}

/** Called on every status poll. */
function updateSession() {
  if (!Sess.ready || !connected() || Sess.pending) return;
  const ld = S.status.last_data;
  if (!ld) return;
  const rep = countersOf(ld);
  const active = rep.e > 0 || rep.d > 0;

  if (!Sess.cur) {
    const prev = Sess.prev;
    Sess.prev = null;
    const recent = prev && Date.now() - prev.last < SESSION_GAP_MS;
    if (recent && !isReset(prev.seg, rep)) {
      Sess.cur = prev; // Same treadmill run, the page was just reloaded
    } else if (recent) {
      askResume(prev);
      return;
    } else if (active) {
      startSession(rep);
    } else {
      return; // Idle at zero: nothing to record yet
    }
  }

  const s = Sess.cur;
  if (isReset(s.seg, rep)) {
    // The treadmill started over under us (server restart, reconnect)
    Sess.cur = null;
    askResume(s);
    return;
  }
  s.seg = rep;
  if (active) s.last = Date.now();

  if (Date.now() - Sess.savedAt >= SAVE_EVERY_MS && s.id != null) {
    Sess.savedAt = Date.now();
    saveSession();
    const tot = sessionTotals();
    store("samples", "readwrite", (st) => st.add({
      session: s.id,
      t: Date.now(),
      distance_m: tot.d,
      elapsed_s: tot.e,
      kcal: tot.k,
      speed_kmh: ld.speed_kmh,
      incline_pct: ld.incline_pct,
      target_speed_kmh: S.status.target_speed_kmh,
      target_incline_pct: S.status.target_incline_pct,
      heart_rate: ld.heart_rate_bpm,
      bpm: musicLive() ? S.music?.detected_bpm ?? null : null,
    })).catch(() => {});
  }
  renderSessionLine();
}

function askResume(prev) {
  Sess.pending = prev;
  const tot = add(prev.offset, prev.seg);
  const ago = Math.max(1, Math.round((Date.now() - prev.last) / 60000));
  $("#resume-detail").textContent =
    `${fmtDistance(tot.d)} · ${fmtDuration(tot.e)} · ${tot.k} kcal · ${ago} min ago`;
  $("#resume").hidden = false;
  renderSessionLine();
}

function resolveResume(resume) {
  const prev = Sess.pending;
  Sess.pending = null;
  $("#resume").hidden = true;
  const ld = S.status?.last_data;
  const rep = ld ? countersOf(ld) : ZERO;
  if (resume && prev) {
    // Fold everything so far into the offset; the treadmill's new counters add on top
    prev.offset = add(prev.offset, prev.seg);
    prev.seg = rep;
    prev.last = Date.now();
    Sess.cur = prev;
    saveSession();
  } else {
    startSession(rep);
  }
  renderStats();
  renderSessionLine();
}

function renderSessionLine() {
  const el = $("#session-text");
  if (!Sess.cur) {
    el.textContent = "";
    return;
  }
  const t = new Date(Sess.cur.start);
  const text = `Session since ${t.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
  if (el.textContent !== text) el.textContent = text;
}

function bindSessions() {
  $("#resume-yes").addEventListener("click", () => resolveResume(true));
  $("#resume-no").addEventListener("click", () => resolveResume(false));
  $("#session-new").addEventListener("click", () => {
    const ld = S.status?.last_data;
    if (!ld) return;
    startSession(countersOf(ld), { fromNow: true }).then(() => {
      renderStats();
      renderSessionLine();
      toast("New session started");
    });
  });
}
