"use strict";

// Distance journey along the bottom of the screen: a pixel stick figure walks
// a ruler toward landmark distances. Legs are solved with two-bone IK so the
// planted foot moves exactly with the ground; the gait is locked to the beat
// when music plays, and the arms dance with a move picked by the dominant note.

// One landmark is drawn at random from each group per session. Groups are
// close together early on so there is always something coming up.
const MILESTONE_GROUPS = [
  [[24, "a tennis court"], [25, "a short-course pool"], [28, "a basketball court"], [30, "a blue whale, nose to tail"]],
  [[46, "the Statue of Liberty (just her)"], [50, "an Olympic pool"], [56, "the Leaning Tower of Pisa"]],
  [[71, "a Boeing 747"], [73, "an Airbus A380"]],
  [[93, "the Statue of Liberty, ground to torch"], [96, "Big Ben's tower"], [100, "a 100 m sprint"]],
  [[105, "a football pitch"], [110, "an American football field"]],
  [[135, "the London Eye"], [139, "the Great Pyramid of Giza"], [150, "a par-3 golf hole"]],
  [[169, "the Washington Monument"], [184, "the Space Needle"], [200, "half a lap of a track"]],
  [[227, "a Golden Gate Bridge tower"], [244, "Tower Bridge"], [245, "the Hindenburg"]],
  [[269, "the Titanic, bow to stern"]],
  [[319, "the Chrysler Building"], [330, "the Eiffel Tower"], [333, "an aircraft carrier"]],
  [[362, "the Symphony of the Seas"], [381, "the Empire State Building"], [400, "a lap of a running track"]],
  [[443, "the Empire State, antenna and all"], [452, "the Petronas Towers"]],
  [[508, "Taipei 101"], [541, "One World Trade Center"], [553, "the CN Tower"]],
  [[632, "the Shanghai Tower"], [634, "Tokyo Skytree"]],
  [[800, "two laps of a track"], [828, "the Burj Khalifa"]],
  [[1000, "a kilometre"]],
  [[1149, "the Sydney Harbour Bridge"], [1280, "the Golden Gate's main span"]],
  [[1500, "a 1500 m race"], [1609, "a mile"]],
  [[1810, "Edinburgh's Royal Mile"], [1834, "the Brooklyn Bridge"], [1910, "the Champs-Élysées"]],
  [[2100, "the Hollywood Walk of Fame"]],
  [[2460, "the Millau Viaduct"], [2737, "the Golden Gate Bridge"]],
  [[3218, "two miles"], [3800, "Venice's Grand Canal"]],
  [[4000, "Central Park, end to end"], [4828, "three miles"]],
  [[5000, "a parkrun"]],
  [[6800, "the Las Vegas Strip"]],
  [[9700, "the Central Park loop"], [10000, "a 10K"]],
  [[16093, "ten miles"]],
  [[21097, "a half marathon"], [21600, "Manhattan, tip to tip"]],
  [[33800, "the English Channel"]],
  [[42195, "a marathon"]],
  [[50000, "a 50K ultra"]],
  [[88000, "roughly London to Oxford"], [100000, "a 100 km ultra"]],
];

// Figure, in figure pixels (drawn at PIX screen px each)
const PIX = 2;
const FIG_W = 48;
const FIG_H = 64;
const FOOT_X = 24; // Ground contact point in the figure canvas
const FOOT_Y = 62;
const THIGH = 7;
const SHIN = 7;
const TORSO = 9;
const UPPER_ARM = 6;
const FOREARM = 6;
const ARM_SCALE = (UPPER_ARM + FOREARM) / 10; // Poses are authored for a reach of 10
const HEAD_R = 3;
const NECK = 1.5; // Gap between shoulders and head, so overhead arms have room
const HIP_STAND = THIGH + SHIN - 0.6;
const PPM_FIG = 17.5; // Figure pixels per metre (figure is ~1.75 m tall)
const PPM = PPM_FIG * PIX; // Screen px per metre on the ruler
const BAR_H = 128;
const GROUND_Y = 96;
const LABEL_Y = GROUND_Y - 80; // Milestone labels sit above the figure's head

// Dance moves. Each gives IK targets relative to the body, as functions of
// bar-aligned beat position b and the beat envelope env (1 on the beat, decaying):
//   hands: two [x, y] targets from the shoulder (+x forward, +y down; reach 10).
//     The head sits around (0, -5) with radius 3, so overhead hands go clearly
//     in front of or behind it.
//   legs (standing only): { drop, feet: two [x, y] from the hip's ground point, sway? }
//   head (optional): [dx, dy] offset; lean (optional): torso lean in radians
//   upper: the arms work on their own, so they can dance while walking or moonwalking
// The dominant note picks the move: note n gets moves n and n + 12 (when they exist).
const PLANTED = [[2, 0], [-2, 0]];
const bounce = (k) => (b, env) => ({ drop: k * env, feet: PLANTED });
const alt = (b) => (Math.floor(b) % 2 ? -1 : 1);

/** Step through key poses, `rate` per beat, snapping to each within `snap` of a step. */
function seqPose(b, poses, rate = 1, snap = 0.3) {
  const x = b * rate;
  const i = Math.floor(x);
  const n = poses.length;
  const from = poses[(((i - 1) % n) + n) % n];
  const to = poses[((i % n) + n) % n];
  const k = snapEase(x, snap);
  return to.map((pt, j) => [mix(from[j][0], pt[0], k), mix(from[j][1], pt[1], k)]);
}

/** A value that snaps between -1 and 1 on each step of `rate` per beat. */
function seqSign(b, rate = 1, snap = 0.3) {
  const x = b * rate;
  const s = Math.floor(x) % 2 ? -1 : 1;
  return mix(-s, s, snapEase(x, snap));
}

const DANCE_MOVES = [
  { name: "the groove", upper: true, // big alternating arm swings, knees and heels bouncing
    hands: (b) => seqPose(b, [[[7, 4], [-6, 5]], [[-6, 5], [7, 4]]], 1, 0.35),
    lean: (b) => 0.08 + 0.07 * seqSign(b, 1, 0.35),
    legs: (b, env) => ({ drop: 1.8 * env, feet: [[2, alt(b) > 0 ? -env : 0], [-2, alt(b) > 0 ? 0 : -env]] }),
  },
  { name: "raise the roof", upper: true, // palms push up on the beat, a hop with each
    hands: (b) => seqPose(b, [[[5.5, -8.5], [-5, -8.5]], [[5, -3], [-4.5, -3]]], 2, 0.2),
    legs: (b) => hop(b),
  },
  { name: "the wave", upper: true, // one hand high and waving, the other on the hip, swaying
    hands: (b) => [[6 + 1.8 * Math.sin(4 * Math.PI * b), -8], [-3, 4]],
    legs: (b, env) => ({ drop: env, feet: PLANTED, sway: 1.2 * Math.sin(Math.PI * b) }),
  },
  { name: "the clap", upper: true, // hands meet out in front on the beat
    hands: (b) => seqPose(b, [[[8, 0], [8, 0.5]], [[6, -4], [5, 5]]], 2, 0.15),
    lean: () => 0.1,
    legs: bounce(1.4),
  },
  { name: "Saturday Night Fever", upper: true, // point up, point down, hand on hip
    hands: (b) => seqPose(b, [[[6, -8], [-3, 3]], [[-5, 7], [-3, 3]]], 1, 0.25),
    legs: (b, env) => ({ drop: env, feet: [[2 + 2 * Math.max(0, -alt(b)), 0], [-2, 0]], sway: 0.8 * seqSign(b, 1, 0.25) }),
  },
  { name: "the robot", upper: true, // forearms tick up and down, head ticks, knees lock
    hands: (b) => { const s = seqSign(b, 1, 0.1); return [[6, 1 - 4 * s], [6, 1 + 4 * s]]; },
    head: (b) => [0.8 * seqSign(b, 1, 0.1), 0],
    legs: (b) => ({ drop: 0.4 * snapEase(b, 0.1), feet: PLANTED }),
  },
  { name: "the floss", upper: true, // arms whip front and back twice a beat, hips opposite
    hands: (b) => seqPose(b, [[[7, 6], [-7, 6]], [[-7, 6], [7, 6]]], 2, 0.35),
    legs: (b, env) => ({ drop: 0.5 * env, feet: PLANTED, sway: -1.6 * seqSign(b, 2, 0.35) }),
  },
  { name: "the dab", upper: true, // head into the elbow on the beat, reset on the next
    hands: (b) => seqPose(b, [[[7.5, -6.5], [6, -2.5]], [[1.5, 9], [-1, 9]]], 1, 0.15),
    head: (b) => (Math.floor(b) % 2 === 0 ? [2.2 * snapEase(b, 0.15), 1.8 * snapEase(b, 0.15)] : [0, 0]),
    lean: (b) => (Math.floor(b) % 2 === 0 ? 0.3 : 0),
    legs: bounce(0.8),
  },
  { name: "YMCA", upper: true, // one letter per beat: the bar spells it out
    hands: (b) => seqPose(b, [
      [[5, -8], [-4.5, -8]], // Y
      [[4, -7.5], [-3.5, -7.5]], // M (hands to the head)
      [[7, -4], [6, 3]], // C
      [[3.5, -9.3], [3.5, -9.3]], // A
    ], 1, 0.2),
    // Each letter has its own body too: rise, crouch, lean in, stretch
    lean: (b) => [0, 0, 0.3, -0.05][((Math.floor(b) % 4) + 4) % 4],
    legs: (b, env) => {
      const letter = ((Math.floor(b) % 4) + 4) % 4;
      const rise = letter === 0 || letter === 3; // Y and A: up on the toes
      return {
        drop: [-0.8, 2.5, 1, -1][letter] + 0.6 * env,
        feet: rise ? [[2, 0, 1.8], [-2, 0, 1.8]] : PLANTED,
      };
    },
  },
  { name: "the Macarena", upper: true, // one arm at a time, two moves a beat, hips to finish
    hands: (b) => seqPose(b, [
      [[8, 0], [1, 9]], [[8, 0], [8, 0.5]], [[8, 1.5], [8, 2]], [[2, -0.5], [8, 2]],
      [[2, -0.5], [2, 0]], [[-4, -6.5], [2, 0]], [[-4, -6.5], [-4, -6]], [[-3, 5], [-3, 5.5]],
    ], 2, 0.3),
    legs: (b, env) => ({ drop: 0.8 * env, feet: PLANTED, sway: (b % 4) > 3.5 ? 1.4 * Math.sin(4 * Math.PI * b) : 0 }),
  },
  { name: "Gangnam Style", upper: true, // holding the reins, galloping; a lasso every fourth beat
    hands: (b, env) => (Math.floor(b) % 4 === 3
      ? [[5.5 + 1.5 * Math.cos(4 * Math.PI * b), -8.5 + 1.2 * Math.sin(4 * Math.PI * b)], [6, 3]]
      : [[6, 2.5 + env], [5.5, 3 + env]]),
    legs: (b, env) => {
      const p = Math.sin(2 * Math.PI * b);
      return { drop: 1.2 * env, feet: [[3, -2.5 * Math.max(0, p)], [-1, -2.5 * Math.max(0, -p)]] };
    },
  },
  { name: "the running man", // high knee on every beat, arms pumping
    hands: (b) => seqPose(b, [[[7, -1], [-5, 4]], [[-5, 4], [7, -1]]], 1, 0.3),
    legs: (b) => {
      const lift = Math.sin(Math.PI * (b - Math.floor(b)));
      const up = [2 + 1.5 * lift, -5.5 * lift];
      return { drop: 0.8, feet: alt(b) > 0 ? [up, [-1, 0]] : [[1, 0], up] };
    },
  },
  { name: "Thriller", upper: true, // zombie claws, shoulders shrugging, head lolling
    hands: (b, env) => [[7.5, -3 - 1.5 * env], [6.5, -1 - 1.5 * env]],
    head: (b) => [1.2, 0.8 * Math.sin(Math.PI * b)],
    lean: () => 0.18,
    legs: (b, env) => ({ drop: 1.5 * env, feet: PLANTED, sway: 1.2 * seqSign(b, 1, 0.2) }),
  },
  { name: "the twist", // low and twisting, heels taking turns
    hands: (b) => { const p = Math.sin(2 * Math.PI * b); return [[5 * p, 3], [-5 * p, 3]]; },
    legs: (b) => {
      const p = Math.sin(2 * Math.PI * b);
      return { drop: 2.2 + 0.8 * Math.abs(p), feet: [[2, -Math.max(0, p)], [-2, -Math.max(0, -p)]], sway: p };
    },
  },
  { name: "the Carlton", upper: true, // both arms swing together, snapping, hips the other way
    hands: (b) => seqPose(b, [[[6, 5], [5, 6]], [[-6, 5], [-5, 6]]], 1, 0.25),
    head: (b) => [-0.8 * seqSign(b, 1, 0.25), 0],
    legs: (b, env) => ({ drop: 0.8 * env, feet: PLANTED, sway: -1.2 * seqSign(b, 1, 0.25) }),
  },
  { name: "the sprinkler", upper: true, // hand behind the head, the other ticks down and sweeps back
    hands: (b) => seqPose(b, [
      [[8.5, -3], [-4, -6.5]], [[8.5, -1.5], [-4, -6.5]], [[8.5, 0], [-4, -6.5]],
      [[8.5, 1.5], [-4, -6.5]], [[8.5, 3], [-4, -6.5]], [[8.5, 0], [-4, -6.5]],
    ], 2, 0.12),
    legs: bounce(0.6),
  },
  { name: "walk like an Egyptian", upper: true, // one hand forward and up, one back and down; head slides
    hands: (b) => seqPose(b, [[[6, -5], [-5, 3]], [[6, -5], [-5, 3]], [[-5, 3], [6, -5]], [[-5, 3], [6, -5]]], 1, 0.3),
    head: (b) => [1.5 * seqSign(b, 1, 0.2), 0],
    legs: bounce(0.8),
  },
  { name: "vogue", upper: true, // hands frame the face, a new frame every beat
    hands: (b) => seqPose(b, [[[4.5, -7.5], [5, -2]], [[6, -6], [-5, -6]], [[5, -1], [4.5, -8]], [[7, -4], [4, -8]]], 1, 0.1),
    head: (b) => [0.8 * seqSign(b, 1, 0.1), 0],
    legs: (b, env) => ({ drop: 0.6 * env, feet: PLANTED, sway: seqSign(b, 1, 0.1) }),
  },
  { name: "the spiral", // arabesque: free leg high behind, arms long, a gentle rise on the beat
    hands: () => [[7, -2], [-6, -1]],
    lean: () => 0.55,
    legs: (b, env) => ({ drop: 0.3 * env, feet: [[1, 0], [-12, -6]] }),
  },
  { name: "a scratch spin", // one turn every two beats, arms tucked, free foot crossed
    spin: (b) => b / 2,
    hands: () => [[3, 1], [2.5, 1.5]],
    legs: () => ({ drop: 0.3, feet: [[0.5, 0], [1.5, -4]] }),
  },
  { name: "a camel spin", // torso and free leg level, one turn a bar
    spin: (b) => b / 4,
    hands: () => [[6, 1], [-4, 2]],
    lean: () => 1.35,
    legs: () => ({ drop: 1, feet: [[0, 0], [-13.5, -11.5]] }),
  },
  { name: "the Biellmann", // free foot pulled up behind the head, one turn a bar
    spin: (b) => b / 4,
    hands: () => [[-4.5, -9], [-4, -8.5]],
    legs: () => ({ drop: 0.2, feet: [[0.5, 0], [-3, -24]] }),
  },
];

// Jumps from the walk. rot = turns in the air; half turns land facing backwards
// (as skaters do), which rolls straight into a moonwalk.
const JUMPS = [
  { name: "a jump", rot: 0 },
  { name: "a 360", rot: 1 },
  { name: "a double toe loop", rot: 2, minAir: 0.45 },
  { name: "a triple lutz", rot: 3, minAir: 0.5 },
  { name: "an axel", rot: 1.5 },
  { name: "a triple axel", rot: 3.5, minAir: 0.55 },
];

/** Moves for a note: n and n + 12, where they exist. */
const movesForNote = (n) => [n, n + 12].filter((i) => i < DANCE_MOVES.length);

/** Crouch just before the beat, spring up after it. */
function hop(b) {
  const f = b - Math.floor(b);
  const air = f < 0.45 ? 3.2 * Math.sin((Math.PI * f) / 0.45) : 0;
  const crouch = f > 0.75 ? 1.8 * smoothstep((f - 0.75) / 0.25) : f < 0.08 ? 1.8 * (1 - f / 0.08) : 0;
  return { drop: crouch - air, feet: [[2, -air], [-2, -air]] };
}

/**
 * Moonwalk feet, in the figure's own frame (it's drawn mirrored, so the ground
 * moves toward its front). The weight-bearing foot is up on its toes and fixed
 * to the floor; the flat foot slides back past it. They swap each step with a
 * heel pop, so the body glides "backwards".
 */
function moonwalkFeet(gp, stepPx) {
  const q = gp - Math.floor(gp);
  const toe = Math.floor(gp) % 2 === 0 ? 0 : 1;
  const pop = smoothstep(q / 0.15); // Heel change at the start of each step
  const feet = [];
  feet[toe] = [-stepPx / 2 + q * stepPx, 0, 2 * pop]; // Rides with the floor, heel up
  feet[1 - toe] = [stepPx / 2 - q * stepPx, 0, 2 * (1 - pop)]; // Slides back, heel settling
  return feet;
}

function mix(a, b, t) { return a + (b - a) * t; }
function smoothstep(t) { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); }
/** Reach the new pose within the first `frac` of a beat, then hold. */
function snapEase(b, frac) { return smoothstep((b - Math.floor(b)) / frac); }

const J = {
  chain: [],
  d: null, // Smoothed distance (m); the treadmill reports it in coarse steps
  res: null, // Learned distance resolution
  lastRep: null,
  passedIdx: null, // Last landmark passed (null until the first reading)
  cheer: null, // { label, m, t }
  gp: 0, // Gait phase in steps; foot strikes at integers
  maxStrike: null,
  pose: null, // Displayed pose: { drop, sway, feet, hands, head, lean }
  src: null, // What the pose comes from ("walk", "dance:3", ...); a change crossfades
  from: null, // Pose at the start of a crossfade
  blendT: 0,
  blendDur: 300,
  cad: null, // Eased cadence (steps/s)
  gcorr: 0, // Eased phase-lock correction (steps/s)
  groundV: null, // Speed the ruler actually scrolls (m/s)
  est: null, // Where the distance should be, dead-reckoned from the last counter tick
  corr: 0, // Current catch-up speed (m/s) on top of the belt speed
  trick: null, // A jump in progress: { name, rot, takeoff, land (beats), height (fig px), face0 }
  airborne: false,
  puffs: [], // Foot strikes on the ruler: { m (world metres), t, big? }
  moonUntil: null, // Beat at which the moonwalk ends
  moonCooldown: 0, // Beat before which no new moonwalk starts
  face: 1, // Facing: 1 forward, -1 backwards; the turn animates through 0
  turn: null, // { from, to, t, dur }
  move: 0,
  moveBeat: null,
  fig: null,
  figCtx: null,
  img: null,
  ctx: null,
  w: 0,
};

function initJourney() {
  let picks = null;
  try {
    picks = JSON.parse(sessionStorage.getItem("treadpal.journey"));
  } catch { /* fresh draw */ }
  if (!Array.isArray(picks) || picks.length !== MILESTONE_GROUPS.length) {
    picks = MILESTONE_GROUPS.map((g) => Math.floor(Math.random() * g.length));
    sessionStorage.setItem("treadpal.journey", JSON.stringify(picks));
  }
  J.chain = MILESTONE_GROUPS.map((g, i) => {
    const [m, label] = g[picks[i] % g.length];
    return { m, label };
  });
  J.fig = document.createElement("canvas");
  J.fig.width = FIG_W;
  J.fig.height = FIG_H;
  J.figCtx = J.fig.getContext("2d");
  J.img = J.figCtx.createImageData(FIG_W, FIG_H);
}

function fmtDistance(m) {
  if (!Number.isFinite(m)) return "–";
  if (unit() === "mph") {
    const mi = m / 1609.34;
    return mi < 0.25 ? `${Math.round(m * 1.09361)} yd` : `${mi.toFixed(mi < 10 ? 2 : 1)} mi`;
  }
  return m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(m < 10000 ? 2 : 1)} km`;
}

// ---------- distance ----------

/**
 * Smooth distance. The counter ticks (10 m on some treadmills) as the true distance
 * crosses each step, so at a tick we know exactly where we are; between ticks
 * we dead-reckon at belt speed, never past the next tick. The ruler follows
 * that at no more than ±30% of belt speed off, never backwards: no lurches.
 */
function updateDistance(dt) {
  const rep = Math.max(0, sessionTotals().d);
  const v = S.disp.speed / 3.6;
  if (J.lastRep != null && rep > J.lastRep && rep - J.lastRep <= 50) {
    J.res = Math.min(J.res ?? Infinity, rep - J.lastRep);
  }
  const res = J.res ?? 1;
  // First reading, a new session, or a resumed one: jump rather than race there
  if (J.d == null || rep < J.d - 3 * res - 20 || rep > J.d + 3 * res + 50) {
    J.d = J.est = rep;
    J.lastRep = rep;
    J.passedIdx = null;
    return;
  }
  J.est = rep !== J.lastRep ? rep : Math.min(J.est + v * dt, rep + res);
  J.lastRep = rep;
  // The correction itself eases in, so a new tick changes the ruler's speed gently
  const lim = 0.3 * v + 0.03;
  J.corr += (clamp((J.est - J.d) * 0.8, -lim, lim) - J.corr) * (1 - Math.exp(-dt / 0.4));
  const vd = Math.max(0, v + J.corr);
  J.d += vd * dt;
  J.groundV = J.groundV == null ? vd : J.groundV + (vd - J.groundV) * (1 - Math.exp(-dt / 0.15));
}

// ---------- gait ----------

/** Natural step length (m) at a speed: the same biomechanics model BPM sync uses. */
const naturalStep = (kmh) => (kmh < 7 ? 0.35 + 0.075 * kmh : 0.55 + 0.065 * kmh);

/**
 * The walker is a pacer: its cadence is what the model says you should walk at.
 * With music, that's the harmonic of the beat whose step length at the current
 * speed is most natural (what BPM sync itself picks; it's the server's own
 * choice when the music is steering). Without, the natural cadence for the speed.
 */
function pickCadence(kmh, P) {
  const v = kmh / 3.6;
  const natural = { cadence: v / naturalStep(kmh), perBeat: null };
  const musical = P != null;
  if (!musical || v <= 0) return natural;
  const serverH = M.harmonic ?? S.music?.selected_harmonic ?? 1;
  const harmonics = S.music?.harmonics?.length ? S.music.harmonics : [serverH];
  const errOf = (h) => Math.abs(v / (h / P) - naturalStep(kmh)) / naturalStep(kmh);
  // Keep the current harmonic unless another is clearly better (no flip-flopping)
  let best = J.perBeat != null && harmonics.includes(J.perBeat) ? J.perBeat : serverH;
  for (const h of harmonics) if (errOf(h) < errOf(best) - 0.08) best = h;
  // Hysteresis on locking to the beat at all
  if (errOf(best) > (J.perBeat != null ? 0.3 : 0.22)) return natural;
  return { cadence: best / P, perBeat: best };
}

function updateGait(dt) {
  const kmh = S.disp.speed;
  const walking = kmh > 0.3 && machineState() !== "paused";
  const P = beatPeriod();
  const musical = P != null;
  const { cadence, perBeat } = pickCadence(kmh, P);
  // Tempo changes ease in over ~0.4 s rather than switching leg speed instantly
  J.cad = J.cad == null ? cadence : J.cad + (cadence - J.cad) * (1 - Math.exp(-dt / 0.4));

  if (walking) {
    // Drift into phase with the beat, at most 30% faster or slower than the
    // stride, and ease that in too, so a new song never jerks the legs
    let want = 0;
    if (perBeat) {
      let err = (M.phase - M.barOffset) * perBeat - J.gp;
      err -= Math.round(err);
      want = clamp(err * 2.5, -0.3 * J.cad, 0.3 * J.cad);
    }
    J.gcorr += (want - J.gcorr) * (1 - Math.exp(-dt / 0.3));
    J.gp += (J.cad + J.gcorr) * dt;
    const strike = Math.floor(J.gp);
    if (J.maxStrike != null && strike > J.maxStrike && !J.airborne) {
      if (musical) spawnStep(strike % 2 === 0);
      J.puffs.push({ m: J.d + J.face * J.stepLen * (J.running ? 0.4 : 0.62), t: now() }); // Where the foot lands
    }
    if (J.wasAirborne && !J.airborne) {
      if (musical) {
        spawnStep(true); // Both feet land together
        spawnStep(false);
      }
      J.puffs.push({ m: J.d, t: now(), big: true });
    }
    J.wasAirborne = J.airborne;
    J.maxStrike = Math.max(J.maxStrike ?? strike, strike);
  } else {
    J.maxStrike = Math.floor(J.gp);
  }
  J.walking = walking;
  J.perBeat = perBeat;
  // Stride from the speed the ruler really scrolls, so planted feet stay planted
  J.stepLen = walking && J.cad > 0 ? (J.groundV ?? kmh / 3.6) / J.cad : 0;
  J.running = kmh >= 7.5;
}

/** Foot position relative to the hip's ground point, for a foot at gait phase f (0..1 per stride). */
function footAt(f, stepPx, duty, lift) {
  const travel = duty * 2 * stepPx; // Ground passed under a planted foot
  if (f < duty) return [travel / 2 - (f / duty) * travel, 0];
  const t = (f - duty) / (1 - duty);
  return [-travel / 2 + smoothstep(t) * travel, -lift * Math.sin(Math.PI * t)];
}

/**
 * Two-bone IK from joint (ax, ay) toward target (tx, ty). Returns the middle
 * joint and the (reach-clamped) end. bend = -1 bends forward (knees), +1 back (elbows).
 */
function solveIK(ax, ay, tx, ty, l1, l2, bend) {
  const dx = tx - ax;
  const dy = ty - ay;
  const d = clamp(Math.hypot(dx, dy), Math.abs(l1 - l2) + 0.01, l1 + l2 - 0.01);
  const base = Math.atan2(dy, dx);
  const a = Math.acos((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d));
  const ang = base + bend * a;
  return [ax + l1 * Math.cos(ang), ay + l1 * Math.sin(ang), ax + d * Math.cos(base), ay + d * Math.sin(base)];
}

// ---------- pixel drawing ----------

function plot(buf, x, y, alpha) {
  x = Math.round(x);
  y = Math.round(y);
  if (x < 0 || y < 0 || x >= FIG_W || y >= FIG_H) return;
  const i = (y * FIG_W + x) * 4;
  if (buf[i + 3] >= alpha) return;
  buf[i] = 230;
  buf[i + 1] = 232;
  buf[i + 2] = 235;
  buf[i + 3] = alpha;
}

function line(buf, x0, y0, x1, y1, alpha) {
  x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1);
  const dx = Math.abs(x1 - x0);
  const dy = -Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  for (;;) {
    plot(buf, x0, y0, alpha);
    if (x0 === x1 && y0 === y1) break;
    const e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

function disc(buf, cx, cy, r, alpha) {
  cx = Math.round(cx); cy = Math.round(cy);
  for (let y = -r; y <= r; y++) {
    for (let x = -r; x <= r; x++) {
      if (x * x + y * y <= r * r + r * 0.8) plot(buf, cx + x, cy + y, alpha);
    }
  }
}

/** On bar lines: choose the move for the dominant note, and maybe start or end a moonwalk or jump. */
function updateDance() {
  const beat = Math.floor(M.phase);
  const P = beatPeriod();
  const musical = P != null;
  if (J.moveBeat != null && (beat === J.moveBeat || beatInBar() !== 0)) return;
  J.moveBeat = beat;
  if (!musical) {
    if (J.moonUntil != null) endMoonwalk(P);
    return;
  }
  if (M.note != null) {
    const options = movesForNote(M.note);
    // New note: its move. Same note: sometimes switch to its other move
    if (!options.includes(J.move) || (options.length > 1 && Math.random() < 0.3)) {
      J.move = options[Math.floor(Math.random() * options.length)];
    }
  }
  const bar = M.bar ?? 4;
  if (J.moonUntil != null && beat >= J.moonUntil) {
    endMoonwalk(P);
  } else if (J.moonUntil == null && !J.trick && J.walking && beat >= J.moonCooldown) {
    const r = Math.random();
    if (r < 0.12) {
      J.moonUntil = beat + 2 * bar;
      J.turn = { from: J.face, to: -1, t: now(), dur: (P ?? 0.5) * 500 };
    } else if (r < 0.28) {
      startJump(beat, P);
    }
  }
}

/**
 * Take off on a step and land on a later one. Air time is whole steps, as close
 * to a natural 0.55 s as the tempo allows; height follows from it (h = gT²/8),
 * and the scrolling ground carries the jump at walking speed.
 */
function startJump(beat, P) {
  const h = J.perBeat;
  if (!h) return;
  const stepBeats = 1 / h;
  const opts = [1, 2, 3].map((m) => ({ n: m * stepBeats, T: m * stepBeats * P })).filter((o) => o.T >= 0.3 && o.T <= 0.85);
  if (!opts.length) return;
  const { n, T } = opts.reduce((a, o) => (Math.abs(o.T - 0.55) < Math.abs(a.T - 0.55) ? o : a));
  const options = JUMPS.filter((j) => T >= (j.minAir ?? 0));
  const jump = options[Math.floor(Math.random() * options.length)];
  const height = Math.min(0.6, (9.81 * T * T) / 8) * PPM_FIG;
  const takeoff = (Math.floor((M.phase - M.barOffset) * h) + 1) / h + M.barOffset;
  J.trick = { ...jump, takeoff, land: takeoff + n, height, face0: J.face };
  J.moonCooldown = beat + 2 * (M.bar ?? 4);
}

function endMoonwalk(P) {
  J.moonUntil = null;
  J.moonCooldown = Math.floor(M.phase) + 4 * (M.bar ?? 4);
  J.turn = { from: J.face, to: 1, t: now(), dur: (P ?? 0.5) * 500 };
}

const clonePose = (p) => ({
  drop: p.drop, sway: p.sway, lean: p.lean, head: [...p.head],
  hands: p.hands.map((h) => [...h]), feet: p.feet.map((f) => [...f]),
});
const mix2 = (a, b, w) => a.map((v, i) => mix(v, b[i], w));
const mixPose = (a, b, w) => ({
  drop: mix(a.drop, b.drop, w), sway: mix(a.sway, b.sway, w), lean: mix(a.lean, b.lean, w),
  head: mix2(a.head, b.head, w),
  hands: a.hands.map((h, i) => mix2(h, b.hands[i], w)),
  feet: a.feet.map((f, i) => mix2(f, b.feet[i], w)),
});

/** Jump phase from the beat clock: crouch, air (with turns), land; null otherwise. */
function jumpPhase(musical) {
  const tr = J.trick;
  if (!tr) return { phase: null, k: 0 };
  if (!J.walking || !musical) {
    J.trick = null;
    J.turn = { from: J.face, to: 1, t: now(), dur: 250 };
    return { phase: null, k: 0 };
  }
  const tb = M.phase - tr.takeoff;
  const n = tr.land - tr.takeoff;
  const landedFace = tr.face0 * Math.round(Math.cos(2 * Math.PI * tr.rot)); // ±1
  if (tb >= -0.35 && tb < 0) return { phase: "crouch", k: smoothstep((tb + 0.35) / 0.35) };
  if (tb >= 0 && tb < n) {
    const k = tb / n;
    J.face = tr.face0 * Math.cos(2 * Math.PI * tr.rot * smoothstep(k));
    return { phase: "air", k };
  }
  if (tb >= n && tb < n + 0.4) {
    J.face = landedFace;
    return { phase: "land", k: 1 - (tb - n) / 0.4 };
  }
  if (tb >= n + 0.4) {
    J.trick = null;
    J.face = landedFace;
    if (landedFace < 0) J.moonUntil = Math.floor(M.phase) + 2 * (M.bar ?? 4); // Landed backwards
  }
  return { phase: null, k: 0 };
}

function drawFigure(dt) {
  const buf = J.img.data;
  buf.fill(0);
  const P = beatPeriod();
  const musical = P != null;
  const b = M.phase - M.barOffset; // Beats, bar-aligned
  const env = musical ? Math.exp(-(b - Math.floor(b)) * 5) : 0;
  const move = DANCE_MOVES[J.move];

  // Facing: turns sweep through 0 (a quick spin)
  if (J.turn) {
    const k = smoothstep((now() - J.turn.t) / J.turn.dur);
    J.face = mix(J.turn.from, J.turn.to, k);
    if (k >= 1) J.turn = null;
  }
  if (!J.walking && J.moonUntil != null) endMoonwalk(P);
  const tr = J.trick;
  const { phase, k } = jumpPhase(musical);
  J.airborne = phase === "air";
  const moonwalking = J.walking && J.face < 0 && phase !== "air";
  const spinning = !J.walking && musical && move.spin;
  if (spinning) J.face = Math.cos(2 * Math.PI * move.spin(b));
  const faceTarget = J.moonUntil != null ? -1 : 1;
  if (!J.trick && !J.turn && !spinning && J.face !== faceTarget) {
    J.turn = { from: J.face, to: faceTarget, t: now(), dur: 250 };
  }

  // Target pose from whatever the figure is doing
  const stepPx = J.stepLen * PPM_FIG;
  const duty = J.running ? 0.4 : 0.62;
  const s = J.gp / 2; // Stride phase
  const armsDance = musical && move.upper;
  const pose = {
    drop: 0.6, sway: 0, lean: 0, head: [0, 0],
    feet: PLANTED, hands: [[0.5, 9.5], [-0.5, 9.5]],
  };
  let src;
  let exact = false; // Walking feet must track the ground exactly
  if (phase === "air") {
    const lift = 4 * tr.height * k * (1 - k); // Parabolic flight, legs tucked
    pose.drop = 0.6 - lift;
    pose.feet = [[1.5, -lift - 3.5], [-1.5, -lift - 2.5]];
    pose.hands = tr.rot > 0 ? [[3, 1], [2.5, 1.5]] : [[5, -8.5], [-4.5, -8.5]]; // Pulled in to turn, or up
    src = "air";
  } else if (J.walking) {
    exact = true;
    if (moonwalking) {
      pose.drop += 1.2 + 0.4 * env;
      pose.feet = moonwalkFeet(J.gp, stepPx);
      pose.lean = 0.1;
      pose.hands = [[3, 6 + 0.6 * env], [-2, 7.5]]; // Loose, one hand at the waist
      src = "moon";
    } else {
      // Lowest at foot strike (both feet down), highest mid-stance
      pose.drop += (J.running ? 0.4 : 0.9) * (0.5 + 0.5 * Math.cos(2 * Math.PI * J.gp));
      const lift = J.running ? 3 : 2;
      pose.feet = [footAt(s - Math.floor(s), stepPx, duty, lift), footAt((s + 0.5) % 1, stepPx, duty, lift)];
      pose.lean = 0.05 + 0.012 * S.disp.speed;
      const a = (2.5 + 0.35 * S.disp.speed) * Math.sin(2 * Math.PI * s);
      pose.hands = J.running ? [[-a + 2, 4], [a + 2, 4]] : [[-a, 8.5], [a, 8.5]];
      src = "walk";
    }
    if (armsDance) {
      // Arms-only moves carry on over the walk (or the moonwalk)
      pose.hands = move.hands(b, env);
      if (move.head) pose.head = move.head(b, env);
      src += `:${J.move}`;
    }
    if (phase === "crouch" || phase === "land") {
      pose.drop += (phase === "crouch" ? 2.2 : 2) * k; // Load, absorb
      pose.hands = phase === "crouch" ? [[-3, 7], [-3.5, 7]] : [[5, 2], [4, 3]];
      src = phase;
    }
  } else if (musical) {
    const legs = move.legs(b, env);
    pose.drop += legs.drop;
    pose.sway = legs.sway ?? 0;
    pose.feet = legs.feet;
    pose.hands = move.hands(b, env);
    if (move.head) pose.head = move.head(b, env);
    if (move.lean) pose.lean = move.lean(b, env);
    src = `dance:${J.move}`;
  } else {
    src = "idle";
  }
  if (J.cheer && now() - J.cheer.t < 2200) {
    const w = 1.5 * Math.sin(now() / 90);
    pose.hands = [[4.5 + w, -9], [-4.5 + w, -9]];
    src += ":cheer";
  }
  pose.feet = pose.feet.map((f) => [f[0], f[1], f[2] ?? 0]); // [x, y, heel lift]

  // Changing what drives the pose (another move, walk <-> dance, moonwalk,
  // jump...) crossfades from where the limbs are now over about half a beat;
  // IK solves every in-between frame, so arms and legs travel rather than snap
  J.pose ??= clonePose(pose);
  if (src !== J.src) {
    J.src = src;
    J.from = clonePose(J.pose);
    J.blendT = now();
    J.blendDur = clamp((P ?? 0.6) * 0.5, 0.18, 0.4) * 1000;
  }
  const w = J.from ? smoothstep((now() - J.blendT) / J.blendDur) : 1;
  if (w >= 1) J.from = null;
  const target = J.from ? mixPose(J.from, pose, w) : pose;

  // Follow the target: tight enough for beat-snapped moves, soft enough to never pop
  const cur = J.pose;
  const kf = 1 - Math.exp(-dt / 0.05);
  const follow = (a, t) => a + (t - a) * kf;
  cur.drop = follow(cur.drop, target.drop);
  cur.sway = follow(cur.sway, target.sway);
  cur.lean = follow(cur.lean, target.lean);
  cur.head = cur.head.map((v, i) => follow(v, target.head[i]));
  cur.hands = cur.hands.map((h, i) => h.map((v, j) => follow(v, target.hands[i][j])));
  const exactFeet = exact && !J.from && !phase;
  cur.feet = cur.feet.map((f, i) => f.map((v, j) => (exactFeet ? target.feet[i][j] : follow(v, target.feet[i][j]))));

  // Skeleton
  const lean = cur.lean;
  const hx = FOOT_X + cur.sway;
  const hy = FOOT_Y - (HIP_STAND + 0.6 - cur.drop);
  const sx = hx + Math.sin(lean) * TORSO;
  const sy = hy - Math.cos(lean) * TORSO;
  const bob = musical ? env * 0.8 : 0;
  const headX = sx + Math.sin(lean) * (HEAD_R + NECK) + cur.head[0];
  const headY = sy - Math.cos(lean) * (HEAD_R + NECK) + bob + cur.head[1];

  /** Hand target in figure pixels, pushed out of the head so hands never clip through it. */
  const handAt = (h) => {
    let x = sx + h[0] * ARM_SCALE;
    let y = sy + 1 + h[1] * ARM_SCALE;
    const dx = x - headX;
    const dy = y - headY;
    const d = Math.hypot(dx, dy);
    const clear = HEAD_R + 1.5;
    if (d < clear) {
      x = d > 0.01 ? headX + (dx / d) * clear : headX + clear;
      y = d > 0.01 ? headY + (dy / d) * clear : headY;
    }
    return [x, y];
  };

  // Far limbs dimmer for depth; draw far first
  const FAR = 120;
  const NEAR = 235;
  const drawLeg = (foot, alpha) => {
    // Ankle rises by the heel lift; the toe stays down
    const [kx, ky, ax, ay] = solveIK(hx, hy, FOOT_X + foot[0], FOOT_Y + foot[1] - foot[2], THIGH, SHIN, -1);
    line(buf, hx, hy, kx, ky, alpha);
    line(buf, kx, ky, ax, ay, alpha);
    line(buf, ax, ay, ax + 2, ay + foot[2], alpha); // Foot
  };
  /** Closest the segment a-b comes to the head centre. */
  const headGap = (ax, ay, bx, by) => {
    const vx = bx - ax;
    const vy = by - ay;
    const t = clamp(((headX - ax) * vx + (headY - ay) * vy) / (vx * vx + vy * vy || 1), 0, 1);
    return Math.hypot(ax + vx * t - headX, ay + vy * t - headY);
  };
  const drawArm = (hand, alpha) => {
    const [tx, ty] = handAt(hand);
    // Two ways to bend an elbow; keep the natural one unless the other clears the head better
    const solve = (bend) => {
      const j = solveIK(sx, sy + 1, tx, ty, UPPER_ARM, FOREARM, bend);
      return { j, gap: Math.min(headGap(sx, sy + 1, j[0], j[1]), headGap(j[0], j[1], j[2], j[3])) };
    };
    const natural = solve(1);
    const other = solve(-1);
    const { j } = natural.gap >= HEAD_R + 0.8 || natural.gap >= other.gap ? natural : other;
    line(buf, sx, sy + 1, j[0], j[1], alpha);
    line(buf, j[0], j[1], j[2], j[3], alpha);
  };
  drawLeg(cur.feet[1], FAR);
  drawArm(cur.hands[1], FAR);
  line(buf, hx, hy, sx, sy, NEAR);
  disc(buf, headX, headY, HEAD_R, NEAR);
  drawLeg(cur.feet[0], NEAR);
  drawArm(cur.hands[0], NEAR);
  J.figCtx.putImageData(J.img, 0, 0);
}

// ---------- scene ----------

function updateJourney(dt) {
  const c = document.getElementById("journey-canvas");
  const w = c.clientWidth;
  if (!w) return;
  const dpr = window.devicePixelRatio || 1;
  if (w !== J.w || c.height !== Math.round(BAR_H * dpr)) {
    J.w = w;
    c.width = Math.round(w * dpr);
    c.height = Math.round(BAR_H * dpr);
    J.ctx = c.getContext("2d");
    J.ctx.scale(dpr, dpr);
  }
  const ctx = J.ctx;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, w, BAR_H);

  updateDistance(dt);
  updateGait(dt);

  updateDance();

  const d = J.d ?? 0;
  const pad = 40;
  // Walk in from the left, then hold the middle while the world scrolls
  const figX = Math.min(pad + d * PPM, w / 2);
  const toX = (m) => figX + (m - d) * PPM;
  const fromX = (x) => d + (x - figX) / PPM;
  const y = GROUND_Y;

  // Ground and ruler (metres, or yards in mph mode)
  const u = unit() === "mph" ? 0.9144 : 1;
  const uName = unit() === "mph" ? "yd" : "m";
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  ctx.fillRect(0, y, w, 1);
  ctx.fillStyle = "rgba(255,255,255,0.28)";
  ctx.fillRect(Math.max(0, toX(0)), y, Math.max(0, figX - Math.max(0, toX(0))), 1);
  const first = Math.max(0, Math.floor(fromX(0) / u));
  const last = Math.ceil(fromX(w) / u);
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  for (let n = first; n <= last; n++) {
    const x = Math.round(toX(n * u)) + 0.5;
    const major = n % 10 === 0;
    ctx.fillStyle = `rgba(255,255,255,${major ? 0.22 : 0.08})`;
    ctx.fillRect(x - 0.5, y + 2, 1, major ? 6 : 3);
    if (major && n > 0) {
      ctx.fillStyle = n % 100 === 0 ? "rgba(255,255,255,0.45)" : "rgba(255,255,255,0.22)";
      ctx.fillText(n % 100 === 0 ? fmtDistance(n * u) : `${n} ${uName}`, x, y + 20);
    }
  }
  drawPixelPost(ctx, toX(0), y);

  // Foot strikes: a short bright tick where each foot lands, riding away with the ground
  J.puffs = J.puffs.filter((pf) => now() - pf.t < 600);
  for (const pf of J.puffs) {
    const a = 1 - (now() - pf.t) / 600;
    const wpx = (pf.big ? 14 : 8) + (1 - a) * 6;
    ctx.fillStyle = `hsla(${M.hue.toFixed(0)}, 40%, 80%, ${0.55 * a})`;
    ctx.fillRect(Math.round(toX(pf.m) - wpx / 2), y - 1, Math.round(wpx), 2);
  }

  // Milestones: passed ones stay behind you, the next one's label waits at the right edge
  const nextIdx = J.chain.findIndex((m) => m.m > d);
  const passedIdx = (nextIdx === -1 ? J.chain.length : nextIdx) - 1;
  if (J.passedIdx != null && passedIdx > J.passedIdx) J.cheer = { ...J.chain[passedIdx], t: now() };
  J.passedIdx = passedIdx;
  J.chain.forEach((m, i) => {
    const x = toX(m.m);
    const isNext = i === nextIdx;
    if (x < -200 || (x > w + 20 && !isNext)) return;
    const passed = m.m <= d;
    if (x <= w + 20) drawPixelFlag(ctx, x, y, passed ? 0.3 : 0.85);
    if (passed && x < -10) return;
    if (J.cheer && J.cheer.m === m.m && now() - J.cheer.t < 2200) return; // The cheer says it
    if (x <= w + 20) {
      // Hairline from the flag up to its label
      ctx.fillStyle = `rgba(255,255,255,${passed ? 0.05 : 0.1})`;
      ctx.fillRect(Math.round(x), LABEL_Y + 18, 1, y - 24 - (LABEL_Y + 18));
    }
    ctx.textAlign = "center";
    ctx.font = isNext ? "13px system-ui, sans-serif" : "12px system-ui, sans-serif";
    // Kept on screen: the next landmark's label waits at the right edge until its flag arrives
    const lw = ctx.measureText(passed ? `✓ ${m.label}` : m.label).width;
    const lx = clamp(x, 8 + lw / 2, w - 16 - lw / 2);
    ctx.fillStyle = passed ? "rgba(255,255,255,0.3)" : isNext ? "rgba(230,232,235,0.92)" : "rgba(255,255,255,0.45)";
    ctx.fillText(passed ? `✓ ${m.label}` : m.label, lx, LABEL_Y);
    if (isNext) {
      const togo = m.m - d;
      const v = S.disp.speed / 3.6;
      const eta = v > 0.2 ? ` · ${fmtDuration(Math.round(togo / v))}` : "";
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.fillText(`${fmtDistance(togo)} to go${eta}`, lx, LABEL_Y + 14);
      if (x > w + 20) {
        ctx.fillStyle = "rgba(255,255,255,0.35)";
        ctx.fillText("→", w - 8, y - 4);
      }
    }
  });

  // Cheer: a little label floating up from the landmark you just passed
  if (J.cheer && now() - J.cheer.t < 2200) {
    const t = (now() - J.cheer.t) / 2200;
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = `rgba(230,232,235,${(1 - t) * 0.9})`;
    ctx.fillText(`passed ${J.cheer.label}!`, figX, y - 66 - t * 10);
  }

  drawFigure(dt);
  ctx.save();
  ctx.translate(Math.round(figX), 0);
  // Mirrored while moonwalking; narrows as it spins, but never to a sliver
  ctx.scale((J.face < 0 ? -1 : 1) * Math.max(Math.abs(J.face), 0.3), 1);
  ctx.drawImage(J.fig, -FOOT_X * PIX, Math.round(y + 1 - (FOOT_Y + 1) * PIX), FIG_W * PIX, FIG_H * PIX);
  ctx.restore();


}

function drawPixelFlag(ctx, x, y, alpha) {
  x = Math.round(x);
  ctx.fillStyle = `rgba(230,232,235,${alpha})`;
  ctx.fillRect(x, y - 24, PIX, 24);
  const pennant = [10, 8, 6, 4, 2];
  pennant.forEach((len, i) => ctx.fillRect(x + PIX, y - 24 + i * PIX, len, PIX));
}

function drawPixelPost(ctx, x, y) {
  if (x < -10) return;
  ctx.fillStyle = "rgba(230,232,235,0.35)";
  ctx.fillRect(Math.round(x), y - 10, PIX, 10);
}
