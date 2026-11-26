/* LTV playground engine: the models, in the browser.
 *
 * A JavaScript copy of the inference path of the Python package, for the static (GitHub
 * Pages) version of the playground, where there is no server to run Python:
 *
 *   features           src/ltv/data/np_features.py        (tabular_features, calendar_features)
 *   calendar           src/ltv/data/splits.py             (project_calendar, median_season_length)
 *   trees A1, A2       src/ltv/models/trees.py            (predict_a1, predict_a2) + LightGBM's tree walk
 *   GRU C1             src/ltv/models/seq.py, data/seq_inputs.py (gru2res encoder, ZILN heads)
 *   calibration, path  src/ltv/eval/calibration.py, eval/examples.py
 *   forecast, timeline src/ltv/playground/core.py
 *
 * It is the THIRD copy of the features (after the SQL and np_features.py). It exists only
 * for the public demo, which carries models trained on synthetic players. If any of the
 * files above changes, change this one too: tests/test_playground.py scores the same
 * players both ways and fails when they disagree.
 *
 * No DOM here. Works in a browser (window.LTVEngine) and in Node (module.exports).
 */
(function (root) {
  "use strict";
  const HORIZON = 52, CUM_HORIZONS = [4, 13, 26, 39, 52], BUCKET_LEN = [4, 9, 13, 13, 13];
  const WINDOWS = [4, 13, 26, 52, 104], LAPSES = [4, 13, 26], MAX_HOURS = 168, LOOKBACK = 104;
  const BLOCK = 4, RECENT = 26, MAX_BLOCKS = 130, SIGMA_MIN = 0.05, SIGMA_MAX = 2.5;
  const TARGETS = ["p_reengage", "exp_active_weeks", "exp_hours", "p_spend", "exp_spend"];
  const PROB = { p_reengage: true, p_spend: true };
  const f32 = Math.fround;

  // ---- typed arrays from base64 -------------------------------------------------------------------
  function bytes(b64) {
    if (typeof Buffer !== "undefined") { const b = Buffer.from(b64, "base64"); return new Uint8Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); }
    const s = atob(b64), out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }
  const TYPES = { u8: Uint8Array, i8: Int8Array, i32: Int32Array, f32: Float32Array, f64: Float64Array };
  const arr = (spec) => new TYPES[spec.t](bytes(spec.b).buffer);

  // ---- calendar (splits.py) -----------------------------------------------------------------------
  function roundHalfEven(x) { const r = Math.round(x); return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; }
  function medianSeasonLength(si, upto) {
    const n = upto + 1;
    if (n <= 0) return 8;
    const change = [];
    for (let i = 0; i < n - 1; i++) if (si[i + 1] !== si[i]) change.push(i);
    if (!change.length) return Math.max(n, 1);
    const starts = [0].concat(change.map((c) => c + 1)), lengths = [];
    for (let i = 1; i < starts.length; i++) lengths.push(starts[i] - starts[i - 1]);
    if (!lengths.length) return Math.max(change[0] + 1, 1);
    lengths.sort((a, b) => a - b);
    const m = lengths.length, med = m % 2 ? lengths[(m - 1) / 2] : (lengths[m / 2 - 1] + lengths[m / 2]) / 2;
    return Math.max(roundHalfEven(med), 1);
  }
  function projectCalendar(sw, si, origin, horizon, knownWindow) {
    const n = sw.length;
    let k = 0;
    if (knownWindow === "current_season") { while (origin + k + 1 < n && si[origin + k + 1] === si[origin] && k < horizon) k++; }
    else k = Math.min(+knownWindow, horizon, n - 1 - origin);
    const med = medianSeasonLength(si, origin), out = new Array(horizon), known = new Array(horizon).fill(false);
    let cur = sw[origin];
    for (let h = 1; h <= horizon; h++) {
      if (h <= k) { cur = sw[origin + h]; known[h - 1] = true; }
      else if (knownWindow === "current_season" && h === k + 1 && origin + k + 1 < n) cur = 1;
      else cur = cur < med ? cur + 1 : 1;
      out[h - 1] = cur;
    }
    return { sw: out, known };
  }

  // ---- features (np_features.py), one player --------------------------------------------------------
  // H, S: weekly hours and spend as float32 values, index = week_idx, length >= t + 1.
  function features(names, H, S, first, sw, si, t, proj) {
    const X = {}, act = (w) => H[w] > 0, sp = (w) => S[w] > 0;
    const sum = (fn, lo, hi) => { let s = 0; for (let w = Math.max(lo, 0); w < Math.max(hi, 0); w++) s += fn(w); return s; };
    const win = (fn, w) => sum(fn, t - w + 1, t + 1);
    let lastAct = -1, lastSp = -1, lastRow = -1;
    for (let w = 0; w <= t; w++) { if (act(w)) lastAct = w; if (sp(w)) lastSp = w; if (act(w) || sp(w)) lastRow = w; }
    X.tenure_weeks = t - first;
    X.wk_since_active = lastAct >= 0 ? t - lastAct : NaN;
    X.wk_since_spend = lastSp >= 0 ? t - lastSp : NaN;
    for (const w of WINDOWS) {
      X["aw_" + w] = win((i) => +act(i), w); X["hrs_" + w] = win((i) => H[i], w);
      X["spd_" + w] = win((i) => S[i], w); X["sw_" + w] = win((i) => +sp(i), w);
    }
    const awLife = sum((i) => +act(i), 0, t + 1);
    X.aw_life = awLife; X.hrs_life = sum((i) => H[i], 0, t + 1); X.spd_life = sum((i) => S[i], 0, t + 1); X.sw_life = sum((i) => +sp(i), 0, t + 1);
    let mx = -Infinity; for (let w = Math.max(t - 51, 0); w <= t; w++) mx = Math.max(mx, H[w]);
    X.max_hrs_52 = mx;
    const prior13 = sum((i) => +act(i), t - 16, t - 3);
    X.ratio_aw_4_13 = (X.aw_4 / 4.0) / ((prior13 + 0.5) / 13.0);
    let slope = 0; for (let w = Math.max(t - 25, 0); w <= t; w++) slope += H[w] * (w - (t - 25) - 12.5);
    X.hrs_slope_26 = slope / 1462.5;
    const launch = (w) => sw[w] <= 2;
    X.share_aw_launch = awLife > 0 ? sum((i) => +(act(i) && launch(i)), 0, t + 1) / awLife : NaN;
    let seasons = 0, any = false;
    for (let w = 0; w <= t; w++) {
      if (w > 0 && si[w] !== si[w - 1]) { seasons += +any; any = false; }
      any = any || act(w);
    }
    X.n_seasons_active = seasons + +any;
    X.aw_next13_1y = sum((i) => +act(i), t - 51, t - 38);
    X.aw_next13_2y = sum((i) => +act(i), t - 103, t - 90);
    X.origin_season_week = sw[t];
    // lapse and return
    const gaps4 = []; let prev = -1, nSpells = 0, extra = 0, lastRet = -1, maxGap = -1, retLaunch = 0;
    const nRet = { 4: 0, 13: 0, 26: 0 };
    for (let w = 0; w <= t; w++) {
      if (!act(w)) continue;
      const gap = prev >= 0 ? w - prev - 1 : -1, start = prev < 0 || gap >= 4;
      for (const thr of LAPSES) if (gap >= thr) nRet[thr]++;
      if (gap >= 4) { gaps4.push(gap); if (launch(w)) retLaunch++; }
      if (gap > maxGap) maxGap = gap;
      if (start) { nSpells++; lastRet = w; } else extra += w - prev;
      prev = w;
    }
    for (const thr of LAPSES) X["n_returns_" + thr] = nRet[thr];
    gaps4.sort((a, b) => a - b);
    const m = gaps4.length, med = !m ? NaN : m % 2 ? gaps4[(m - 1) / 2] : (gaps4[m / 2 - 1] + gaps4[m / 2]) / 2;
    X.median_lapse = med;
    X.max_lapse = Math.max(maxGap, 0);
    X.cur_lapse_rel = med > 0 ? X.wk_since_active / med : NaN;
    X.n_spells = nSpells;
    X.mean_spell_len = nSpells > 0 ? (extra + nSpells) / nSpells : NaN;
    X.wk_since_return = lastRet >= 0 ? t - lastRet : NaN;
    X.share_returns_launch = nRet[4] > 0 ? retLaunch / nRet[4] : NaN;
    // context: the playground gives every week with a row platform 1 and country 1
    X.platform_id = lastRow >= 0 ? 1 : 0; X.country_id = lastRow >= 0 ? 1 : 0; X.n_platforms = lastRow >= 0 ? 1 : 0;
    // projected calendar
    const first1 = proj.sw.indexOf(1);
    X.wk_to_next_launch = first1 >= 0 ? first1 + 1 : NaN;
    const q = [0, 0, 0, 0];
    proj.sw.forEach((v, i) => { if (v === 1) { const h = i + 1; q[h <= 13 ? 0 : h <= 26 ? 1 : h <= 39 ? 2 : 3]++; } });
    X.launches_q1 = q[0]; X.launches_q2 = q[1]; X.launches_q3 = q[2]; X.launches_q4 = q[3];
    const out = new Float64Array(names.length);
    names.forEach((nm, i) => { out[i] = f32(X[nm]); });          // the Python frame is float32
    return out;
  }

  // ---- trees (LightGBM's decision rule; trees.py for how the heads combine) ---------------------------
  function Trees(spec) {
    const T = spec.arrays, feature = arr(T.feature), flags = arr(T.flags), threshold = arr(T.threshold);
    const left = arr(T.left), right = arr(T.right), value = arr(T.value), nodeStart = arr(T.node_start), leafStart = arr(T.leaf_start);
    const cats = spec.cats.map((c) => new Set(c));
    function tree(i, x) {
      const n0 = nodeStart[i], l0 = leafStart[i];
      if (nodeStart[i + 1] === n0) return value[l0];                 // a tree that is one leaf
      let node = 0;
      for (;;) {
        const g = n0 + node, fl = flags[g], missing = (fl >> 1) & 3;
        let v = x[feature[g]], goLeft;
        if (fl & 8) {                                                // categorical: left if the value is in the set
          goLeft = !Number.isNaN(v) && v >= 0 && cats[threshold[g]].has(Math.trunc(v));
        } else {
          if (Number.isNaN(v) && missing !== 2) v = 0.0;
          if ((missing === 1 && Math.abs(v) <= 1e-35) || (missing === 2 && Number.isNaN(v))) goLeft = !!(fl & 1);
          else goLeft = v <= threshold[g];
        }
        const nx = goLeft ? left[g] : right[g];
        if (nx < 0) return value[l0 + ~nx];
        node = nx;
      }
    }
    function predict(name, x) {
      const b = spec.boosters[name];
      let s = 0;
      for (let i = b.first; i < b.first + b.n; i++) s += tree(i, x);
      return b.link === "sigmoid" ? 1 / (1 + Math.exp(-s)) : b.link === "exp" ? Math.exp(s) : s;
    }
    return {
      a1(x) {
        const pSpend = predict("any_spend", x);
        return { p_reengage: predict("any_active", x), exp_active_weeks: predict("active_weeks", x), exp_hours: predict("hours", x),
                 p_spend: pSpend, exp_spend: pSpend * Math.exp(predict("log_spend_pos", x)) * spec.smear_spend,
                 cum_hours: null, cum_spend: null, weekly: null };
      },
      a2(x) {
        const aw = [], hrs = [], spd = [], weekly = [];
        let ch = 0, cs = 0;
        for (let k = 0; k < 5; k++) {
          aw.push(Math.min(predict("aw_b" + k, x), BUCKET_LEN[k]));
          ch += predict("hrs_b" + k, x); hrs.push(ch); cs += predict("spd_b" + k, x); spd.push(cs);
          for (let j = 0; j < BUCKET_LEN[k]; j++) weekly.push(aw[k] / BUCKET_LEN[k]);
        }
        return { p_reengage: predict("any_active", x), exp_active_weeks: aw.reduce((a, b) => a + b, 0), exp_hours: hrs[4],
                 p_spend: predict("any_spend", x), exp_spend: spd[4], cum_hours: hrs, cum_spend: spd, weekly };
      },
    };
  }

  // ---- GRU C1 (seq.py, seq_inputs.py) ----------------------------------------------------------------
  function Gru(spec, names) {
    const W = {};
    for (const k in spec.weights) W[k] = { d: arr(spec.weights[k]), s: spec.weights[k].s };
    const d = spec.cfg.d, catIdx = spec.cat_idx, numIdx = names.map((_, i) => i).filter((i) => !catIdx.includes(i));
    const sig = (v) => 1 / (1 + Math.exp(-v));
    const linear = (name, x) => {                                    // y = W x + b
      const w = W[name + ".weight"], b = W[name + ".bias"], rows = w.s[0], cols = w.s[1], y = new Float64Array(rows);
      for (let r = 0; r < rows; r++) { let s = b.d[r]; const o = r * cols; for (let c = 0; c < cols; c++) s += w.d[o + c] * x[c]; y[r] = s; }
      return y;
    };
    const map = (y, fn) => { for (let i = 0; i < y.length; i++) y[i] = fn(y[i]); return y; };
    function gruRun(name, xs, h) {                                   // PyTorch gate order: reset, update, new
      const wi = W[name + ".weight_ih_l0"], wh = W[name + ".weight_hh_l0"], bi = W[name + ".bias_ih_l0"].d, bh = W[name + ".bias_hh_l0"].d;
      const nin = wi.s[1], gi = new Float64Array(3 * d), gh = new Float64Array(3 * d);
      for (const x of xs) {
        for (let r = 0; r < 3 * d; r++) {
          let a = bi[r], b = bh[r]; const oi = r * nin, oh = r * d;
          for (let c = 0; c < nin; c++) a += wi.d[oi + c] * x[c];
          for (let c = 0; c < d; c++) b += wh.d[oh + c] * h[c];
          gi[r] = a; gh[r] = b;
        }
        const nh = new Float64Array(d);
        for (let j = 0; j < d; j++) {
          const r = sig(gi[j] + gh[j]), z = sig(gi[d + j] + gh[d + j]), n = Math.tanh(gi[2 * d + j] + r * gh[2 * d + j]);
          nh[j] = (1 - z) * n + z * h[j];
        }
        h = nh;
      }
      return h;
    }
    const weekFeats = (H, S, w) => [+(H[w] > 0), f32(Math.log1p(H[w])), f32(Math.log1p(S[w])), +(S[w] > 0)];
    return function score(H, S, first, sw, t, x, proj) {
      // history blocks on the fixed 4-week grid, then the recent weeks one by one
      const nb = Math.max(Math.floor((t - RECENT + 1) / BLOCK), 0), fb = Math.floor(first / BLOCK);
      let len = Math.max(nb - fb, 0), start = fb;
      if (len > MAX_BLOCKS) { start = nb - MAX_BLOCKS; len = MAX_BLOCKS; }
      const blocks = [];
      for (let j = start; j < start + len; j++) {
        const v = [];
        for (let w = j * BLOCK; w < (j + 1) * BLOCK; w++) v.push(...weekFeats(H, S, w));
        for (let w = j * BLOCK; w < (j + 1) * BLOCK; w++) v.push(+(sw[w] === 1));
        v.push(f32(sw[j * BLOCK] / 10));
        blocks.push(map(linear("hist_proj.0", v), Math.tanh));
      }
      let h = new Float64Array(d);
      if (len > 0) h = gruRun("hist_gru", blocks, h);                // no complete block: the state stays zero
      h = map(linear("bridge", h), Math.tanh);
      const lo = nb * BLOCK, weeks = [];
      for (let w = lo + Math.max(first - lo, 0); w <= t; w++) weeks.push(map(linear("week_proj.0", weekFeats(H, S, w).concat([f32(sw[w] / 10), +(sw[w] === 1)])), Math.tanh));
      h = gruRun("week_gru", weeks, h);
      // static inputs: the trees' features, compressed, with missing flags; then two embeddings
      const sm = W.s_mean.d, ss = W.s_std.d, n = numIdx.length, st = new Float64Array(2 * n + 10);
      numIdx.forEach((i, j) => {
        const miss = Number.isNaN(x[i]), v = miss ? 0 : x[i];
        st[j] = (f32(Math.sign(v) * Math.log1p(Math.abs(v))) - sm[j]) / ss[j];
        st[n + j] = ((miss ? 1 : 0) - sm[n + j]) / ss[n + j];
      });
      const clampId = (v, hi) => Math.min(Math.max(Number.isNaN(v) ? 0 : Math.trunc(v), 0), hi - 1);
      const ep = W["emb_platform.weight"], ec = W["emb_country.weight"], p = clampId(x[catIdx[0]], ep.s[0]), c = clampId(x[catIdx[1]], ec.s[0]);
      for (let j = 0; j < 4; j++) st[2 * n + j] = ep.d[p * 4 + j];
      for (let j = 0; j < 6; j++) st[2 * n + 4 + j] = ec.d[c * 6 + j];
      const relu = (v) => (v > 0 ? v : 0), sv = map(linear("static_net.0", st), relu);
      const both = new Float64Array(2 * d); both.set(h); both.set(sv, d);
      const z = map(linear("trunk.3", map(linear("trunk.0", both), relu)), relu);
      // heads
      const zp = linear("z_proj", z), bias = linear("act_bias", z)[0], we = W.week_emb, weekly = [];
      for (let k = 0; k < HORIZON; k++) {
        const cal = linear("cal_proj", [f32(proj.sw[k] / 10), +(proj.sw[k] === 1), +proj.known[k]]), u = new Float64Array(zp.length);
        for (let j = 0; j < zp.length; j++) u[j] = Math.tanh(zp[j] + cal[j] + we.d[k * we.s[1] + j]);
        weekly.push(sig(linear("act_out", u)[0] + bias));
      }
      const softplus = (v) => (v > 20 ? v : Math.log1p(Math.exp(v)));
      const mean = (name) => { const o = linear(name + ".out", z), s = Math.min(Math.max(softplus(o[1]), SIGMA_MIN), SIGMA_MAX); return Math.exp(o[0] + 0.5 * s * s); };
      const cum = (name, total) => {
        const o = linear(name, z), mx = Math.max(...o); let lse = 0; for (const v of o) lse += Math.exp(v - mx);
        lse = mx + Math.log(lse); let run = 0; return Array.from(o, (v) => total * (run += Math.exp(v - lse)));
      };
      const pRe = sig(linear("any_active", z)[0]), pSp = sig(linear("any_spend", z)[0]);
      const eh = pRe * mean("hours_head"), es = pSp * mean("spend_head");
      return { p_reengage: pRe, exp_active_weeks: weekly.reduce((a, b) => a + b, 0), exp_hours: eh, p_spend: pSp, exp_spend: es,
               cum_hours: cum("share_hours", eh), cum_spend: cum("share_spend", es), weekly };
    };
  }

  // ---- calibration and weekly paths (calibration.py, examples.py) -----------------------------------
  function interp(p, x, v) {
    const n = x.length;
    if (p <= x[0]) return v[0];
    if (p >= x[n - 1]) return v[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (x[mid] <= p) lo = mid; else hi = mid; }
    return v[lo] + (v[hi] - v[lo]) * (p - x[lo]) / (x[hi] - x[lo]);
  }
  function mapValue(m, p, prob) {
    const last = m.x.length - 1;
    return (!prob && m.x[last] > 0 && p > m.x[last]) ? m.v[last] * p / m.x[last] : interp(p, m.x, m.v);
  }
  function calibrate(raw, maps) {
    const p = {};
    for (const t of TARGETS) {
      let v = PROB[t] ? Math.min(Math.max(raw[t], 0), 1) : Math.max(raw[t], 0);       // Preds clips on construction
      p[t + "_raw"] = v;
      p[t] = maps ? mapValue(maps[t], v, !!PROB[t]) : v;
      p[t] = PROB[t] ? Math.min(Math.max(p[t], 0), 1) : Math.max(p[t], 0);
    }
    const scaled = (a, total, cap) => {
      if (!a) return null;
      if (!maps) return a.slice();
      const nw = p[total], old = p[total + "_raw"], even = a.length === 5 ? CUM_HORIZONS.map((c) => c / HORIZON) : a.map(() => 1 / a.length);
      return a.map((v, i) => { const o = old > 1e-12 ? v * (nw / Math.max(old, 1e-12)) : nw * even[i]; return Math.min(Math.max(o, 0), cap === undefined ? Infinity : cap); });
    };
    p.cum_hours = scaled(raw.cum_hours, "exp_hours"); p.cum_spend = scaled(raw.cum_spend, "exp_spend"); p.weekly = scaled(raw.weekly, "exp_active_weeks", 1.0);
    return p;
  }
  function weeklyPath(cum, weekly) {
    const out = new Array(HORIZON).fill(0);
    let lo = 0, prev = 0;
    CUM_HORIZONS.forEach((hi, b) => {
      const inc = Math.max(cum[b] - prev, 0); prev = cum[b];
      let tot = 0; for (let w = lo; w < hi; w++) tot += weekly ? weekly[w] : 1;
      for (let w = lo; w < hi; w++) out[w] = inc * (tot > 0 ? (weekly ? weekly[w] : 1) / Math.max(tot, 1e-12) : 1 / (hi - lo));
      lo = hi;
    });
    return out;
  }
  const even = (total) => CUM_HORIZONS.map((c) => total * c / HORIZON);

  // ---- the playground (playground/core.py) -----------------------------------------------------------
  function cleanHistory(hours, spend, maxWeeks) {
    const num = (v) => (Number.isFinite(+v) ? +v : 0);
    let h = Array.from(hours, num), s = Array.from(spend || [], num);
    if (s.length < h.length) s = new Array(h.length - s.length).fill(0).concat(s);
    s = s.slice(s.length - h.length);
    h = h.map((v) => Math.min(Math.max(v, 0), MAX_HOURS)); s = s.map((v) => Math.max(v, 0));
    const first = h.findIndex((v, i) => v > 0 || s[i] > 0);
    if (first < 0) throw new Error("The history is empty. Give the player at least one week with hours or spend.");
    return { h: h.slice(first).slice(-maxWeeks), s: s.slice(first).slice(-maxWeeks) };
  }

  function create(model, players) {
    const meta = model.meta, origin = meta.origin_week, sw = model.calendar.season_week, si = model.calendar.season_idx;
    const names = meta.feature_names, trees = model.trees ? Trees(model.trees) : null, gru = model.gru ? Gru(model.gru, names) : null;
    const ids = meta.models.filter((m) => (m === "C1" ? gru : trees));
    const maps = model.calibration || {};
    function store(hours, spend) {
      const c = cleanHistory(hours, spend, origin + 1), first = origin + 1 - c.h.length;
      const H = new Float32Array(origin + 1), S = new Float32Array(origin + 1);
      c.h.forEach((v, i) => { H[first + i] = v; }); c.s.forEach((v, i) => { S[first + i] = v; });
      return { H, S, first };
    }
    function scoreAll(st, t) {                                       // raw outputs of every model at week t
      const proj = projectCalendar(sw, si, t, HORIZON, meta.known_window);
      const x = features(names, st.H, st.S, st.first, sw, si, t, proj), out = {};
      for (const m of ids) out[m] = m === "A1" ? trees.a1(x) : m === "A2" ? trees.a2(x) : gru(st.H, st.S, st.first, sw, t, x, proj);
      return { out, x };
    }
    function status(st) {
      let lastAct = -1, lastRow = -1;
      for (let w = 0; w <= origin; w++) { if (st.H[w] > 0) lastAct = w; if (st.H[w] > 0 || st.S[w] > 0) lastRow = w; }
      return { weeks_of_history: origin + 1 - st.first, weeks_since_active: lastAct >= 0 ? origin - lastAct : null,
               in_population: lastRow >= 0 && origin - lastRow < LOOKBACK };
    }
    function forecast(hours, spend) {
      const st = store(hours, spend), raw = scoreAll(st, origin).out, models = {};
      for (const m of ids) {
        const p = calibrate(raw[m], maps[m]), totals = {};
        for (const t of TARGETS) totals[t] = p[t];
        models[m] = { totals, has_shape: raw[m].cum_hours !== null, calibrated: !!maps[m],
          path: { hours: weeklyPath(p.cum_hours || even(p.exp_hours), p.weekly), spend: weeklyPath(p.cum_spend || even(p.exp_spend), p.weekly),
                  active: p.weekly || new Array(HORIZON).fill(p.exp_active_weeks / HORIZON) } };
      }
      return Object.assign({ models }, status(st));
    }
    function timeline(hours, spend, weeks, step) {
      const st = store(hours, spend), lo = Math.max(st.first, origin - (weeks === undefined ? 104 : weeks)), origins = [], models = {};
      for (let t = origin; t >= lo; t -= Math.max(step || 1, 1)) origins.unshift(t);
      for (const m of ids) { models[m] = {}; for (const t of TARGETS) models[m][t] = []; }
      for (const t of origins) {
        const raw = scoreAll(st, t).out;
        for (const m of ids) { const p = calibrate(raw[m], maps[m]); for (const tg of TARGETS) models[m][tg].push(p[tg]); }
      }
      return { offsets: origins.map((t) => t - origin), models };
    }
    function calendarMarks() {
      const past = [], proj = projectCalendar(sw, si, origin, HORIZON, meta.known_window), future = [], known = [];
      for (let w = 0; w <= origin; w++) if (sw[w] === 1) past.push(w - origin);
      proj.sw.forEach((v, i) => { if (v === 1) { future.push(i + 1); if (proj.known[i]) known.push(i + 1); } });
      return { past, future, future_known: known };
    }
    function simulate(world, want, seed) {                           // a draw from the library of simulated players
      const pool = (players && players[world] && players[world][want]) || [];
      if (!pool.length) throw new Error("no simulated players of that kind in this demo");
      const i = seed === undefined || seed === null ? Math.floor(Math.random() * pool.length) : Math.abs(seed) % pool.length;
      const p = pool[i], dense = (rows, n, col) => { const a = new Array(n).fill(0); rows.forEach((r) => { a[r[0]] = r[col]; }); return a; };
      return { hours: dense(p.h, p.n, 1), spend: dense(p.h, p.n, 2), future_hours: dense(p.f, HORIZON, 1), future_spend: dense(p.f, HORIZON, 2),
               world, want, seed: p.seed, matched: true, returns: p.returns };
    }
    function info() {
      return { models: ids.map((m) => ({ id: m, label: meta.labels[m] || m, calibrated: !!maps[m] })), fit: meta.fit, origin_week: origin,
               max_weeks: origin + 1, horizon: HORIZON, synthetic: !!meta.synthetic, world: meta.world, trained_users: meta.users,
               bundle: "static", worlds: meta.worlds, wants: meta.wants, presets: model.presets, calendar: calendarMarks(), lookback: LOOKBACK, static: true };
    }
    return { info, forecast, timeline, simulate, _features: (hours, spend, t) => { const st = store(hours, spend); return Array.from(scoreAll(st, t === undefined ? origin : t).x); } };
  }

  const api = { create };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.LTVEngine = api;
})(typeof self !== "undefined" ? self : this);
