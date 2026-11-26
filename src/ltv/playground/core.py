"""Playground logic: load saved fits, turn a designed history into a one-player store, score it.

A *bundle* is a folder laid out like the pipeline's artifact path:

    trees/suite_<fit>.pkl             the tree models (A1, A2)
    seq/<variant>/model_<fit>.pkl     a sequence model (C1, C1e, ...)
    examples/calibration.pkl          optional: the shared calibration layer per model
    examples/calendar.csv             optional: week_idx, season_week, season_idx
    examples/meta.json                optional: fit, origin_week, known_window, models, synthetic

``notebooks/55_examples`` writes the ``examples/`` files, so the artifact folder of a real
run is a bundle as it stands. ``train_demo`` builds one from synthetic data.

The player being designed always ends "now", at the bundle's origin week: the week the fit
was built to be scored at. Every number shown comes from the same code the pipeline runs
(``ltv.eval.examples``): features are rebuilt from the history, and nothing after a week is
read when scoring that week.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ltv.data.sequences import CUM_HORIZONS, SequenceStore
from ltv.eval import examples as ex
from ltv.models.base import TARGETS

FIT_ORDER = ("final_1", "final_2", "tune")
ALL_MODELS = ("A1", "A2", "C1", "C1e", "C2")
MODEL_LABELS = {"A1": "Trees, totals only (A1)", "A2": "Trees, coherent (A2)", "C1": "GRU (C1)",
                "C1e": "GRU on play spells (C1e)", "C2": "Transformer (C2)", "A1tw": "Trees, Tweedie spend (A1tw)"}
HORIZON = CUM_HORIZONS[-1]
MAX_HOURS = 168.0
LOOKBACK = 104                     # the population the models were trained on: active or spent in the last 104 weeks
DEFAULT_ORIGIN, DEFAULT_SEASON = 260, 9


@dataclass
class Bundle:
    path: Path
    fit: str
    scorers: dict
    recal: dict
    season_week: np.ndarray
    season_idx: np.ndarray
    origin: int
    known_window: object = "current_season"
    meta: dict = field(default_factory=dict)

    @property
    def max_weeks(self) -> int:
        return self.origin + 1


def _default_calendar(n_weeks: int, season: int = DEFAULT_SEASON) -> tuple[np.ndarray, np.ndarray]:
    w = np.arange(n_weeks)
    return (w % season + 1).astype(np.int16), (w // season).astype(np.int16)


def extend_calendar(season_week: np.ndarray, season_idx: np.ndarray, n_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """The calendar out to ``n_weeks``, continuing with seasons of the median length seen so far."""
    sw, si = np.asarray(season_week).astype(int), np.asarray(season_idx).astype(int)
    if len(sw) >= n_weeks:
        return sw[:n_weeks].astype(np.int16), si[:n_weeks].astype(np.int16)
    starts = np.flatnonzero(sw == 1)
    length = int(np.median(np.diff(starts))) if len(starts) > 2 else DEFAULT_SEASON
    sw, si = list(sw), list(si)
    while len(sw) < n_weeks:
        new = sw[-1] >= length
        sw.append(1 if new else sw[-1] + 1)
        si.append(si[-1] + 1 if new else si[-1])
    return np.asarray(sw, np.int16), np.asarray(si, np.int16)


def find_fit(path: Path) -> str | None:
    for fit in FIT_ORDER:
        if (path / "trees" / f"suite_{fit}.pkl").exists() or list(path.glob(f"seq/*/model_{fit}.pkl")):
            return fit
    return None


def load_bundle(path, fit: str | None = None, models=None, calibrated: bool = True) -> Bundle:
    """Load every saved model under ``path``. Raises FileNotFoundError when there is none."""
    from ltv.train.timebox import load_pickle
    path = Path(path)
    meta_file = path / "examples" / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    fit = fit or meta.get("fit") or find_fit(path)
    known = meta.get("known_window", "current_season")
    scorers = ex.saved_scorers(path, fit, list(models or ALL_MODELS), known) if fit else {}
    if not scorers:
        raise FileNotFoundError(
            f"no saved models under {path}. Expected trees/suite_<fit>.pkl or seq/<variant>/model_<fit>.pkl. "
            "Run `python -m ltv.playground --train-demo` for a synthetic demo, or copy the fits from a real run.")
    origin = int(meta.get("origin_week", DEFAULT_ORIGIN))
    cal_file = path / "examples" / "calendar.csv"
    if cal_file.exists():
        cal = pd.read_csv(cal_file).sort_values("week_idx")
        sw, si = cal["season_week"].to_numpy(), cal["season_idx"].to_numpy()
    else:
        sw, si = _default_calendar(origin + 1)
    sw, si = extend_calendar(sw, si, origin + 1 + HORIZON)
    recal = (load_pickle(path / "examples" / "calibration.pkl", {}) or {}) if calibrated else {}
    return Bundle(path, fit, scorers, {m: r for m, r in recal.items() if m in scorers}, sw, si, origin, known, meta)


# ---------------------------------------------------------------------------
# One designed player
# ---------------------------------------------------------------------------

def clean_history(hours, spend, max_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    """Weekly hours and spend from the player's first week to now, as the data would hold them.

    Negative and non-finite values become 0, hours are capped at 168 a week, weeks before the
    first week with any row are dropped (a player's history starts when they first appear),
    and only the latest ``max_weeks`` are kept.
    """
    h = np.nan_to_num(np.asarray(hours, float), nan=0.0, posinf=0.0, neginf=0.0)
    s = np.nan_to_num(np.asarray(spend if spend is not None else [], float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(s) < len(h):
        s = np.concatenate([np.zeros(len(h) - len(s)), s])
    s = s[len(s) - len(h):]
    h, s = np.clip(h, 0.0, MAX_HOURS), np.maximum(s, 0.0)
    present = np.flatnonzero((h > 0) | (s > 0))
    if not len(present):
        raise ValueError("The history is empty. Give the player at least one week with hours or spend.")
    h, s = h[present[0]:], s[present[0]:]
    return h[-max_weeks:], s[-max_weeks:]


def make_store(bundle: Bundle, hours, spend=None) -> SequenceStore:
    """A one-player SequenceStore whose last week is the bundle's origin."""
    h, s = clean_history(hours, spend, bundle.max_weeks)
    w, first = bundle.origin + 1, bundle.origin + 1 - len(h)
    H, S = np.zeros((1, w), np.float32), np.zeros((1, w), np.float32)
    H[0, first:], S[0, first:] = h, s
    row = ((H > 0) | (S > 0)).astype(np.int16)                # platform 1 / country 1 wherever there is a row
    return SequenceStore(np.array(["designed"]), np.array(["test"]), np.array([first], np.int32), H, S, row, row.copy(),
                         bundle.season_week, bundle.season_idx)


def _status(store: SequenceStore, origin: int) -> dict:
    h, s = store.hours[0, : origin + 1], store.spend[0, : origin + 1]
    active = np.flatnonzero(h > 0)
    any_row = np.flatnonzero((h > 0) | (s > 0))
    since = int(origin - active[-1]) if len(active) else None
    return {"weeks_of_history": int(origin + 1 - store.first_week[0]),
            "weeks_since_active": since,
            "in_population": bool(len(any_row) and origin - any_row[-1] < LOOKBACK)}


def forecast(bundle: Bundle, hours, spend=None) -> dict:
    """Every model's view of the next 52 weeks, as of now: the five totals and the weekly path."""
    store = make_store(bundle, hours, spend)
    rows, out = np.array([0]), {}
    for m, score in bundle.scorers.items():
        raw = score(store, rows, bundle.origin)
        p = ex._calibrate(raw, bundle.recal.get(m))
        path = ex.path_rows(p, m)
        out[m] = {"totals": {t: float(getattr(p, t)[0]) for t in TARGETS},
                  "path": {k: [float(x) for x in path[k].to_numpy()] for k in ("hours", "active", "spend")},
                  "has_shape": raw.cum_hours is not None, "calibrated": m in bundle.recal}
    return {"models": out, **_status(store, bundle.origin)}


def timeline(bundle: Bundle, hours, spend=None, weeks: int = 104, step: int = 1) -> dict:
    """Each model's opinion as it stood at the end of earlier weeks: a fresh score of the
    history up to that week, every ``step`` weeks over the last ``weeks`` weeks, ending now.

    ``offsets`` are weeks relative to now (0 = now, -4 = four weeks ago).
    """
    store = make_store(bundle, hours, spend)
    origin, first = bundle.origin, int(store.first_week[0])
    lo = max(first, origin - int(weeks))
    origins = list(range(origin, lo - 1, -max(int(step), 1)))[::-1]
    rows, out, frames = np.array([0]), {}, {}
    for m, score in bundle.scorers.items():
        if hasattr(score, "predict"):                      # trees: one stacked frame, one prediction
            if score.frame not in frames:
                frames[score.frame] = pd.concat([score.frame(store, rows, int(t)) for t in origins], ignore_index=True)
            p = ex._calibrate(score.predict(frames[score.frame], origin), bundle.recal.get(m))
            out[m] = {t: [float(x) for x in getattr(p, t)] for t in TARGETS}
        else:
            cols = {t: [] for t in TARGETS}
            for t_week in origins:
                p = ex._calibrate(score(store, rows, int(t_week)), bundle.recal.get(m))
                for t in TARGETS:
                    cols[t].append(float(getattr(p, t)[0]))
            out[m] = cols
    return {"offsets": [int(t - origin) for t in origins], "models": out}


def calendar_marks(bundle: Bundle, n_history: int) -> dict:
    """Season launches as week offsets from now: real ones in the history, projected ones ahead."""
    from ltv.data.splits import project_calendar
    o = bundle.origin
    past = [int(w - o) for w in range(max(o - n_history + 1, 0), o + 1) if bundle.season_week[w] == 1]
    sw, known = project_calendar(bundle.season_week, bundle.season_idx, o, HORIZON, bundle.known_window)
    return {"past": past, "future": [int(i + 1) for i in np.flatnonzero(sw == 1)],
            "future_known": [int(i + 1) for i in np.flatnonzero((sw == 1) & known.astype(bool))]}


# ---------------------------------------------------------------------------
# Simulated players and presets
# ---------------------------------------------------------------------------

WORLDS = {"rhythm": "Personal rhythm (spells and lapses of personal length)",
          "markov": "Markov (engaged / casual / dormant states, four player types)"}
WANTS = {"any": "Any player", "playing": "Playing now", "lapsed": "Lapsed 13+ weeks", "returner": "Has come back from a long lapse"}


def _returns(h: np.ndarray, gap: int = 26) -> int:
    """How many times the player came back after ``gap`` or more quiet weeks."""
    active = np.flatnonzero(h > 0)
    return int((np.diff(active) > gap).sum()) if len(active) > 1 else 0


def simulate(bundle: Bundle, world: str = "rhythm", want: str = "any", seed: int | None = None, n: int = 600) -> dict:
    """One simulated player from a synthetic world: history up to now, and the 52 weeks that followed.

    The future is one draw of a random process, not an expectation: a model that is right on
    average will still miss any single player. ``want`` filters for a kind of player.
    """
    from ltv.synth import generator, worlds
    if world not in WORLDS:
        raise ValueError(f"unknown world {world!r}; choose from {sorted(WORLDS)}")
    seed = int(np.random.SeedSequence().entropy % 2**31) if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    o, n_weeks = bundle.origin, bundle.origin + 1 + HORIZON
    sw, _ = extend_calendar(bundle.season_week, bundle.season_idx, n_weeks)
    sim = (worlds.simulate_rhythm if world == "rhythm" else generator.simulate)(n, n_weeks, sw.astype(int), rng)
    H, S, first = sim["hours"], sim["spend"], sim["first_week"]
    recent = ((H[:, max(o - LOOKBACK + 1, 0): o + 1] > 0) | (S[:, max(o - LOOKBACK + 1, 0): o + 1] > 0)).any(axis=1)
    ok = np.flatnonzero((first <= o - 12) & recent)
    if not len(ok):
        raise RuntimeError("the simulation produced no player in the population; try again")
    quiet = np.array([o - np.flatnonzero(H[i, : o + 1] > 0)[-1] if (H[i, : o + 1] > 0).any() else 999 for i in ok])
    rule = {"any": np.ones(len(ok), bool), "playing": quiet <= 1, "lapsed": quiet >= 13,
            "returner": np.array([_returns(H[i, : o + 1]) >= 1 for i in ok])}.get(want)
    if rule is None:
        raise ValueError(f"unknown kind {want!r}; choose from {sorted(WANTS)}")
    pool = ok[rule] if rule.any() else ok
    i = int(rng.choice(pool))
    f = int(first[i])
    r = lambda a: [round(float(x), 2) for x in a]
    return {"hours": r(H[i, f: o + 1]), "spend": r(S[i, f: o + 1]), "future_hours": r(H[i, o + 1:]),
            "future_spend": r(S[i, o + 1:]), "world": world, "want": want, "seed": seed,
            "matched": bool(rule.any()), "returns": _returns(H[i, : o + 1])}


def presets(max_weeks: int) -> list[dict]:
    """A few hand-built histories to start from. Hours only; no spend."""
    rng = np.random.default_rng(12)

    def play(n, level, skip=0.12):
        h = np.round(np.clip(rng.lognormal(np.log(level), 0.45, n), 0.3, 60), 1)
        h[rng.random(n) < skip] = 0.0
        if n:
            h[0] = max(h[0], round(level, 1))
        return h

    quiet = lambda n: np.zeros(n)
    fade = np.round(8 * np.linspace(1.0, 0.15, 14) * rng.uniform(0.8, 1.2, 14), 1)
    made = [
        ("steady", "Steady player", [play(150, 8)]),
        ("lapse_return_lapse", "Lapsed 6 months, came back, lapsed again", [play(60, 7), quiet(26), play(30, 6), quiet(26), play(18, 5), quiet(20)]),
        ("serial_returner", "Serial returner, away now", [play(14, 6), quiet(30), play(10, 6), quiet(34), play(12, 5), quiet(28), play(9, 5), quiet(30), play(8, 5), quiet(22)]),
        ("fading", "Hours fading", [play(120, 8), fade]),
        ("new", "New player (8 weeks)", [play(8, 6, skip=0.0)]),
        ("gone_a_year", "Gone a year", [play(90, 7), quiet(52)]),
        ("back_this_week", "Back this week after 9 months", [play(70, 7), quiet(39), play(1, 5, skip=0.0)]),
    ]
    out = []
    for key, name, parts in made:
        h = np.concatenate(parts)[-max_weeks:]
        out.append({"id": key, "name": name, "hours": [float(x) for x in h], "spend": [0.0] * len(h)})
    return out


def info(bundle: Bundle) -> dict:
    meta = bundle.meta
    return {"models": [{"id": m, "label": MODEL_LABELS.get(m, m), "calibrated": m in bundle.recal} for m in bundle.scorers],
            "fit": bundle.fit, "origin_week": bundle.origin, "max_weeks": bundle.max_weeks, "horizon": HORIZON,
            "synthetic": bool(meta.get("synthetic", False)), "world": meta.get("world"), "trained_users": meta.get("users"),
            "bundle": str(bundle.path), "worlds": WORLDS, "wants": WANTS, "presets": presets(bundle.max_weeks),
            "calendar": calendar_marks(bundle, bundle.max_weeks), "lookback": LOOKBACK}


# ---------------------------------------------------------------------------
# A demo bundle from synthetic data
# ---------------------------------------------------------------------------

def train_demo(out, world: str = "rhythm", users: int = 60_000, rows: int = 400_000, rounds: int = 200,
               seq_steps: int = 3000, seed: int = 7, keep_world: bool = False, log=print) -> Path:
    """Train small tree and GRU models on SYNTHETIC players and save them as a bundle under ``out``.

    It is the simulation study in miniature: one fit per family, trained on the plan's
    ``final_1`` origins, with early stopping and the calibration layer fitted on 'val' users
    at the last training origin. The result shows how the models behave. It says nothing
    about real players and is never a model comparison (AGENTS.md, rule 3).
    """
    import shutil
    import time

    from ltv.config import load_config
    from ltv.eval.calibration import Recalibrator
    from ltv.models.base import LABEL_OF
    from ltv.models.seq import SeqConfig, predict_seq
    from ltv.models.trees import DEFAULT_PARAMS, TreeSuite
    from ltv.pipeline import SEQ_VARIANTS
    from ltv.sim import Sim, SimSettings
    from ltv.synth import worlds
    from ltv.train.seq_fit import fit_seq, prepare_config, validation_batches
    from ltv.train.timebox import TimeBox, save_pickle
    conf = Path(__file__).resolve().parents[3] / "conf"
    if not (conf / "bakeoff.yml").exists():
        raise RuntimeError("train_demo needs the repo checkout (conf/bakeoff.yml); install with `pip install -e .`")
    out, t0 = Path(out), time.time()
    world_dir = out / "_world"
    shard_users = min(users, 100_000)
    log(f"[demo] simulating {users:,} {world} players x 320 weeks")
    worlds.write_world(world, users, world_dir, n_weeks=320, seed=seed, shard_users=shard_users)
    cfg = load_config(conf / "bakeoff.yml", conf / "synth.yml", overrides={"artifact_path": str(world_dir)})
    n_shards = len(list((world_dir / "shards").glob("shard_*.npz")))
    sim = Sim(world_dir, cfg, SimSettings(eval_shards=n_shards, tab_train_shards=n_shards, tab_train_rows=rows,
                                          tab_rounds=rounds, seq_train_shards=n_shards, seq_steps=seq_steps))
    fit, slot = "final_1", "final_1@cal"
    last = int(sim.sched.train_origins(fit)[-1])
    log(f"[demo] building features ({rows:,} training rows)")
    tr, va = sim.train_frame(fit), sim.score_frame(slot)          # 'val' users at the last training origin
    stop, cal = (np.arange(len(va)) % 2 == 0), (np.arange(len(va)) % 2 == 1)
    log(f"[demo] fitting 24 tree models on {len(tr):,} rows")
    suite = TreeSuite(params=dict(DEFAULT_PARAMS), max_rounds=rounds, seed=seed)
    suite.fit(tr, va[stop])
    save_pickle(out / "trees" / f"suite_{fit}.pkl", suite.state())
    s = cfg["seq"]
    scfg = prepare_config(SeqConfig(d=int(s["d"]), layers=int(s["layers"]), **SEQ_VARIANTS["C1"]), sim._store0,
                          sim.sched.train_origins(fit), sim.known, seed, lookback=sim.lookback)
    val = validation_batches(sim._store0, last, scfg, sim.known, lookback=sim.lookback)
    log(f"[demo] training the GRU (up to {seq_steps:,} steps)")
    gru, info_ = fit_seq(scfg, sim.shards, sim.sched.train_origins(fit), TimeBox(600), world_dir / "gru.pt", val=val,
                         max_steps=seq_steps, eval_every=int(s["eval_every"]), patience=int(s["patience"]),
                         lr_drops=1, **sim._seq_kw())
    save_pickle(out / "seq" / "C1" / f"model_{fit}.pkl", gru.state())
    log("[demo] fitting the calibration layer")
    preds = {"A1": suite.predict_a1(va, last), "A2": suite.predict_a2(va, last),
             "C1": sim.predict_shards(slot, lambda st, r, o: predict_seq(gru, st, r, o, sim.known))}
    recal = {}
    for m, p in preds.items():
        assert (np.asarray(p.user_ids) == va["user_id"].to_numpy()).all()
        f = pd.DataFrame({t: getattr(p, t) for t in TARGETS} | {LABEL_OF[t]: va[LABEL_OF[t]].to_numpy(float) for t in TARGETS})
        recal[m] = Recalibrator().fit(f[cal])
    ex_dir = out / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(ex_dir / "calibration.pkl", recal)
    st = sim._store0
    pd.DataFrame({"week_idx": np.arange(len(st.season_week)), "season_week": st.season_week,
                  "season_idx": st.season_idx}).to_csv(ex_dir / "calendar.csv", index=False)
    (ex_dir / "meta.json").write_text(json.dumps({
        "fit": fit, "origin_week": int(sim.sched.score_origin(fit)), "known_window": sim.known, "models": list(preds),
        "synthetic": True, "world": world, "users": int(users), "train_rows": int(len(tr)),
        "gru_steps": int(info_["best_step"]), "source": "ltv.playground.train_demo"}))
    if not keep_world:
        shutil.rmtree(world_dir, ignore_errors=True)
    log(f"[demo] bundle written to {out} in {time.time() - t0:.0f} s")
    return out
