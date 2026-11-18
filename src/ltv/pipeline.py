"""Phase-level entry points. Notebooks call these; tests call the same functions.

Keeping the notebooks thin means the code that runs in Databricks is the code that was
tested locally. Each function takes a SparkSession and a Config and writes named tables.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ltv.config import Config
from ltv.data import sqlrun
from ltv.data.np_features import LAPSE_THRESHOLDS, WINDOWS
from ltv.data.sequences import CUM_HORIZONS
from ltv.data.splits import Schedule, projected_calendar_frame, schedule_from_config

AUDIT_RAW = ("a01_weekly_row_counts", "a02_row_semantics", "a03_spend_distribution",
             "a04_duplicate_user_weeks", "a05_week_alignment", "a06_history_and_users",
             "a07_suspect_accounts", "a08_seasons")
AUDIT_CLEAN = ("a09_dormancy_base_rates", "a10_lapse_and_return")


def _check_fixed_lists(cfg: Config) -> None:
    """The SQL writes these lists out by hand; refuse to run if the config disagrees."""
    if tuple(cfg["feature_windows"]) != WINDOWS:
        raise ValueError(f"feature_windows must be {WINDOWS}: they are fixed in 07_features_tabular.sql")
    if tuple(cfg["lapse_thresholds"]) != LAPSE_THRESHOLDS:
        raise ValueError(f"lapse_thresholds must be {LAPSE_THRESHOLDS}")
    if tuple(cfg["cum_horizons"]) != CUM_HORIZONS:
        raise ValueError(f"cum_horizons must be {CUM_HORIZONS}: they are fixed in 08_labels.sql")
    if int(cfg["horizon_weeks"]) != 52:
        raise ValueError("horizon_weeks must be 52: label column names carry the horizon")


def _clean_params(cfg: Config) -> dict:
    excl = cfg["source"].get("exclude_users_table")
    join = f"LEFT ANTI JOIN {excl} e ON r.user_id = CAST(e.user_id AS STRING)" if excl else ""
    flt = "TRUE" if cfg["cleaning"]["negative_spend"] == "clip" else "r.spend_raw >= 0"
    return {"sql_exclude_join": join, "sql_negative_spend_filter": flt}


def last_data_week(spark, cfg: Config) -> int:
    row = spark.sql(f"SELECT MAX(week_idx) AS w FROM {cfg.table('calendar_weeks')} WHERE has_data").first()
    return int(row["w"])


def get_schedule(spark, cfg: Config) -> Schedule:
    return schedule_from_config(last_data_week(spark, cfg), cfg)


def build_clean(spark, cfg: Config) -> dict:
    """Phase 1, step 1: calendar, clean sampled activity, dimensions, users."""
    _check_fixed_lists(cfg)
    sqlrun.ensure_schema(spark, cfg)
    t0 = time.time()
    sqlrun.build(spark, cfg, "build/01_calendar_weeks", "calendar_weeks")
    sqlrun.build(spark, cfg, "build/02_activity_clean", "activity_clean",
                 cluster_by=["user_id", "week_idx"], **_clean_params(cfg))
    sqlrun.build(spark, cfg, "build/03_dim_platform", "dim_platform")
    sqlrun.build(spark, cfg, "build/04_dim_country", "dim_country")
    sqlrun.build(spark, cfg, "build/05_users", "users")
    return {"seconds": round(time.time() - t0, 1), "last_data_week": last_data_week(spark, cfg)}


def build_splits(spark, cfg: Config) -> Schedule:
    """Phase 1e: the origin schedule and the projected calendar for every origin."""
    sched = get_schedule(spark, cfg)
    sqlrun.write_table(spark, cfg, spark.createDataFrame(sched.to_frame()), "origins")
    cal = spark.table(cfg.table("calendar_weeks")).orderBy("week_idx").toPandas()
    proj = projected_calendar_frame(cal["season_week"].to_numpy(), cal["season_idx"].to_numpy(),
                                    sched.all_origins(), int(cfg["horizon_weeks"]),
                                    cfg["calendar"]["known_window"])
    sqlrun.write_table(spark, cfg, spark.createDataFrame(proj), "calendar_projected")
    n_train = max(len(set(sched.tune_train) | set(sched.final_train_1) | set(sched.final_train_2)), 1)
    keep = min(1.0, float(cfg["splits"]["max_train_origins_per_user"]) / n_train)
    sqlrun.build(spark, cfg, "build/06_user_origins", "user_origins",
                 cluster_by=["origin_week", "user_id"],
                 train_origin_keep_permille=int(round(1000 * keep)))
    return sched


def build_features(spark, cfg: Config, with_panel: bool = True) -> dict:
    """Phases 1a-1d: tabular features, labels, user sequences and (optionally) the dense panel."""
    t0 = time.time()
    sqlrun.build(spark, cfg, "build/07_features_tabular", "features_tabular",
                 cluster_by=["origin_week", "user_id"])
    sqlrun.build(spark, cfg, "build/08_labels", "labels", cluster_by=["origin_week", "user_id"])
    sqlrun.build(spark, cfg, "build/09_user_sequences", "user_sequences")
    if with_panel:
        sqlrun.build(spark, cfg, "build/10_panel_weekly", "panel_weekly",
                     cluster_by=["user_id", "week_idx"])
    return {"seconds": round(time.time() - t0, 1)}


def build_all(spark, cfg: Config, with_panel: bool = True) -> dict:
    info = build_clean(spark, cfg)
    sched = build_splits(spark, cfg)
    info.update(build_features(spark, cfg, with_panel))
    info["schedule"] = sched
    return info


# ---------------------------------------------------------------------------
# Phase 0 audit
# ---------------------------------------------------------------------------

def run_audit(spark, cfg: Config, include_clean: bool = True) -> dict[str, pd.DataFrame]:
    """Run every audit query and return the results as pandas DataFrames.

    The a01-a08 checks read the RAW source tables. a09 and a10 read the clean tables, so
    ``build_clean`` must have run first when ``include_clean`` is true.
    """
    out = {}
    for name in AUDIT_RAW + (AUDIT_CLEAN if include_clean else ()):
        out[name] = sqlrun.select(spark, cfg, f"audit/{name}").toPandas()
    return out


def audit_findings(audit: dict[str, pd.DataFrame], min_weeks: int = 312) -> list[dict]:
    """Turn audit results into a list of findings with a status: 'ok', 'warn' or 'fail'."""
    f: list[dict] = []

    def add(check: str, status: str, detail: str) -> None:
        f.append({"check": check, "status": status, "detail": detail})

    w = audit["a01_weekly_row_counts"]
    bad = w[w["suspect_outage"]]
    add("weekly row counts", "warn" if len(bad) else "ok",
        f"{len(bad)} week(s) under half the trailing median: {', '.join(map(str, bad['week_start'].tolist()[:10]))}"
        if len(bad) else "no week under half the trailing median")
    r = audit["a02_row_semantics"].iloc[0]
    for col, label in (("zero_zero_rows", "rows with zero hours and zero spend"),
                       ("negative_spend_rows", "rows with negative spend"),
                       ("null_platform_rows", "rows with no platform"),
                       ("null_country_rows", "rows with no country"),
                       ("over_max_hours_rows", "rows above cleaning.max_weekly_hours")):
        add(label, "warn" if r[col] > 0 else "ok", f"{int(r[col])} of {int(r['n_rows'])} rows")
    add("spend-only weeks", "ok", f"{r['spend_only_share']:.4%} of rows are spend-only weeks")
    d = audit["a04_duplicate_user_weeks"].iloc[0]
    add("duplicate user-weeks", "warn" if d["duplicate_user_weeks"] > 0 else "ok",
        f"{int(d['duplicate_user_weeks'])} user-weeks have more than one row; "
        f"{int(d['multi_platform_user_weeks'])} have more than one platform")
    a = audit["a05_week_alignment"]
    off = int(a["n_rows_not_in_calendar"].sum())
    add("week alignment", "fail" if len(a) > 1 or off > 0 else "ok",
        f"{len(a)} distinct weekday(s) for week_start; {off} rows are not on a calendar week")
    h = audit["a06_history_and_users"].iloc[0]
    add("weeks of history", "fail" if h["weeks_of_history"] < min_weeks else "ok",
        f"{int(h['weeks_of_history'])} weeks ({h['first_week']} to {h['last_week']}); "
        f"{int(h['distinct_weeks'])} distinct weeks present")
    add("missing weeks", "warn" if h["distinct_weeks"] < h["weeks_of_history"] else "ok",
        f"{int(h['weeks_of_history'] - h['distinct_weeks'])} week(s) have no rows at all")
    add("data starts at launch", "warn" if h["share_users_first_seen_in_first_week"] > 0.3 else "ok",
        f"{h['share_users_first_seen_in_first_week']:.1%} of users are first seen in the first data week")
    add("user counts", "ok", f"{int(h['n_users_all_time'])} users all-time; "
        f"{int(h['n_users_active_last_104w'])} with a row in the last 104 weeks")
    s = audit["a07_suspect_accounts"]
    add("suspect accounts", "warn" if len(s) else "ok",
        f"{len(s)}{'+' if len(s) == 100 else ''} user(s) exceed the weekly hours limit and will be dropped")
    se = audit["a08_seasons"]
    done = se[~se["starts_after_last_data_week"]]
    gaps = int((se["n_weeks"] != se["max_season_week"]).sum())
    add("season lengths", "warn" if gaps else "ok",
        f"{len(se)} seasons; median length {done['n_weeks'].median():.0f} weeks "
        f"(min {done['n_weeks'].min()}, max {done['n_weeks'].max()}); {gaps} season(s) with gaps; "
        f"calendar runs {int(se['weeks_known_ahead'].iloc[0])} weeks past the data")
    return f


def write_audit_report(audit: dict[str, pd.DataFrame], findings: list[dict], path: str | Path,
                       title: str = "Phase 0 data audit") -> Path:
    """Write the audit as Markdown (reports/00_data_audit.md)."""
    lines = [f"# {title}", "", "Generated by `ltv.pipeline.run_audit`. Stop here for review before any modeling.",
             "", "## Findings", "", "| Check | Status | Detail |", "| --- | --- | --- |"]
    lines += [f"| {x['check']} | {x['status'].upper()} | {x['detail']} |" for x in findings]
    for name, df in audit.items():
        lines += ["", f"## {name}", ""]
        show = df if len(df) <= 60 else pd.concat([df.head(30), df.tail(30)])
        if name == "a01_weekly_row_counts" and len(df) > 60:
            lines += [f"Showing the first and last 30 of {len(df)} weeks, plus any suspect weeks.", ""]
            show = pd.concat([show, df[df["suspect_outage"]]]).drop_duplicates().sort_values("week_start")
        lines += [show.to_markdown(index=False, floatfmt=".4g")]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Phase 2: candidates. Each run_* function is resumable and writes predictions.
# ---------------------------------------------------------------------------

def _profile(cfg: Config) -> str:
    return cfg["budget"]["profile"]


def _box(cfg: Config):
    from ltv.train.timebox import TimeBox
    return TimeBox(float(cfg["budget"]["step_minutes"]))


def frames_for_fit(spark, cfg: Config, sched: Schedule, fit: str, need_train: bool = True):
    """Training, early-stopping and scoring frames for one fit.

    train : 'train'-split users at the fit's training origins
    val   : 'val'-split users at the validation origin ('tune' fit only; else None)
    score : every user in the population at the origin the fit is scored on
    """
    from ltv.models.base import load_xy
    tr = load_xy(spark, cfg, sched.train_origins(fit), "train",
                 int(cfg["training"]["max_tabular_rows"])) if need_train else None
    va = load_xy(spark, cfg, [sched.validation], "val") if fit == "tune" else None
    sc = load_xy(spark, cfg, [sched.score_origin(fit)])
    return tr, va, sc


def run_baselines(spark, cfg: Config) -> dict:
    """R1 segment lookup, R2 persistence and R3 BG/BB, for all three fits."""
    from ltv.models.base import FITS, write_predictions
    from ltv.models.baselines import Persistence, SegmentLookup
    from ltv.models.bgbb import BGBB
    sched = get_schedule(spark, cfg)
    frames, info = [], {}
    for fit in FITS:
        tr, _, sc = frames_for_fit(spark, cfg, sched, fit)
        origin = sched.score_origin(fit)
        for model in (SegmentLookup(), Persistence(), BGBB()):
            model.fit(tr)
            frames.append(model.predict(sc, origin).frame(model.model_id, fit))
            if isinstance(model, BGBB):
                info[f"bgbb_{fit}"] = {"activity": model.p_act.round(4).tolist(),
                                       "spend": model.p_spd.round(4).tolist()}
    info["rows"] = write_predictions(spark, cfg, frames)
    return info


def run_trees(spark, cfg: Config, step: str) -> dict:
    """Candidate A in four resumable steps, one notebook command each:

    'tune'    time-boxed random search for one shared hyperparameter set (call again until complete)
    'fit'     fit every model at the tuned settings, freeze tree counts, score the validation origin
    'final_1' refit with frozen settings and score test_1
    'final_2' the same for test_2
    """
    from ltv.models.base import write_predictions
    from ltv.models.trees import DEFAULT_PARAMS, SEARCH_SPACE, TreeSuite, tuning_objective
    from ltv.train.timebox import TrialLog, load_pickle, random_search, save_pickle
    sched = get_schedule(spark, cfg)
    art = cfg.artifact_dir("trees")
    quick = _profile(cfg) == "quick"
    max_rounds, n_trials = (300, 4) if quick else (2000, 30)
    seed = int(cfg["training"]["seed"])
    if step == "tune":
        tr, va, _ = frames_for_fit(spark, cfg, sched, "tune")
        base = {}
        for r in TrialLog(art / "tuning.jsonl").read():          # restore the baseline losses
            if r["trial"] == 0:
                base.update(r["info"]["losses"])
        best = random_search(SEARCH_SPACE, tuning_objective(tr, va, max_rounds, 50, seed, base),
                             n_trials, art / "tuning.jsonl", _box(cfg), seed, defaults={})
        return {"best_params": best["params"], "best_loss": best["loss"], "trials_done": best["n_done"],
                "complete": best["complete"], "train_rows": len(tr)}
    if step == "fit":
        trials = TrialLog(art / "tuning.jsonl").read()
        params = dict(DEFAULT_PARAMS)
        if trials:
            params.update(min(trials, key=lambda r: r["loss"])["params"])
        tr, va, sc = frames_for_fit(spark, cfg, sched, "tune")
        suite = TreeSuite(params=params, max_rounds=max_rounds, seed=seed)
        losses = suite.fit(tr, va)
        save_pickle(art / "suite_tune.pkl", suite.state())
        fit, origin = "tune", sched.validation
    elif step in ("final_1", "final_2"):
        st = load_pickle(art / "suite_tune.pkl")
        if st is None:
            raise RuntimeError("run_trees(step='fit') must finish before the final fits")
        tr, _, sc = frames_for_fit(spark, cfg, sched, step)
        suite = TreeSuite(params=st["params"], max_rounds=st["max_rounds"], seed=seed, rounds=dict(st["rounds"]))
        losses = suite.fit(tr, None)
        save_pickle(art / f"suite_{step}.pkl", suite.state())
        fit, origin = step, sched.score_origin(step)
    else:
        raise ValueError(f"unknown step {step!r}")
    frames = [suite.predict_a1(sc, origin).frame("A1", fit),
              suite.predict_a1(sc, origin, tweedie_spend=True).frame("A1tw", fit),
              suite.predict_a2(sc, origin).frame("A2", fit)]
    return {"rows": write_predictions(spark, cfg, frames), "train_rows": len(tr),
            "rounds": dict(suite.rounds), "val_losses": losses}


def export_sequences(spark, cfg: Config) -> list:
    """Write the user sample as NumPy shards (the input of the HMM and the sequence models)."""
    from ltv.data.sequences import export_shards
    return export_shards(spark, cfg, cfg.artifact_dir("shards"))


def shard_paths(cfg: Config) -> list:
    paths = sorted(Path(cfg["artifact_path"]).joinpath("shards").glob("shard_*.npz"))
    if not paths:
        raise RuntimeError("no sequence shards found; run pipeline.export_sequences first")
    return paths


def _predict_over_shards(cfg: Config, origin: int, predict_fn) -> list:
    """Call ``predict_fn(store, rows)`` for the population of every shard at ``origin``."""
    from ltv.data.sequences import SequenceStore
    out = []
    for p in shard_paths(cfg):
        store = SequenceStore.load(p)
        rows = store.population(origin, int(cfg["population_lookback_weeks"]))
        if len(rows):
            out.append(predict_fn(store, rows))
    return out


def _hmm_candidates(cfg: Config) -> list:
    from ltv.models.hmm import spec_b1, spec_b2
    h, quick = cfg["hmm"], _profile(cfg) == "quick"
    nb = len(cfg["calendar"]["season_week_buckets"]) + 1
    b1 = h["b1_states"][:1] if quick else h["b1_states"]
    b2 = h["b2_types"][:1] if quick else h["b2_types"]
    return ([("B1", spec_b1(int(k), nb)) for k in b1]
            + [("B2", spec_b2(int(t), int(h["b2_active_states"]), int(h["b2_dormant_states"]),
                              int(h["b2_spend_classes"]), nb)) for t in b2])


def run_hmm(spark, cfg: Config, step: str) -> dict:
    """Candidate B in resumable steps, one notebook command each:

    'tune'    EM for every candidate size on weeks <= the validation origin; records the
              validation log-likelihood of each. Call again until ``complete`` is true.
    'score'   pick the best B1 and the best B2 by validation log-likelihood, score the validation origin
    'final_1' refit the chosen models on weeks <= test_1 (warm-started), score test_1, save states
    'final_2' the same for test_2
    """
    from ltv.models.base import write_predictions
    from ltv.models.hmm import HMM, fit_hmm, loglik_per_week, predict_hmm, state_profile
    from ltv.train.timebox import load_pickle, save_pickle
    sched = get_schedule(spark, cfg)
    art, shards, box = cfg.artifact_dir("hmm"), shard_paths(cfg), _box(cfg)
    edges = tuple(cfg["calendar"]["season_week_buckets"])
    kw = dict(edges=edges, batch=int(cfg["hmm"]["batch_users"]), seed=int(cfg["training"]["seed"]),
              max_iters=15 if _profile(cfg) == "quick" else int(cfg["hmm"]["max_iters"]))
    known = cfg["calendar"]["known_window"]
    cands = _hmm_candidates(cfg)
    sel_path = art / "selection.pkl"
    sel = load_pickle(sel_path, {})
    if step == "tune":
        for family, spec in cands:
            if spec.name in sel or box.expired():
                continue
            model, info = fit_hmm(spec, shards, sched.validation, box, art / f"tune_{spec.name}.pkl", **kw)
            if info["complete"]:
                sel[spec.name] = {"family": family, "iters": info["iters"], "train_ll": info["train_ll_per_week"],
                                  "val_ll": loglik_per_week(model, shards, sched.validation, "val", edges),
                                  "params": model.export()}
                save_pickle(sel_path, sel)
        return {"complete": len(sel) == len(cands),
                "candidates": {k: {x: v[x] for x in ("family", "iters", "train_ll", "val_ll")} for k, v in sel.items()}}
    specs = dict((s.name, s) for _, s in cands)
    best = {}
    for name, rec in sel.items():
        if name in specs and (rec["family"] not in best or rec["val_ll"] > sel[best[rec["family"]]]["val_ll"]):
            best[rec["family"]] = name
    if not best:
        raise RuntimeError("run_hmm(step='tune') has not completed any candidate yet")
    frames, info = [], {"chosen": best}
    if step == "score":
        fit, origin = "tune", sched.validation
        models = {fam: HMM(specs[name], sel[name]["params"]) for fam, name in best.items()}
    elif step in ("final_1", "final_2"):
        fit, origin = step, sched.score_origin(step)
        models = {}
        for fam, name in best.items():
            m, i = fit_hmm(specs[name], shards, origin, box, art / f"{step}_{name}.pkl",
                           start=sel[name]["params"], **kw)
            if not i["complete"]:
                return {"complete": False, "chosen": best, "waiting_on": name, "iters": i["iters"]}
            models[fam] = m
    else:
        raise ValueError(f"unknown step {step!r}")
    states = []
    for fam, m in models.items():
        preds = _predict_over_shards(cfg, origin, lambda st, rows, m=m: predict_hmm(m, st, rows, origin, known_window=known, edges=edges))
        frames += [p.frame(fam, fit) for p in preds]
        info[f"profile_{fam}"] = state_profile(m)
        save_pickle(art / f"model_{fit}_{fam}.pkl", {"spec": m.spec, "params": m.export()})
        if fam == "B2":
            states += [pd.DataFrame({"user_id": p.user_ids, "origin_week": np.int32(origin),
                                     "alpha": [a.tolist() for a in p.state]}) for p in preds]
    info["rows"] = write_predictions(spark, cfg, frames)
    if states and fit != "tune":
        # The per-user state table that weekly scoring would carry forward (docs/DESIGN.md, "Scoring").
        sdf = spark.createDataFrame(pd.concat(states, ignore_index=True),
                                    "user_id string, origin_week int, alpha array<double>")
        fq, fmt = cfg.table("hmm_user_state"), cfg.get("table_format", "delta")
        w = sdf.write.mode("overwrite").format(fmt)
        if spark.catalog.tableExists(fq) and fmt == "delta":
            w = w.option("replaceWhere", f"origin_week = {int(origin)}")
        w.saveAsTable(fq)
    info["complete"] = True
    return info


SEQ_VARIANTS = {
    "C1": dict(encoder="gru2res"),
    "C1e": dict(encoder="episodes"),
    "C2": dict(encoder="transformer"),
    "C1mix": dict(encoder="gru2res", head="mixture"),          # head ablations
    "C1ord": dict(encoder="gru2res", head="ordinal"),
    "C1whale": dict(encoder="gru2res", whale_aux=True),
    "C1notab": dict(encoder="gru2res", use_tabular=False),      # input ablation
}


def _seq_settings(cfg: Config) -> dict:
    s = cfg["seq"]
    return dict(batch=int(s["batch_users"]), lr=float(s["lr"]), eval_every=int(s["eval_every"]),
                patience=int(s["patience"]), seed=int(cfg["training"]["seed"]),
                known_window=cfg["calendar"]["known_window"],
                lookback=int(cfg["population_lookback_weeks"]), threads=cfg["budget"].get("torch_threads"),
                max_steps=int(s["quick_max_steps"] if _profile(cfg) == "quick" else s["max_steps"]))


def run_seq(spark, cfg: Config, step: str, variant: str = "C1") -> dict:
    """Candidate C in resumable steps, one notebook command each, for one variant:

    'tune'    train with early stopping on the validation origin. Call again until ``complete``.
    'score'   score the validation origin with the best weights
    'final_1' retrain for the number of steps the tune fit chose, on final_train_1; score test_1
    'final_2' the same for test_2
    """
    from ltv.data.sequences import SequenceStore
    from ltv.models.base import write_predictions
    from ltv.models.seq import SeqConfig, predict_seq
    from ltv.train.seq_fit import fit_seq, prepare_config, validation_batches
    from ltv.train.timebox import load_pickle, save_pickle
    if variant not in SEQ_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(SEQ_VARIANTS)}")
    sched = get_schedule(spark, cfg)
    art, shards, box = cfg.artifact_dir("seq", variant), shard_paths(cfg), _box(cfg)
    kw = _seq_settings(cfg)
    known = kw["known_window"]
    scfg = load_pickle(art / "config.pkl")
    if scfg is None:
        base = SeqConfig(d=int(cfg["seq"]["d"]), layers=int(cfg["seq"]["layers"]), **SEQ_VARIANTS[variant])
        scfg = prepare_config(base, SequenceStore.load(shards[0]), sched.tune_train, known, kw["seed"],
                              lookback=kw["lookback"])
        save_pickle(art / "config.pkl", scfg)
    if step in ("tune", "score"):
        fit, origin = "tune", sched.validation
        val = validation_batches(SequenceStore.load(shards[0]), sched.validation, scfg, known,
                                 lookback=kw["lookback"]) if step == "tune" else None
        from ltv.train.timebox import TimeBox
        model, info = fit_seq(scfg, shards, sched.tune_train, box if step == "tune" else TimeBox(0.0),
                              art / "tune.pt", val=val, **kw)
        save_pickle(art / "tune_info.pkl", {k: v for k, v in info.items()})
        if step == "tune":
            return {k: v for k, v in info.items() if k != "history"} | {"last_eval": (info["history"] or [None])[-1]}
    elif step in ("final_1", "final_2"):
        tinfo = load_pickle(art / "tune_info.pkl")
        if not tinfo or not tinfo["complete"]:
            raise RuntimeError(f"run_seq(step='tune', variant={variant!r}) must complete before the final fits")
        fit, origin = step, sched.score_origin(step)
        kw2 = {k: v for k, v in kw.items() if k not in ("max_steps", "eval_every", "patience")}
        model, info = fit_seq(scfg, shards, sched.train_origins(step), box, art / f"{step}.pt",
                              fixed_steps=max(int(tinfo["best_step"]), 1), **kw2)
        if not info["complete"]:
            return {k: v for k, v in info.items() if k != "history"}
    else:
        raise ValueError(f"unknown step {step!r}")
    save_pickle(art / f"model_{fit}.pkl", model.state())
    preds = _predict_over_shards(cfg, origin, lambda st, rows: predict_seq(model, st, rows, origin, known))
    rows = write_predictions(spark, cfg, [p.frame(variant, fit) for p in preds])
    return {"rows": rows, "complete": True, "steps": info["steps"], "best_step": info["best_step"],
            "steps_per_second": info["steps_per_second"]}


def run_seq_learning_curve(spark, cfg: Config, variant: str = "C1") -> pd.DataFrame:
    """Validation loss per head as the number of training users grows (Phase 2, learning curve).

    Each size trains with early stopping from scratch on the first ``n`` users of the first
    shard. Resumable; sizes that do not fit the time box are picked up by the next call. If
    the curve is still falling at the largest size, capacity (CPU), not the model, set the result.
    """
    from ltv.data.sequences import SequenceStore
    from ltv.train.seq_fit import fit_seq, validation_batches
    from ltv.train.timebox import load_pickle, save_pickle
    sched = get_schedule(spark, cfg)
    art, shards, box = cfg.artifact_dir("seq", variant), shard_paths(cfg), _box(cfg)
    scfg = load_pickle(art / "config.pkl")
    if scfg is None:
        raise RuntimeError(f"run_seq(step='tune', variant={variant!r}) must run first")
    kw = _seq_settings(cfg)
    store = SequenceStore.load(shards[0])
    val = validation_batches(store, sched.validation, scfg, kw["known_window"], lookback=kw["lookback"])
    done = load_pickle(art / "learning_curve.pkl", {})
    for n in cfg["seq"]["learning_curve_users"]:
        n = int(min(n, store.n_users))
        if n in done or box.expired():
            continue
        _, info = fit_seq(scfg, [store], sched.tune_train, box, art / f"curve_{n}.pt", val=val, max_users=n, **kw)
        if info["complete"] and info["history"]:
            best = min(info["history"], key=lambda h: h["total"])
            done[n] = {"users": n, **best, "seconds": info["seconds"]}
            save_pickle(art / "learning_curve.pkl", done)
    return pd.DataFrame(sorted(done.values(), key=lambda r: r["users"]))
