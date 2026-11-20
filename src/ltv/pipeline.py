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
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")       # reports go to the repo, not a volume
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
    info = {"rows": 0}
    for fit in FITS:
        tr, _, sc = frames_for_fit(spark, cfg, sched, fit)
        origin = sched.score_origin(fit)
        for model in (SegmentLookup(), Persistence(), BGBB()):
            model.fit(tr)
            info["rows"] += write_predictions(spark, cfg, model.predict(sc, origin).frame(model.model_id, fit))
            if isinstance(model, BGBB):
                info[f"bgbb_{fit}"] = {"activity": model.p_act.round(4).tolist(),
                                       "spend": model.p_spd.round(4).tolist()}
    return info


def run_trees(spark, cfg: Config, step: str) -> dict:
    """Candidate A in four resumable steps, one notebook command each:

    'tune'    time-boxed random search for one shared hyperparameter set (call again until complete)
    'fit'     fit every model at the tuned settings, freeze tree counts, score the validation origin
    'final_1' refit with frozen settings and score test_1
    'final_2' the same for test_2

    'fit' and the final fits are time-boxed too: they save after each of the 24 models
    (``partial_<step>.pkl``) and return ``complete: False`` when the box ends first. At 2M
    simulated rows one boosting round over the suite cost about 0.12 s on 4 cores, so 5M rows
    at up to 2,000 rounds can pass an hour (``reports/01_simulation_study.md``).
    """
    from ltv.models.base import write_predictions
    from ltv.models.trees import DEFAULT_PARAMS, SEARCH_SPACE, TreeSuite, tuning_objective
    from ltv.train.timebox import TrialLog, load_pickle, random_search, save_pickle
    sched = get_schedule(spark, cfg)
    art, box = cfg.artifact_dir("trees"), _box(cfg)
    quick = _profile(cfg) == "quick"

    def resume(suite: TreeSuite, step: str):
        """Continue from the models already fitted for this step, if the settings still match."""
        part = art / f"partial_{step}.pkl"
        saved = load_pickle(part)
        frozen = dict(suite.rounds)
        if saved is not None and saved["params"] == suite.params and saved["max_rounds"] == suite.max_rounds \
                and all(saved["rounds"].get(k) == v for k, v in frozen.items()):
            suite = TreeSuite.from_state(saved)
            suite.rounds = {**frozen, **suite.rounds}
        return suite, (lambda: save_pickle(part, suite.state()))
    max_rounds, n_trials = (300, 4) if quick else (2000, 30)
    seed = int(cfg["training"]["seed"])
    if step == "tune":
        tr, va, _ = frames_for_fit(spark, cfg, sched, "tune")
        base = {}
        for r in TrialLog(art / "tuning.jsonl").read():          # restore the baseline losses
            if r["trial"] == 0:
                base.update(r["info"]["losses"])
        best = random_search(SEARCH_SPACE, tuning_objective(tr, va, max_rounds, 50, seed, base),
                             n_trials, art / "tuning.jsonl", box, seed, defaults={})
        return {"best_params": best["params"], "best_loss": best["loss"], "trials_done": best["n_done"],
                "complete": best["complete"], "train_rows": len(tr)}
    if step == "fit":
        trials = TrialLog(art / "tuning.jsonl").read()
        params = dict(DEFAULT_PARAMS)
        if trials:
            params.update(min(trials, key=lambda r: r["loss"])["params"])
        tr, va, sc = frames_for_fit(spark, cfg, sched, "tune")
        suite, save = resume(TreeSuite(params=params, max_rounds=max_rounds, seed=seed), step)
        losses = suite.fit(tr, va, box=box, on_model=save)
        if not suite.complete:
            return {"complete": False, "models_done": len(suite.boosters), "train_rows": len(tr)}
        save_pickle(art / "suite_tune.pkl", suite.state())
        fit, origin = "tune", sched.validation
    elif step in ("final_1", "final_2"):
        st = load_pickle(art / "suite_tune.pkl")
        if st is None:
            raise RuntimeError("run_trees(step='fit') must finish before the final fits")
        tr, _, sc = frames_for_fit(spark, cfg, sched, step)
        suite, save = resume(TreeSuite(params=st["params"], max_rounds=st["max_rounds"], seed=seed,
                                       rounds=dict(st["rounds"])), step)
        losses = suite.fit(tr, None, box=box, on_model=save)
        if not suite.complete:
            return {"complete": False, "models_done": len(suite.boosters), "train_rows": len(tr)}
        save_pickle(art / f"suite_{step}.pkl", suite.state())
        fit, origin = step, sched.score_origin(step)
    else:
        raise ValueError(f"unknown step {step!r}")
    n_train = len(tr)
    del tr                                      # free the training frame before building prediction rows
    frames = (make().frame(name, fit) for name, make in (
        ("A1", lambda: suite.predict_a1(sc, origin)),
        ("A1tw", lambda: suite.predict_a1(sc, origin, tweedie_spend=True)),
        ("A2", lambda: suite.predict_a2(sc, origin))))
    return {"rows": write_predictions(spark, cfg, frames), "train_rows": n_train, "complete": True,
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
                              int(h["b2_spend_classes"]), nb, bool(h.get("b2_user_hours_level", True))))
               for t in b2])


def run_hmm(spark, cfg: Config, step: str) -> dict:
    """Candidate B in resumable steps, one notebook command each:

    'tune'    EM for every candidate size on weeks <= the validation origin; records the
              validation log-likelihood of each. Call again until ``complete`` is true.
    'score'   pick the best B1 and the best B2 by validation log-likelihood, score the validation origin
    'final_1' refit the chosen models on weeks <= test_1 (warm-started), score test_1, save states
    'final_2' the same for test_2

    EM reads the 'train' users of the first ``hmm.train_shards`` shards only; scoring and the
    validation log-likelihood use every shard. One EM pass of the 60-state B2 costs about a
    millisecond per user on 4 cores, and in simulation its accuracy did not change between
    2,000 and 60,000 training users (``reports/01_simulation_study.md``).
    """
    from ltv.models.base import write_predictions
    from ltv.models.hmm import HMM, fit_hmm, level_state, loglik_per_week, predict_hmm, state_profile
    from ltv.train.timebox import load_pickle, save_pickle
    sched = get_schedule(spark, cfg)
    art, shards, box = cfg.artifact_dir("hmm"), shard_paths(cfg), _box(cfg)
    em_shards = shards[: max(1, int(cfg["hmm"].get("train_shards", 1)))]
    edges = tuple(cfg["calendar"]["season_week_buckets"])
    kw = dict(edges=edges, batch=int(cfg["hmm"]["batch_users"]), seed=int(cfg["training"]["seed"]),
              max_iters=30 if _profile(cfg) == "quick" else int(cfg["hmm"]["max_iters"]))
    known = cfg["calendar"]["known_window"]
    cands = _hmm_candidates(cfg)
    sel_path = art / "selection.pkl"
    sel = load_pickle(sel_path, {})
    if step == "tune":
        for family, spec in cands:
            if spec.name in sel or box.expired():
                continue
            model, info = fit_hmm(spec, em_shards, sched.validation, box, art / f"tune_{spec.name}.pkl", **kw)
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
    info = {"chosen": best, "rows": 0}
    if step == "score":
        fit, origin = "tune", sched.validation
        models = {fam: HMM(specs[name], sel[name]["params"]) for fam, name in best.items()}
    elif step in ("final_1", "final_2"):
        fit, origin = step, sched.score_origin(step)
        models = {}
        for fam, name in best.items():
            m, i = fit_hmm(specs[name], em_shards, origin, box, art / f"{step}_{name}.pkl",
                           start=sel[name]["params"], **kw)
            if not i["complete"]:
                return {"complete": False, "chosen": best, "waiting_on": name, "iters": i["iters"]}
            models[fam] = m
    else:
        raise ValueError(f"unknown step {step!r}")
    states = []
    for fam, m in models.items():
        both = _predict_over_shards(cfg, origin, lambda st, rows, m=m: (
            predict_hmm(m, st, rows, origin, known_window=known, edges=edges), level_state(m, st, rows, origin)))
        preds = [p for p, _ in both]
        info["rows"] += write_predictions(spark, cfg, (p.frame(fam, fit) for p in preds))
        info[f"profile_{fam}"] = state_profile(m)
        save_pickle(art / f"model_{fit}_{fam}.pkl", {"spec": m.spec, "params": m.export()})
        if fam == "B2":
            states += [pd.DataFrame({"user_id": p.user_ids, "origin_week": np.int32(origin),
                                     "alpha": [a.tolist() for a in p.state],
                                     "level_sum": ls.astype(float), "level_n": ln.astype(np.int32)})
                       for p, (ls, ln) in both]
    if states and fit != "tune":
        # The per-user state table that weekly scoring would carry forward (docs/DESIGN.md, "Scoring").
        all_states = pd.concat(states, ignore_index=True)
        fq, fmt = cfg.table("hmm_user_state"), cfg.get("table_format", "delta")
        for i in range(0, len(all_states), 50_000):          # upload in pieces (Spark Connect message size)
            sdf = spark.createDataFrame(all_states.iloc[i:i + 50_000],
                                        "user_id string, origin_week int, alpha array<double>, "
                                        "level_sum double, level_n int")
            w = sdf.write.mode("overwrite" if i == 0 else "append").format(fmt)
            if i == 0 and spark.catalog.tableExists(fq) and fmt == "delta":
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
                lr_drops=int(s.get("lr_drops", 0)),
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
        kw2 = {k: v for k, v in kw.items() if k not in ("max_steps", "eval_every", "patience", "lr_drops")}
        best = max(int(tinfo["best_step"]), 1)
        model, info = fit_seq(scfg, shards, sched.train_origins(step), box, art / f"{step}.pt", fixed_steps=best,
                              lr_steps=[x for x in tinfo.get("lr_steps", []) if x <= best], **kw2)
        if not info["complete"]:
            return {k: v for k, v in info.items() if k != "history"}
    else:
        raise ValueError(f"unknown step {step!r}")
    save_pickle(art / f"model_{fit}.pkl", model.state())
    preds = _predict_over_shards(cfg, origin, lambda st, rows: predict_seq(model, st, rows, origin, known))
    rows = write_predictions(spark, cfg, (p.frame(variant, fit) for p in preds))
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


# ---------------------------------------------------------------------------
# Phase 3: evaluation
# ---------------------------------------------------------------------------

def load_eval_frame(spark, cfg: Config, sched: Schedule) -> pd.DataFrame:
    """Predictions joined to labels and features at the validation and test origins.

    Only the user splits in ``evaluation.splits`` are loaded (default: val and test). The
    train split is four fifths of the rows and is used only for an overfitting diagnostic;
    at a 1M-user sample it does not fit in driver memory next to the other two.
    """
    origins = ", ".join(map(str, sched.eval_origins()))
    splits = ", ".join(f"'{s}'" for s in cfg["evaluation"].get("splits", ["val", "test"]))
    p = spark.table(cfg.table("predictions")).where(f"origin_week IN ({origins})").drop("state")
    lab = spark.table(cfg.table("labels")).drop("cum_hours", "cum_spend", "weekly_active")
    feat = spark.table(cfg.table("features_tabular")).where(f"split IN ({splits})")
    return p.join(lab, ["user_id", "origin_week"]).join(feat, ["user_id", "origin_week"]).toPandas()


def run_eval(spark, cfg: Config, report_path: str | Path | None = None) -> dict:
    """Score every model in the predictions table and write eval_metrics and eval_pairwise.

    Returns the result tables (pandas) plus a scorecard. With ``report_path`` it also writes
    the Markdown results report.
    """
    from ltv.eval.evaluate import evaluate, scorecard
    sched = get_schedule(spark, cfg)
    frame = load_eval_frame(spark, cfg, sched)
    res = evaluate(frame, sched, int(cfg["evaluation"]["bootstrap_draws"]), int(cfg["training"]["seed"]),
                   int(cfg["evaluation"]["top_countries"]))
    res["scorecard"] = scorecard(res, sched)
    if len(res["metrics"]):
        sqlrun.write_table(spark, cfg, spark.createDataFrame(res["metrics"]), "eval_metrics")
    if len(res["pairwise"]):
        sqlrun.write_table(spark, cfg, spark.createDataFrame(res["pairwise"]), "eval_pairwise")
    if report_path is not None:
        write_results_report(res, sched, report_path, synthetic="synth" in str(cfg["schema"]))
    return res


def write_results_report(res: dict, sched: Schedule, path: str | Path, synthetic: bool = False) -> Path:
    """reports/03_bakeoff_results.md: scorecard, winners, returner segments, coherence, intervals."""
    title = "Bake-off results" + (" (SYNTHETIC DATA: pipeline check only, not a model comparison)" if synthetic else "")
    L = [f"# {title}", ""]
    if synthetic:
        L += ["These numbers come from generated data. They show that the pipeline runs and how the metrics "
              "behave; they do not say which model is better on real data.", ""]
    L += [f"Test origins: week {sched.test_1} and week {sched.test_2}. Validation origin: week {sched.validation}. "
          "Headline numbers are for 'test'-split users, after the shared recalibration layer.", "",
          "## Winners by target", "",
          "A model wins a target only if it leads at both test origins and the 95% interval of its paired "
          "difference from the runner-up excludes zero at both. The runner-up is the best model of another "
          "family: variants of one candidate (A1, A1tw, A2) can share a head and tie exactly.", ""]
    L += [res["winners"].to_markdown(index=False) if len(res["winners"]) else "_no comparisons available_", "",
          "## Scorecard (primary metric and bias or AUC, per test origin)", ""]
    sc = res["scorecard"]
    L += [sc.round(4).to_markdown() if len(sc) else "_empty_", ""]
    m = res["metrics"]
    if len(m):
        r = m[(m["segment_dim"] == "returner_x_dormancy") & (m["target"] == "p_reengage") & (m["split"] == "test")
              & (m["calibrated"]) & (m["metric"].isin(["mean_pred", "mean_actual"]))]
        if len(r):
            piv = r.pivot_table(index=["segment", "origin_week", "n"], columns=["model_id", "metric"], values="value")
            actual = piv.xs("mean_actual", axis=1, level=1).iloc[:, 0].rename("actual")
            tab = pd.concat([actual, piv.xs("mean_pred", axis=1, level=1)], axis=1).reset_index()
            tab = tab.rename(columns={"origin_week": "origin", "n": "users"})
            L += ["## Lapse-and-return: P(re-engage) by past returns and current dormancy", "",
                  "The headline cut for long-term behavior. `actual` is the realised re-engagement rate; the "
                  "other columns are each model's mean prediction. A model that recognises habitual returners "
                  "tracks `actual` across the three past-return groups.", "", tab.round(3).to_markdown(index=False), ""]
        raw = m[(m["segment_dim"] == "all") & (m["split"] == "test") & (~m["calibrated"])
                & (m["metric"] == "aggregate_bias")]
        if len(raw):
            bias = raw.pivot_table(index="model_id", columns=["target", "origin_week"], values="value")
            bias.columns = [f"{t} @{o}" for t, o in bias.columns]
            L += ["## Aggregate bias before recalibration (predicted total / actual total)", "",
                  "1.0 is unbiased. This is what each model does on its own, before the shared layer.", "",
                  bias.round(3).to_markdown(), ""]
    if len(res["coherence"]):
        c = res["coherence"].pivot_table(index="check", columns="model_id", values="share", aggfunc="mean")
        L += ["## Coherence violations (share of users, mean over test origins)", "", c.round(4).to_markdown(), ""]
    if len(res["intervals"]):
        i = res["intervals"]
        i = i[i["segment"] == "all"].pivot_table(index=["target", "kind", "level"], columns="model_id",
                                                 values="coverage", aggfunc="mean").reset_index()
        L += ["## Interval coverage (should be near the stated level)", "",
              "`conformal` is the shared layer; `model_quantiles` is a model's own 10%-90% band (level 0.8).", "",
              i.round(3).to_markdown(index=False), ""]
    if len(res["pairwise"]):
        L += ["## Paired comparisons on the primary metric", "", res["pairwise"].round(5).to_markdown(index=False), ""]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Phase 3: speed and cost
# ---------------------------------------------------------------------------

def run_cost(spark, cfg: Config, report_path: str | Path | None = None) -> pd.DataFrame:
    """Time each scoring path on the sample at the test_1 origin and extrapolate.

    This is a measuring harness, not a result. Run it on the real platform: the numbers
    depend entirely on the compute it runs on. Extrapolation to 30M and 150M users is linear
    in users and assumes the same compute, so treat it as a first estimate and confirm the
    finalist with one full run. Models whose artifacts are missing are skipped.
    """
    from ltv.data.sequences import SequenceStore
    from ltv.models import kernels_np as kn
    from ltv.models.hmm import HMM, calendar_buckets, filter_users, forecast_matrices, future_buckets, level_state
    from ltv.models.seq import SeqModel, predict_seq
    from ltv.score.batch import Timer, hmm_forecast_frame, score_trees_driver, score_trees_spark
    from ltv.train.timebox import load_pickle
    sched = get_schedule(spark, cfg)
    origin, art = sched.test_1, Path(cfg["artifact_path"])
    known, edges = cfg["calendar"]["known_window"], tuple(cfg["calendar"]["season_week_buckets"])
    lookback = int(cfg["population_lookback_weeks"])
    T = Timer()
    # 1. Features for one origin, built in SQL from the sparse table.
    uo = spark.table(cfg.table("user_origins")).where(f"origin_week = {origin}")
    sqlrun.write_table(spark, cfg, uo, "score_user_origins")
    n = spark.table(cfg.table("score_user_origins")).count()
    T.run("build features (SQL)", "A", n, lambda: sqlrun.build(
        spark, cfg, "build/07_features_tabular", "score_features", t_user_origins=cfg.table("score_user_origins")))
    feats = spark.table(cfg.table("score_features"))
    # 2. Trees.
    st = load_pickle(art / "trees" / "suite_final_1.pkl")
    if st is not None:
        T.run("score: mapInPandas", "A1", n, lambda: sqlrun.write_table(
            spark, cfg, score_trees_spark(feats, st, "A1"), "score_predictions"), "includes the table write")
        T.run("score: driver batch", "A1", n, lambda: score_trees_driver(feats, st, "A1"), "includes toPandas")
        T.run("score: driver batch", "A2", n, lambda: score_trees_driver(feats, st, "A2"), "24 models")
    shards = shard_paths(cfg)
    store = SequenceStore.load(shards[0])
    rows = store.population(origin, lookback)
    # 3. HMM: full-history filter, the one-week update, and the forecast.
    for fam in ("B1", "B2"):
        saved = load_pickle(art / "hmm" / f"model_final_1_{fam}.pkl")
        if saved is None:
            continue
        model = HMM(saved["spec"], saved["params"])
        params = model.export()
        alpha = T.run("filter whole history", fam, len(rows), lambda: filter_users(model, store, rows, origin, edges)[0],
                      "what a cold start costs")
        b = int(calendar_buckets(store, edges)[origin])
        h, s = store.hours[rows, origin].astype(float), store.spend[rows, origin].astype(float)
        if model.spec.user_level:                      # this week's hours on the user's own scale
            h = kn.hmm_level_update(params, h, *level_state(model, store, rows, origin - 1))[0]
        T.run("advance state one week (NumPy)", fam, len(rows), lambda: kn.hmm_step(alpha, params, b, h, s),
              "the weekly cost per user")
        V = forecast_matrices(params, future_buckets(store, origin, 52, known, edges))
        T.run("forecast from state", fam, len(rows), lambda: hmm_forecast_frame(alpha, store.user_ids[rows], V))
    # 4. Sequence models: recompute from the whole history.
    for variant in SEQ_VARIANTS:
        saved = load_pickle(art / "seq" / variant / "model_final_1.pkl")
        if saved is None:
            continue
        model = SeqModel.from_state(saved)
        T.run("score from whole history (PyTorch, driver)", variant, len(rows),
              lambda: predict_seq(model, store, rows, origin, known), "includes building inputs")
    out = T.frame()
    if len(out):
        sqlrun.write_table(spark, cfg, spark.createDataFrame(out), "cost_timings")
    if report_path is not None:
        p = Path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Speed and cost timings\n\nMeasured by `ltv.pipeline.run_cost` on the user sample at the "
                     f"test origin (week {origin}). Extrapolations are linear in users on the same compute; "
                     "confirm the finalist with one full run. Requirement: no command over 1 hour; batching is "
                     "allowed.\n\n" + out.to_markdown(index=False) + "\n", encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Storage, synthetic source tables and the smoke test
# ---------------------------------------------------------------------------

def ensure_storage(spark, cfg: Config) -> None:
    """Create the output schema and, if artifact_path is a Unity Catalog volume, the volume."""
    sqlrun.ensure_schema(spark, cfg)
    parts = Path(cfg["artifact_path"]).parts
    if len(parts) >= 5 and parts[1] == "Volumes":
        spark.sql(f"CREATE VOLUME IF NOT EXISTS {parts[2]}.{parts[3]}.{parts[4]}")
    cfg.artifact_dir()


def write_synthetic_source(spark, cfg: Config):
    """Generate synthetic data (conf/synth.yml, ``synthetic``) and write the two source tables."""
    from ltv.synth.generator import generate
    s = cfg["synthetic"]
    data = generate(int(s["n_users"]), int(s["n_weeks"]), int(s["seed"]), plant=bool(s["plant_defects"]))
    fmt = cfg.get("table_format", "delta")
    act = data.activity
    step = 500_000                                  # upload in pieces: Spark Connect limits message size
    for i in range(0, len(act), step):
        (spark.createDataFrame(act.iloc[i:i + step]).write.mode("overwrite" if i == 0 else "append")
         .format(fmt).saveAsTable(cfg["source"]["activity_table"]))
    spark.createDataFrame(data.calendar).write.mode("overwrite").format(fmt).saveAsTable(cfg["source"]["calendar_table"])
    return data


def environment_report(cfg: Config) -> dict:
    """What the compute looks like, with two micro-benchmarks that size the training steps."""
    import os
    import platform
    import time as _t

    import lightgbm as lgb
    import torch
    out = {"python": platform.python_version(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
           "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "lightgbm": lgb.__version__}
    try:
        with open("/proc/meminfo") as fh:
            out["memory_gb"] = round(int(fh.readline().split()[1]) / 1e6, 1)
    except OSError:
        out["memory_gb"] = None
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200_000, 50)).astype(np.float32), rng.random(200_000)
    t0 = _t.time()
    lgb.train({"objective": "regression", "verbose": -1, "num_leaves": 63}, lgb.Dataset(X, y), 50)
    out["lightgbm_rows_x_trees_per_second"] = round(200_000 * 50 / (_t.time() - t0))
    gru = torch.nn.GRU(64, 64, batch_first=True)
    x = torch.randn(512, 100, 64)
    t0 = _t.time()
    for _ in range(5):
        gru(x)[0].sum().backward()
    out["gru_sequences_per_second"] = round(5 * 512 / (_t.time() - t0))
    return out


def smoke_test(spark, cfg: Config, seq_variants=("C1",), report_dir: str | Path | None = None) -> dict:
    """Run the WHOLE pipeline on synthetic data: source tables, audit, build, every model
    family, evaluation and the cost harness. Notebook 05 calls this; so does the test suite.

    Passing means libraries, permissions, storage and the code path all work here. It does
    not mean any model is good: synthetic results are never a model comparison.
    """
    out: dict = {"environment": environment_report(cfg)}
    t0 = time.time()

    def until_complete(fn, *args, max_calls: int = 60):
        """Repeat a resumable step until it reports completion, as a person would across commands."""
        for _ in range(max_calls):
            r = fn(spark, cfg, *args)
            if r.get("complete", True):
                return r
        raise RuntimeError(f"{fn.__name__}{args} did not complete in {max_calls} calls; raise budget.step_minutes")

    ensure_storage(spark, cfg)
    data = write_synthetic_source(spark, cfg)
    out["defects_planted"] = data.defects
    info = build_all(spark, cfg)
    out["schedule"] = info["schedule"]
    audit = run_audit(spark, cfg)
    out["audit_findings"] = audit_findings(audit, min_weeks=int(cfg["synthetic"]["n_weeks"]) - 1)
    out["shards"] = len(export_sequences(spark, cfg))
    out["baselines"] = run_baselines(spark, cfg)
    until_complete(run_trees, "tune")
    for step in ("fit", "final_1", "final_2"):
        out[f"trees_{step}"] = run_trees(spark, cfg, step)
    until_complete(run_hmm, "tune")
    for step in ("score", "final_1", "final_2"):
        out[f"hmm_{step}"] = until_complete(run_hmm, step)
    for v in seq_variants:
        until_complete(run_seq, "tune", v)
        for step in ("score", "final_1", "final_2"):
            out[f"seq_{v}_{step}"] = until_complete(run_seq, step, v)
    if seq_variants:
        out["stack"] = run_stack(spark, cfg, "A2", seq_variants[0])
    rd = Path(report_dir) if report_dir is not None else None
    out["eval"] = run_eval(spark, cfg, rd / "05_smoke_test_results.md" if rd else None)
    out["cost"] = run_cost(spark, cfg, rd / "05_smoke_test_cost.md" if rd else None)
    if rd:
        write_audit_report(audit, out["audit_findings"], rd / "05_smoke_test_audit.md",
                           "Smoke test audit (SYNTHETIC DATA with planted defects)")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run_stack(spark, cfg: Config, model_a: str = "A2", model_c: str = "C1") -> dict:
    """Stack diagnostic (not a candidate): a second-level LightGBM on the tabular features
    plus the predictions of one tree model and one sequence model.

    It is fitted on 'val'-split users at the validation origin, where both models'
    predictions are out of sample, and scored at the test origins as model ``STACK``.
    A large gain over both inputs means each family misses signal the other has
    (Bauer and Jannach 2021 report about 4% lower RMSE from such a stack).
    Caveat: the shared recalibration layer is fitted on the same validation users the stack
    trained on, so STACK's calibrated numbers are slightly optimistic. Read it as a diagnostic.
    """
    import lightgbm as lgb

    from ltv.data.np_features import FEATURE_NAMES
    from ltv.models.base import LABEL_OF, TARGETS, Preds, write_predictions
    sched = get_schedule(spark, cfg)
    origins = ", ".join(map(str, sched.eval_origins()))
    p = (spark.table(cfg.table("predictions")).where(f"origin_week IN ({origins})")
         .where(f"model_id IN ('{model_a}', '{model_c}')").select("model_id", "fit", "origin_week", "user_id", *TARGETS))
    wide = None
    for m, tag in ((model_a, "a"), (model_c, "c")):
        d = p.where(f"model_id = '{m}'").select("fit", "origin_week", "user_id",
                                                *[p[t].alias(f"{t}_{tag}") for t in TARGETS])
        wide = d if wide is None else wide.join(d, ["fit", "origin_week", "user_id"])
    lab = spark.table(cfg.table("labels")).select("user_id", "origin_week", *LABEL_OF.values())
    splits = ", ".join(f"'{x}'" for x in cfg["evaluation"].get("splits", ["val", "test"]))
    feat = spark.table(cfg.table("features_tabular")).where(f"split IN ({splits})")   # only what is evaluated
    df = wide.join(feat, ["user_id", "origin_week"]).join(lab, ["user_id", "origin_week"]).toPandas()
    if not len(df):
        raise RuntimeError(f"no predictions found for {model_a} and {model_c}; run those candidates first")
    cols = list(FEATURE_NAMES) + [f"{t}_{tag}" for t in TARGETS for tag in ("a", "c")]
    tr = df[(df["fit"] == "tune") & (df["split"] == "val")]
    obj = {"p_reengage": "binary", "p_spend": "binary", "exp_active_weeks": "poisson",
           "exp_hours": "tweedie", "exp_spend": "tweedie"}
    models = {}
    for t in TARGETS:
        prm = {"objective": obj[t], "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
               "feature_fraction": 0.8, "lambda_l2": 5.0, "verbose": -1, "seed": int(cfg["training"]["seed"])}
        if obj[t] == "tweedie":
            prm["tweedie_variance_power"] = 1.5
        models[t] = lgb.train(prm, lgb.Dataset(tr[cols], tr[LABEL_OF[t]].to_numpy(float)), num_boost_round=150)
    frames = []
    for (fit, origin), g in df.groupby(["fit", "origin_week"]):
        out = {t: models[t].predict(g[cols]) for t in TARGETS}
        frames.append(Preds(g["user_id"].to_numpy(), int(origin), **out).frame("STACK", fit))
    gain = (pd.DataFrame({"feature": cols, "gain": models["exp_spend"].feature_importance("gain")})
            .sort_values("gain", ascending=False).head(8))
    return {"rows": write_predictions(spark, cfg, frames), "train_rows": int(len(tr)), "inputs": [model_a, model_c],
            "top_features_for_spend": gain.to_dict("records")}


# ---------------------------------------------------------------------------
# Run tracking: every run_* step is logged (run_log.jsonl, and MLflow when available)
# ---------------------------------------------------------------------------

def _tracked(name: str):
    import functools
    import inspect

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(spark, cfg, *args, **kwargs):
            from ltv.tracking import log_run
            t0 = time.time()
            out = fn(spark, cfg, *args, **kwargs)
            bound = sig.bind(spark, cfg, *args, **kwargs)
            params = {k: v for k, v in bound.arguments.items() if k not in ("spark", "cfg")}
            log_run(cfg, name, params, out if isinstance(out, dict) else {}, time.time() - t0)
            return out

        return wrapper

    return deco


run_baselines = _tracked("baselines")(run_baselines)
run_trees = _tracked("trees")(run_trees)
run_hmm = _tracked("hmm")(run_hmm)
run_seq = _tracked("seq")(run_seq)
run_stack = _tracked("stack")(run_stack)
run_eval = _tracked("eval")(run_eval)
run_cost = _tracked("cost")(run_cost)
