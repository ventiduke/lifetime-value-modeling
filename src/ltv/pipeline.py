"""Phase-level entry points. Notebooks call these; tests call the same functions.

Keeping the notebooks thin means the code that runs in Databricks is the code that was
tested locally. Each function takes a SparkSession and a Config and writes named tables.
"""
from __future__ import annotations

import time
from pathlib import Path

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
