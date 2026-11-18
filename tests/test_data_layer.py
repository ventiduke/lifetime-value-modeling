"""The data layer: cleaning, features, labels, sequences, leakage and the audit.

These run the real SQL files on local Spark against synthetic data with planted defects.
"""
import numpy as np
import pandas as pd
import pytest

from ltv import pipeline
from ltv.data import sqlrun
from ltv.data.np_features import FEATURE_NAMES, tabular_features


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_clean_table_has_no_planted_defect(spark, cfg, built, synth):
    act = spark.table(cfg.table("activity_clean")).toPandas()
    assert not act.duplicated(["user_id", "week_idx"]).any(), "one row per user-week"
    assert ((act["hours"] > 0) | (act["spend"] > 0)).all(), "no zero-zero rows"
    assert (act["spend"] >= 0).all(), "negative spend is clipped"
    assert not act["user_id"].str.startswith("test").any(), "suspect accounts are dropped"
    assert not act["user_id"].str.startswith("zero").any(), "zero-only users have no rows"
    assert act["week_idx"].between(0, 319).all()
    # Duplicated user-weeks were summed and took the platform of the row with most hours.
    assert not (act["platform"] == "handheld").any()
    assert act["n_source_rows"].max() >= 2


def test_calendar_weeks_are_contiguous(spark, cfg, built):
    cal = spark.table(cfg.table("calendar_weeks")).orderBy("week_idx").toPandas()
    assert (cal["week_idx"].to_numpy() == np.arange(len(cal))).all()
    assert int(cal["has_data"].sum()) == 320
    assert (cal.loc[cal["is_launch"], "season_week"] == 1).all()


def test_user_split_is_stable_and_roughly_80_10_10(spark, cfg, built):
    u = spark.table(cfg.table("users")).toPandas()
    share = u["split"].value_counts(normalize=True)
    assert abs(share["train"] - 0.8) < 0.04 and abs(share["val"] - 0.1) < 0.03
    again = sqlrun.select(spark, cfg, "build/05_users").toPandas()
    m = u.merge(again, on="user_id", suffixes=("", "_2"))
    assert (m["split"] == m["split_2"]).all()


# ---------------------------------------------------------------------------
# Features: sparse SQL path == dense NumPy path
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sql_features(spark, cfg, built):
    return spark.table(cfg.table("features_tabular")).toPandas()


def test_feature_columns_match_the_numpy_oracle(sql_features):
    assert list(sql_features.columns) == ["user_id", "origin_week", "split"] + list(FEATURE_NAMES)


@pytest.mark.parametrize("which", ["validation", "test_1", "early_train"])
def test_sql_features_equal_numpy_features(sql_features, store, built, cfg, which):
    sched = built["schedule"]
    origin = {"validation": sched.validation, "test_1": sched.test_1,
              "early_train": sched.tune_train[2]}[which]
    f = sql_features[sql_features["origin_week"] == origin].set_index("user_id")
    assert len(f) > 200
    pos = pd.Series(np.arange(store.n_users), index=store.user_ids)
    rows = pos[f.index].to_numpy()
    X = tabular_features(store, rows, origin, known_window=cfg["calendar"]["known_window"])
    for j, name in enumerate(FEATURE_NAMES):
        a, b = f[name].to_numpy(dtype=float), X[:, j].astype(float)
        assert (np.isnan(a) == np.isnan(b)).all(), f"{name}: NULL pattern differs at origin {origin}"
        ok = ~np.isnan(a)
        np.testing.assert_allclose(a[ok], b[ok], rtol=2e-4, atol=2e-3, err_msg=f"{name} at origin {origin}")


def test_population_matches_between_sql_and_store(spark, cfg, built, store):
    sched = built["schedule"]
    uo = spark.table(cfg.table("user_origins")).where(f"origin_week = {sched.test_2}").toPandas()
    rows = store.population(sched.test_2, int(cfg["population_lookback_weeks"]))
    assert set(uo["user_id"]) == set(store.user_ids[rows])


def test_lapse_features_on_a_hand_built_history():
    """A player active in weeks 0-2, 30-31 and 60, scored at week 70."""
    from ltv.data.sequences import SequenceStore
    W = 80
    h = np.zeros((1, W), np.float32); h[0, [0, 1, 2, 30, 31, 60]] = 5.0
    z = np.zeros((1, W), np.int16)
    sw = np.tile(np.arange(1, 9), 20)[:W + 60].astype(np.int16)
    si = (np.arange(W + 60) // 8).astype(np.int16)
    st = SequenceStore(np.array(["a"]), np.array(["train"]), np.array([0], np.int32), h,
                       np.zeros((1, W), np.float32), z, z, sw, si)
    x = dict(zip(FEATURE_NAMES, tabular_features(st, np.array([0]), 70)[0]))
    assert x["n_returns_4"] == 2 and x["n_returns_13"] == 2 and x["n_returns_26"] == 2
    assert x["median_lapse"] == pytest.approx(27.5)          # lapses of 27 and 28 weeks
    assert x["max_lapse"] == 28 and x["n_spells"] == 3
    assert x["mean_spell_len"] == pytest.approx((3 + 2 + 1) / 3)
    assert x["wk_since_active"] == 10 and x["wk_since_return"] == 10
    assert x["cur_lapse_rel"] == pytest.approx(10 / 27.5)
    assert x["aw_life"] == 6 and x["aw_13"] == 1 and x["tenure_weeks"] == 70


# ---------------------------------------------------------------------------
# Labels and sequences
# ---------------------------------------------------------------------------

def test_sequence_store_reproduces_the_labels_table(spark, cfg, built, store):
    sched = built["schedule"]
    for origin in (sched.validation, sched.test_1):
        lab = spark.table(cfg.table("labels")).where(f"origin_week = {origin}").toPandas().set_index("user_id")
        pos = pd.Series(np.arange(store.n_users), index=store.user_ids)
        y = store.labels(pos[lab.index].to_numpy(), origin)
        for k in ("any_active_52", "active_weeks_52", "hours_52", "any_spend_52", "spend_52"):
            np.testing.assert_allclose(lab[k].to_numpy(dtype=float), y[k], rtol=1e-4, atol=1e-2, err_msg=k)
        np.testing.assert_allclose(np.stack(lab["cum_hours"].to_numpy()), y["cum_hours"], rtol=1e-4, atol=1e-2)
        np.testing.assert_allclose(np.stack(lab["cum_spend"].to_numpy()), y["cum_spend"], rtol=1e-4, atol=1e-2)
        np.testing.assert_array_equal(np.stack(lab["weekly_active"].to_numpy()), y["weekly_active"])


def test_labels_are_internally_coherent(spark, cfg, built):
    lab = spark.table(cfg.table("labels")).toPandas()
    wa = np.stack(lab["weekly_active"].to_numpy())
    assert wa.shape[1] == 52
    assert (wa.sum(1) == lab["active_weeks_52"]).all()
    assert ((lab["active_weeks_52"] > 0).astype(int) == lab["any_active_52"]).all()
    ch = np.stack(lab["cum_hours"].to_numpy())
    assert (np.diff(ch, axis=1) >= -1e-9).all() and np.allclose(ch[:, -1], lab["hours_52"])
    assert ((lab["hours_52"] > 0) == (lab["any_active_52"] == 1)).all()


def test_sequences_match_the_synthetic_truth(store, synth):
    """The SQL densification must give back exactly what the generator simulated."""
    truth = synth.truth
    pos = pd.Series(np.arange(len(truth["user_ids"])), index=truth["user_ids"])
    keep = np.flatnonzero(np.isin(store.user_ids, truth["user_ids"]))
    t_rows = pos[store.user_ids[keep]].to_numpy()
    differ = ~np.isclose(store.hours[keep], truth["hours"][t_rows], atol=2e-3)
    # The planted outage week lost most of its rows; planted duplicates added hours to 200 user-weeks.
    outage = int((pd.Timestamp(synth.defects["outage_week"]) - pd.Timestamp("2020-09-07")).days // 7)
    differ[:, outage] = False
    assert differ.sum() <= synth.defects["duplicate_user_weeks"], "only planted duplicates may differ"
    assert np.isclose(store.spend[keep], truth["spend"][t_rows], atol=6e-3)[:, np.arange(320) != outage].mean() > 0.9999


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_features_do_not_read_past_the_origin(spark, cfg, built):
    """Delete every activity row after an origin; the features at that origin must not change."""
    sched = built["schedule"]
    origin = sched.validation
    before = (spark.table(cfg.table("features_tabular")).where(f"origin_week = {origin}")
              .toPandas().set_index("user_id").sort_index())
    full = cfg.table("activity_clean")
    spark.table(full).where(f"week_idx <= {origin}").write.mode("overwrite").format("parquet") \
        .saveAsTable("ltv_test.activity_truncated")
    params = sqlrun.base_params(cfg)
    params["t_activity_clean"] = "ltv_test.activity_truncated"
    after = (spark.sql(sqlrun.load_sql("build/07_features_tabular", params))
             .where(f"origin_week = {origin}").toPandas().set_index("user_id").sort_index())
    assert list(before.index) == list(after.index)
    pd.testing.assert_frame_equal(before[list(FEATURE_NAMES)], after[list(FEATURE_NAMES)],
                                  check_exact=False, rtol=1e-9)


def test_panel_only_looks_backward(spark, cfg, built, store):
    """Carried-forward fields in the dense panel equal a backward-only NumPy computation."""
    p = (spark.table(cfg.table("panel_weekly")).where("week_idx = 200")
         .select("user_id", "weeks_since_last_active", "tenure_weeks", "active_streak").toPandas()
         .set_index("user_id"))
    pos = pd.Series(np.arange(store.n_users), index=store.user_ids)
    rows = pos[p.index].to_numpy()
    act = store.hours[rows, :201] > 0
    last = np.where(act, np.arange(201), -1).max(1)
    expect = np.where(last >= 0, 200 - last, np.nan)
    got = p["weeks_since_last_active"].to_numpy(dtype=float)
    assert (np.isnan(expect) == np.isnan(got)).all()
    np.testing.assert_array_equal(expect[~np.isnan(expect)], got[~np.isnan(got)])
    np.testing.assert_array_equal(p["tenure_weeks"].to_numpy(), 200 - store.first_week[rows])


# ---------------------------------------------------------------------------
# Audit: each planted defect is reported
# ---------------------------------------------------------------------------

def test_audit_reports_every_planted_defect(spark, cfg, built, synth, tmp_path):
    audit = pipeline.run_audit(spark, cfg)
    d = synth.defects
    weekly = audit["a01_weekly_row_counts"]
    flagged = set(map(str, weekly.loc[weekly["suspect_outage"], "week_start"]))
    assert d["outage_week"] in flagged, "the planted outage week is flagged"
    assert len(flagged) <= 3, "ordinary weeks are not flagged"
    r = audit["a02_row_semantics"].iloc[0]
    assert r["zero_zero_rows"] >= d["zero_rows"]
    assert r["negative_spend_rows"] == d["negative_spend_rows"]
    assert r["null_platform_rows"] == d["null_dimension_rows"]
    assert r["over_max_hours_rows"] > 0
    assert audit["a04_duplicate_user_weeks"].iloc[0]["multi_platform_user_weeks"] >= d["duplicate_user_weeks"] * 0.9
    a5 = audit["a05_week_alignment"]
    assert len(a5) == 2 and a5["n_rows_not_in_calendar"].sum() == d["misaligned_rows"]
    assert len(audit["a07_suspect_accounts"]) == d["suspect_users"]
    h = audit["a06_history_and_users"].iloc[0]
    assert h["weeks_of_history"] == 320 and h["distinct_weeks"] == 320
    assert set(audit["a10_lapse_and_return"]["returns_from_26w_lapse"]) >= {"0", "1"}
    base = audit["a09_dormancy_base_rates"].set_index("dormancy_bucket")
    assert base.loc["1: active this week", "p_reengage_52"] > base.loc["4: dormant 27-104", "p_reengage_52"]

    findings = {f["check"]: f["status"] for f in pipeline.audit_findings(audit)}
    assert findings["week alignment"] == "fail"
    assert findings["weekly row counts"] == "warn"
    assert findings["rows with negative spend"] == "warn"
    assert findings["suspect accounts"] == "warn"
    assert findings["weeks of history"] == "ok"
    report = pipeline.write_audit_report(audit, pipeline.audit_findings(audit), tmp_path / "audit.md")
    assert "## Findings" in report.read_text()
