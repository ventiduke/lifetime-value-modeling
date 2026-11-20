"""The whole pipeline, exactly as notebook 05 runs it, on a small synthetic dataset.

Slow (several minutes). It is the test that says "the code handed over has executed".
"""
import numpy as np
import pytest

from ltv import pipeline
from ltv.models.base import TARGETS


@pytest.mark.slow
def test_smoke_test_runs_every_phase(spark, tmp_path):
    from pathlib import Path

    from ltv.config import load_config
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "conf" / "bakeoff.yml", root / "conf" / "synth.yml", overrides={
        "catalog": None, "schema": "ltv_e2e", "table_format": "delta", "cluster_tables": True,
        "artifact_path": str(tmp_path / "artifacts"),
        "source": {"activity_table": "ltv_e2e.synth_weekly_activity", "calendar_table": "ltv_e2e.synth_calendar"},
        "synthetic": {"n_users": 3000, "n_weeks": 320, "seed": 3, "plant_defects": True},
        "budget": {"profile": "quick", "step_minutes": 4},
        "seq": {"quick_max_steps": 120, "eval_every": 40, "batch_users": 256},
        "evaluation": {"bootstrap_draws": 20},
    })
    out = pipeline.smoke_test(spark, cfg, seq_variants=("C1",), report_dir=tmp_path / "reports")

    # Every model wrote predictions for all three fits.
    counts = (spark.table(cfg.table("predictions")).groupBy("model_id", "fit").count().toPandas()
              .pivot(index="model_id", columns="fit", values="count"))
    assert set(counts.index) == {"R1", "R2", "R3", "A1", "A1tw", "A2", "C1", "STACK"}, "the HMM is parked"
    assert counts.notna().all().all() and list(sorted(counts.columns)) == ["final_1", "final_2", "tune"]
    cand = counts.drop(index="STACK")
    assert (cand["final_1"] == cand["final_1"].iloc[0]).all(), "all models score the same users"
    # The stack is a diagnostic: it scores only the splits that are evaluated ('val' and 'test').
    assert 0 < counts.loc["STACK", "final_1"] < cand["final_1"].iloc[0]

    # Predictions are finite and in range.
    p = spark.table(cfg.table("predictions")).select("model_id", *TARGETS).toPandas()
    assert np.isfinite(p[list(TARGETS)].to_numpy()).all()
    assert p["p_reengage"].between(0, 1).all() and p["p_spend"].between(0, 1).all()

    # The evaluation produced every table, for every model.
    res = out["eval"]
    assert set(res["metrics"]["model_id"]) == set(counts.index)
    assert {"all", "returner", "returner_x_dormancy", "dormancy"} <= set(res["metrics"]["segment_dim"])
    assert len(res["pairwise"]) and len(res["winners"]) == len(TARGETS)
    assert len(res["coherence"]) and len(res["intervals"])
    sc = res["scorecard"]
    assert sc.shape[0] == 8 and np.isfinite(sc.to_numpy()).all()

    # Sanity, not a ranking: every learned model's own probabilities should beat blind persistence on
    # re-engagement. Raw predictions are compared: with only about 200 validation users here, the
    # recalibration layer is too noisy to compare models through (real runs have about 100,000).
    m = res["metrics"]
    ll = m[(m.target == "p_reengage") & (m.metric == "log_loss") & (m.segment_dim == "all") & (m.split == "test")
           & ~m.calibrated].groupby("model_id")["value"].mean()
    assert ll["A1"] < ll["R2"] and ll["C1"] < ll["R2"]
    assert ll["A1"] < ll["R1"] + 0.05, "trees should be at least close to the segment lookup"

    # The audit saw the planted defects, and the cost harness timed every family.
    status = {f["check"]: f["status"] for f in out["audit_findings"]}
    assert status["week alignment"] == "fail" and status["suspect accounts"] == "warn"
    assert {"A", "A1", "C1"} <= set(out["cost"]["model"])
    for name in ("05_smoke_test_results.md", "05_smoke_test_cost.md", "05_smoke_test_audit.md"):
        assert (tmp_path / "reports" / name).exists()

    # Like for like: every model scored exactly the shared players, and the notebooks can show it.
    shared = pipeline.describe_inputs(spark, cfg)
    assert set(shared["split"]) == {"train", "val", "test"} and shared["origin_week"].nunique() == 3
    assert shared["fingerprint"].str.len().between(1, 16).all() and (shared["users"] > 0).all()
    same = res["same_users"]
    assert same["same_users"].all() and set(same["model_id"]) == set(counts.index)

    # Example players for plotting: one long table, three views, every saved model overlaid.
    info = out["examples"]
    assert info["models"] == ["A1", "A2", "C1"] and info["players"] >= 20 and set(info["calibrated"]) <= {"A1", "A2", "C1"}
    w = spark.table(cfg.table("example_weekly")).toPandas()
    u = spark.table(cfg.table("example_users")).toPandas()
    assert len(w) == info["rows"] and set(w["view"]) == {"actual", "path", "as_of"}
    assert set(w["series"]) == {"actual", "A1", "A2", "C1"} and set(w["user_id"]) == set(u["user_id"])
    assert w["week_start"].notna().all() and u["example_group"].nunique() >= 4
    origin = info["origin_week"]
    path = w[(w["view"] == "path") & (w["series"] == "C1")]
    assert (path.groupby("user_id").size() == 52).all() and path["week_idx"].min() == origin + 1
    assert path["hours"].notna().all() and (path["hours"] >= 0).all() and path["active"].between(0, 1).all()
    roll = w[w["view"] == "as_of"]
    assert roll["p_reengage_52"].between(0, 1).all() and roll["week_idx"].max() <= info["last_data_week"]
    assert roll["week_idx"].max() > origin, "the rolling view continues past the origin while data lasts"
    assert w.loc[w["view"] != "as_of", "p_reengage_52"].isna().all(), "empty cells are nulls, not NaN strings"
    for name in ("example_weekly.csv", "example_users.csv"):
        assert (tmp_path / "artifacts" / "examples" / name).exists()
    from ltv.eval.examples import plot_player
    assert plot_player(w, u["user_id"].iloc[0]) is not None

    # Every step was logged with its wall-clock time (MLflow too, where it is installed).
    from ltv.train.timebox import TrialLog
    log = TrialLog(tmp_path / "artifacts" / "run_log.jsonl").read()
    assert {"baselines", "trees", "seq", "stack", "eval", "examples", "cost"} <= {r["name"] for r in log}
    assert "hmm" not in {r["name"] for r in log}
    assert all(r["seconds"] >= 0 and r["profile"] == "quick" for r in log)
    tune = [r for r in log if r["name"] == "trees" and r["params"].get("step") == "tune"]
    assert tune and "trials_done" in tune[-1]["metrics"]

    # A model that scored other players is caught, and the evaluation refuses to compare it.
    pred = cfg.table("predictions")
    cols = [c for c in spark.table(pred).columns if c != "model_id"]
    spark.sql(f"INSERT INTO {pred} (model_id, {', '.join(cols)}) SELECT 'ZZ', {', '.join(cols)} FROM {pred} "
              f"WHERE model_id = 'A1' AND PMOD(XXHASH64(user_id), 2) = 0")
    odd = pipeline.check_same_users(spark, cfg)
    assert not odd.loc[odd["model_id"] == "ZZ", "same_users"].any() and odd.loc[odd["model_id"] != "ZZ", "same_users"].all()
    with pytest.raises(ValueError, match="did not score the same users"):
        pipeline.run_eval(spark, cfg)
    spark.sql(f"DELETE FROM {pred} WHERE model_id = 'ZZ'")

    # The parked HMM still runs from its notebook's steps.
    for step in ("tune", "score", "final_1", "final_2"):
        for _ in range(60):
            if pipeline.run_hmm(spark, cfg, step).get("complete", True):
                break
    b = spark.table(pred).where("model_id IN ('B1', 'B2')").groupBy("model_id", "fit").count().toPandas()
    assert len(b) == 6 and spark.table(cfg.table("hmm_user_state")).count() > 0
    assert pipeline.check_same_users(spark, cfg)["same_users"].all()
