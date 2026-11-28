"""MLflow packaging (ltv.serve): a saved fit logged, registered, loaded and scored. No Spark, no retraining."""
import json

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from ltv import serve  # noqa: E402
from ltv.playground import core  # noqa: E402
from ltv.train.timebox import load_pickle  # noqa: E402


@pytest.fixture(scope="module")
def bundle(demo_bundle_dir):
    return core.load_bundle(demo_bundle_dir)


@pytest.fixture(scope="module")
def calendar(demo_bundle_dir):
    return pd.read_csv(demo_bundle_dir / "examples" / "calendar.csv")


@pytest.fixture(scope="module")
def players(bundle):
    """Rows shaped like the user_sequences table: one per player, arrays from the first week."""
    hist = core.presets(bundle.max_weeks)
    return hist, pd.DataFrame({"user_id": [h["id"] for h in hist],
                               "first_week": [bundle.origin + 1 - len(h["hours"]) for h in hist],
                               "hours": [np.asarray(h["hours"], np.float32) for h in hist],
                               "spend": [np.asarray(h["spend"], np.float32) for h in hist]})


@pytest.fixture(scope="module")
def logged(demo_bundle_dir, bundle, calendar):
    """Each demo fit logged and registered once, from the files the training step saved."""
    recal = load_pickle(demo_bundle_dir / "examples" / "calibration.pkl")
    mlflow.set_experiment("ltv-serve-tests")
    out = {}
    for mid in ("A1", "A2", "C1"):
        with mlflow.start_run(run_name=f"register {mid}"):
            out[mid] = serve.log_fit(demo_bundle_dir, mid, "final_1", calendar, bundle.origin, recal[mid],
                                     registered_name=f"ltv_test_{mid.lower()}", calibration="tune_fit")
    return out


def test_a_saved_fit_becomes_a_registered_model_without_retraining(logged, demo_bundle_dir):
    for mid, info in logged.items():
        assert info["version"] >= 1 and info["registered_name"] == f"ltv_test_{mid.lower()}" and info["calibration"] == "tune_fit"
        files = mlflow.artifacts.download_artifacts(info["model_uri"])
        meta = json.loads(next(p for p in __import__("pathlib").Path(files).rglob("meta.json")).read_text())
        assert meta["model_id"] == mid and meta["fit"] == "final_1"
        reqs = (__import__("pathlib").Path(files) / "requirements.txt").read_text()
        assert ("lightgbm" in reqs) == (mid != "C1") and ("torch" in reqs) == (mid == "C1")
        assert "scikit-learn" not in reqs and "pyspark" not in reqs, "the served model needs neither"
        assert (__import__("pathlib").Path(files) / "code" / "ltv" / "serve.py").exists(), "the model carries its own code"
    with pytest.raises(FileNotFoundError, match="Run its notebook first"):
        serve.model_files(demo_bundle_dir, "C2", "final_1")


def test_the_loaded_model_gives_the_same_numbers_as_the_saved_fit(logged, bundle, players):
    hist, rows = players
    for mid, info in logged.items():
        serve.set_alias(info["registered_name"], "candidate", info["version"])
        model = serve.load(f"models:/{info['registered_name']}@candidate")
        out = serve.predict(model, rows)
        assert list(out.columns) == list(serve.OUTPUT_COLS) and (out["origin_week"] == bundle.origin).all()
        assert out["user_id"].tolist() == rows["user_id"].tolist()
        for i, h in enumerate(hist):                                # the playground scores the same fit directly
            want = core.forecast(bundle, h["hours"], h["spend"])["models"][mid]["totals"]
            for t in serve.TARGET_COLS:
                assert out.loc[i, t] == pytest.approx(want[t], rel=1e-4, abs=1e-9), (mid, h["id"], t)
        raw = serve.predict(model, rows, calibrated=False)
        assert not np.allclose(raw["exp_hours"], out["exp_hours"]), "the calibration layer travels with the model"
        assert out["p_reengage"].between(0, 1).all() and (out["exp_hours"] >= 0).all()


def test_scoring_week_calendar_and_population(logged, bundle, players, calendar):
    hist, rows = players
    model = serve.load(logged["A2"]["model_uri"])
    # an earlier scoring week reads only the history up to it
    t = bundle.origin - 30
    early = serve.predict(model, rows, origin_week=t)
    cut = rows.copy()
    cut["hours"] = [h[: max(t - f + 1, 0)] for h, f in zip(rows["hours"], rows["first_week"])]
    cut["spend"] = [s[: max(t - f + 1, 0)] for s, f in zip(rows["spend"], rows["first_week"])]
    arrived = (rows["first_week"] <= t).to_numpy()
    again = serve.predict(model, cut[arrived], origin_week=t)
    np.testing.assert_allclose(early.loc[arrived, "p_reengage"], again["p_reengage"], rtol=1e-12)
    assert early.loc[~arrived, "p_reengage"].isna().all() and not early.loc[~arrived, "in_population"].any()
    # the default scoring week is the last week the arrays reach
    assert (serve.predict(model, rows)["origin_week"] == bundle.origin).all()
    # a player quiet for more than 104 weeks is scored but flagged
    quiet = pd.DataFrame({"user_id": ["gone"], "first_week": [bundle.origin - 159], "hours": [[5.0] + [0.0] * 159], "spend": [[0.0] * 160]})
    out = serve.predict(model, quiet)
    assert not out["in_population"].iloc[0] and np.isfinite(out["p_reengage"].iloc[0])
    # the calendar can be passed, and scoring past the calendar the model holds is refused, not guessed
    same = serve.predict(model, rows, calendar=calendar)
    np.testing.assert_allclose(same["exp_hours"], serve.predict(model, rows)["exp_hours"], rtol=1e-12)
    with pytest.raises(Exception, match="beyond the season calendar"):
        serve.predict(model, rows, origin_week=len(calendar) + 5)
    with pytest.raises(Exception, match="same number of weeks"):
        serve.predict(model, pd.DataFrame({"user_id": ["x"], "first_week": [3], "hours": [[1.0, 2.0]], "spend": [[0.0]]}))
