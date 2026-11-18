"""Metrics against values worked out by hand, and the evaluation helpers on constructed cases."""
import numpy as np
import pandas as pd
import pytest

from ltv.eval import metrics as M
from ltv.eval.calibration import Recalibrator, SegmentConformal
from ltv.eval.evaluate import coherence_table, paired_bootstrap
from ltv.eval.revisions import cumulative_revision, revision_summary, weekly_activity_revision
from ltv.eval.segments import dormancy_bucket, returner_bucket, tenure_bucket


def test_log_loss_by_hand():
    assert M.log_loss([1, 0], [0.8, 0.4]) == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)


def test_auc_by_hand_and_with_ties():
    assert M.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert M.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert np.isnan(M.auc([1, 1], [0.2, 0.3]))


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.3
    s = np.round(rng.random(5000) + 0.3 * y, 2)            # many ties
    assert M.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_poisson_and_tweedie_deviance():
    assert M.poisson_deviance([2.0], [2.0]) == pytest.approx(0.0, abs=1e-12)
    assert M.poisson_deviance([0.0], [1.5]) == pytest.approx(3.0)                       # 2 * mu when y = 0
    assert M.poisson_deviance([3.0], [1.0]) == pytest.approx(2 * (3 * np.log(3) - 2))
    assert M.tweedie_deviance([4.0], [4.0]) == pytest.approx(0.0, abs=1e-9)
    assert M.tweedie_deviance([0.0], [4.0]) == pytest.approx(2 * 4 ** 0.5 / 0.5)        # 2 mu^(2-p) / (2-p)
    assert M.tweedie_deviance([1.0], [2.0]) > 0
    from sklearn.metrics import mean_tweedie_deviance
    rng = np.random.default_rng(1)
    y, mu = rng.gamma(1, 5, 200) * (rng.random(200) < 0.4), rng.gamma(2, 2, 200) + 0.1
    assert M.tweedie_deviance(y, mu) == pytest.approx(mean_tweedie_deviance(y, mu, power=1.5), rel=1e-9)


def test_normalized_gini():
    y = np.array([0.0, 0.0, 1.0, 5.0, 20.0])
    assert M.normalized_gini(y, y) == pytest.approx(1.0)
    assert M.normalized_gini(y, y * 3 + 1) == pytest.approx(1.0), "any order-preserving prediction is perfect"
    assert M.normalized_gini(y, -y) == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    big = rng.lognormal(0, 2, 20000)
    assert abs(M.normalized_gini(big, rng.random(20000))) < 0.05, "a random order scores about zero"


def test_capture_and_bias():
    y = np.array([100.0, 50.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pred = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert M.top_capture(y, pred, 0.1) == pytest.approx(100 / 160)
    assert M.top_capture(y, pred, 0.2) == pytest.approx(150 / 160)
    assert M.aggregate_bias(y, pred) == pytest.approx(45 / 160)
    assert M.tail_bias(y, pred, 0.2) == pytest.approx(17 / 150)


def test_ece_is_zero_for_a_calibrated_forecast_and_large_for_a_bad_one():
    rng = np.random.default_rng(2)
    p = rng.random(200_000)
    y = rng.random(200_000) < p
    assert M.ece(y, p) < 0.01
    assert M.ece(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_segment_buckets():
    assert dormancy_bucket([0, 1, 4, 5, 26, 27, 104, 105, np.nan]).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert tenure_bucket([0, 7, 8, 52, 53]).tolist() == [0, 0, 1, 1, 2]
    assert returner_bucket([0, 1, 2, 7, np.nan]).tolist() == [0, 1, 2, 2, 0]


def test_recalibration_fixes_level_and_keeps_order():
    rng = np.random.default_rng(3)
    n = 40_000
    truth = rng.gamma(2.0, 10.0, n)
    y = rng.poisson(truth).astype(float)
    df = pd.DataFrame({"p_reengage": rng.random(n), "any_active_52": 0.0, "exp_active_weeks": truth * 1.6,
                       "active_weeks_52": y, "exp_hours": truth * 1.6, "hours_52": y, "p_spend": rng.random(n),
                       "any_spend_52": 0.0, "exp_spend": truth * 1.6, "spend_52": y,
                       "cum_hours": list(np.outer(truth * 1.6, [0.1, 0.3, 0.5, 0.8, 1.0]))})
    val, test = df.iloc[: n // 2], df.iloc[n // 2:]
    out = Recalibrator().fit(val).apply(test)
    assert M.aggregate_bias(test["hours_52"], test["exp_hours"]) == pytest.approx(1.6, abs=0.03)
    assert M.aggregate_bias(out["hours_52"], out["exp_hours"]) == pytest.approx(1.0, abs=0.03)
    order = np.argsort(test["exp_hours"].to_numpy())
    assert (np.diff(out["exp_hours"].to_numpy()[order]) >= -1e-9).all(), "isotonic never reorders users"
    c = np.stack(out["cum_hours"].to_numpy())
    np.testing.assert_allclose(c[:, -1], out["exp_hours"], rtol=1e-6)
    assert (np.diff(c, axis=1) >= 0).all()


def test_conformal_intervals_cover_on_exchangeable_data():
    rng = np.random.default_rng(4)
    n = 60_000
    pred = rng.gamma(2.0, 5.0, n)
    seg = rng.integers(0, 3, n)
    y = pred * rng.lognormal(0, 0.5 + 0.3 * seg, n)                    # noisier in some segments
    conf = SegmentConformal((0.8, 0.95)).fit(pred[: n // 2], y[: n // 2], seg[: n // 2])
    for lvl in (0.8, 0.95):
        lo, hi = conf.interval(pred[n // 2:], seg[n // 2:], lvl)
        assert M.coverage(y[n // 2:], lo, hi) == pytest.approx(lvl, abs=0.02)
        for s in range(3):                                             # and within each segment
            m = seg[n // 2:] == s
            assert M.coverage(y[n // 2:][m], lo[m], hi[m]) == pytest.approx(lvl, abs=0.03)


def test_paired_bootstrap_detects_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(5)
    n = 4000
    p_true = rng.random(n)
    y = (rng.random(n) < p_true).astype(float)
    good = np.clip(p_true + 0.02 * rng.normal(size=n), 0.01, 0.99)
    bad = np.clip(0.5 + 0.2 * (p_true - 0.5), 0.01, 0.99)
    r = paired_bootstrap("p_reengage", y, good, bad, 100)
    assert r["a_better"] == 1 and r["ci_high"] < 0
    # Two equally good forecasts: a 95% interval should call a difference only about 5% of the time.
    calls = 0
    for seed in range(12):
        r2 = np.random.default_rng(100 + seed)
        a = np.clip(p_true + 0.02 * r2.normal(size=n), 0.01, 0.99)
        b = np.clip(p_true + 0.02 * r2.normal(size=n), 0.01, 0.99)
        calls += paired_bootstrap("p_reengage", y, a, b, 60, seed)["a_better"] != 0
    assert calls <= 3


def test_coherence_table_counts_violations():
    df = pd.DataFrame({"p_reengage": [0.9, 0.2, 0.5], "exp_active_weeks": [10.0, 0.1, 30.0],
                       "exp_hours": [50.0, 1.0, 9.0],
                       "weekly_p_active": [np.full(52, 10 / 52), np.full(52, 0.1 / 52), np.full(52, 30 / 52)],
                       "cum_hours": [np.array([5, 15, 30, 40, 50.0]), np.array([0.1, 0.3, 0.5, 0.8, 1.0]),
                                     np.array([1, 3, 2, 8, 9.0])],
                       "cum_spend": [None, None, None]})
    t = coherence_table(df, {}).set_index("check")["violations"]
    assert t["p_reengage_above_active_weeks"] == 1          # 0.2 > 0.1
    assert t["cum_hours_decreases"] == 1
    assert t["p_reengage_below_a_weekly_p"] == 1            # 0.5 < 30/52
    assert t["active_weeks_outside_0_52"] == 0
    assert "cum_spend_decreases" not in t.index


def test_revisions_of_a_true_model_average_zero_and_selection_on_the_outcome_breaks_that():
    """Simulate a two-state chain and forecast with the truth. Mean revision is zero, but among users
    who had a quiet week it is negative: the reason segments must be defined before the outcome."""
    rng = np.random.default_rng(6)
    n, H = 60_000, 52
    A = np.array([[0.9, 0.1], [0.05, 0.95]])
    p_act = np.array([0.8, 0.0])
    state = (rng.random(n) < 0.5).astype(int)

    def weekly(dist):
        out, d = np.zeros((len(dist), H)), dist.copy()
        for h in range(H):
            d = d @ A
            out[:, h] = d @ p_act
        return out
    w_t = weekly(np.eye(2)[state])
    nxt = np.where(rng.random(n) < A[state, 0], 0, 1)
    active = (rng.random(n) < p_act[nxt]).astype(float)
    # The forecaster knows the state (a perfect filter), so the forecast at t+1 conditions on it.
    rev = weekly_activity_revision(w_t, weekly(np.eye(2)[nxt]), active)
    s = revision_summary(rev, state, w_t.sum(1))
    assert abs(s.set_index("segment").loc["all", "z"]) < 3.5
    assert (s["z"].abs() < 4).all(), "zero within segments defined at t"
    assert rev[active == 0].mean() < -0.5, "conditioning on a quiet week gives a negative mean revision"
    cum = np.cumsum(w_t, axis=1)[:, [3, 12, 25, 38, 51]]
    assert cumulative_revision(cum, cum, cum[:, 1]).mean() == pytest.approx(
        (cum[:, 1] + cum[:, 3] - cum[:, 4]).mean())
