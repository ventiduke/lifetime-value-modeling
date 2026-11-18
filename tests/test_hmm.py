"""Candidate B: the HMMs. Fitted on synthetic data whose latent process is known."""
import numpy as np
import pytest

from ltv.data.np_features import F, tabular_features
from ltv.data.splits import project_calendar
from ltv.models import kernels_np as kn
from ltv.models.hmm import (HMM, calendar_buckets, filter_users, fit_hmm, forecast_matrices, future_buckets,
                            init_params, predict_hmm, spec_b1, spec_b2)
from ltv.synth.generator import generate, oracle_predict, to_store
from ltv.train.timebox import TimeBox

ORIGIN = 267


@pytest.fixture(scope="module")
def data():
    d = generate(6000, 320, seed=5)
    return d, to_store(d)


@pytest.fixture(scope="module")
def b2(data):
    _, st = data
    model, info = fit_hmm(spec_b2(4, 2, 3, 3), [st], ORIGIN, TimeBox(None), max_iters=40, seed=1)
    return model, info


@pytest.fixture(scope="module")
def b1(data):
    _, st = data
    model, info = fit_hmm(spec_b1(6), [st], ORIGIN, TimeBox(None), max_iters=40, seed=1)
    return model, info


def test_b2_structure():
    spec = spec_b2(2, 2, 3, 2)
    assert spec.n_states == 20 and spec.dormant.sum() == 12
    m = spec.trans_mask
    assert not m[:10, 10:].any() and not m[10:, :10].any(), "spend classes never mix"
    assert not m[:5, 5:10].any(), "user types never mix"
    assert m[2, 3] and not m[2, 4] and not m[3, 2], "dormancy is a one-way chain"
    assert m[4, 0] and m[4, 1], "any dormant state can return to activity"
    assert not m[0, 3], "a lapse starts at the first dormant state"


def test_em_never_decreases_the_likelihood(b1, b2):
    for _, info in (b1, b2):
        assert len(info["history"]) >= 5
        assert (np.diff(info["history"]) > -1e-9).all()


def test_fit_resumes_exactly(data, tmp_path):
    _, st = data
    spec = spec_b1(4)
    straight, _ = fit_hmm(spec, [st], ORIGIN, TimeBox(None), max_iters=6, seed=3)
    ck = tmp_path / "hmm.pkl"
    _, i1 = fit_hmm(spec, [st], ORIGIN, TimeBox(None), ckpt=ck, max_iters=3, seed=3)
    resumed, i2 = fit_hmm(spec, [st], ORIGIN, TimeBox(None), ckpt=ck, max_iters=6, seed=3)
    assert i1["iters"] == 3 and i2["iters"] == 6
    for k, v in straight.params.items():
        np.testing.assert_allclose(resumed.params[k], v, rtol=1e-10, atol=1e-12, err_msg=k)


def test_an_expired_time_box_changes_nothing(data, tmp_path):
    _, st = data
    model, info = fit_hmm(spec_b1(4), [st], ORIGIN, TimeBox(0.0), ckpt=tmp_path / "x.pkl", seed=3)
    assert info["iters"] == 0 and not info["complete"]


def test_shards_give_the_same_fit_as_one_store(data):
    _, st = data
    half = st.n_users // 2
    parts = [st.subset(np.arange(half)), st.subset(np.arange(half, st.n_users))]
    start = init_params(spec_b1(4), 2, 1.5, 2.5)
    a, _ = fit_hmm(spec_b1(4), [st], ORIGIN, TimeBox(None), max_iters=4, start=start)
    b, _ = fit_hmm(spec_b1(4), parts, ORIGIN, TimeBox(None), max_iters=4, start=start)
    for k in a.params:
        np.testing.assert_allclose(a.params[k], b.params[k], rtol=1e-8, atol=1e-10, err_msg=k)


def test_fit_reads_nothing_after_the_origin(data):
    """Scrambling every week after the origin must not change the fit or the forecasts."""
    _, st = data
    import copy
    st2 = copy.deepcopy(st)
    rng = np.random.default_rng(0)
    st2.hours[:, ORIGIN + 1:] = rng.random(st2.hours[:, ORIGIN + 1:].shape).astype(np.float32) * 9
    st2.spend[:, ORIGIN + 1:] = 5.0
    a, _ = fit_hmm(spec_b1(4), [st], ORIGIN, TimeBox(None), max_iters=3, seed=2)
    b, _ = fit_hmm(spec_b1(4), [st2], ORIGIN, TimeBox(None), max_iters=3, seed=2)
    rows = st.rows_for(ORIGIN, "test")
    np.testing.assert_array_equal(predict_hmm(a, st, rows, ORIGIN).p_reengage,
                                  predict_hmm(b, st2, rows, ORIGIN).p_reengage)


def test_b2_recovers_planted_spend_classes(b2):
    """The generator has spender classes with weekly spend rates of about 0.001, 0.10 and 0.25 in the top state."""
    model, _ = b2
    q = model.params["q_spend"].reshape(3, 4, 5)[:, 0, 0]
    assert q[0] < 0.01 and 0.05 < q[1] < 0.16 and 0.17 < q[2] < 0.35
    amounts = model.params["s_mean"].reshape(3, 4, 5)[:, 0, 0]
    assert amounts[2] > 2.5 * amounts[1]


def test_forecasts_are_calibrated_and_track_the_oracle(data, b2):
    d, st = data
    model, _ = b2
    rows = st.rows_for(ORIGIN, "test")
    y = st.labels(rows, ORIGIN)
    p = predict_hmm(model, st, rows, ORIGIN)
    sw, _ = project_calendar(st.season_week, st.season_idx, ORIGIN, 52)
    orc = oracle_predict(d.truth, sw, ORIGIN).set_index("user_id").loc[st.user_ids[rows]]
    assert np.corrcoef(p.p_reengage, orc["p_reengage"])[0, 1] > 0.75
    assert p.exp_active_weeks.sum() / y["active_weeks_52"].sum() == pytest.approx(1.0, abs=0.12)
    assert p.exp_hours.sum() / y["hours_52"].sum() == pytest.approx(1.0, abs=0.15)
    assert p.p_reengage.mean() == pytest.approx(y["any_active_52"].mean(), abs=0.05)
    # coherence, by construction
    assert (p.p_reengage <= p.exp_active_weeks + 1e-9).all()
    assert (p.exp_active_weeks <= 52 * p.p_reengage + 1e-6).all()
    assert (np.diff(p.cum_hours, axis=1) >= -1e-9).all()
    np.testing.assert_allclose(p.weekly_p_active.sum(1), p.exp_active_weeks, rtol=1e-4)
    assert (p.p_reengage >= p.weekly_p_active.max(1) - 1e-6).all()


def test_b2_tells_serial_returners_from_churned_players_and_b1_cannot(data, b1, b2):
    """Among players lapsed 13+ weeks, those with a history of returning do come back more often.
    The lapse-aware model must see that; a plain HMM gives everyone the same answer."""
    _, st = data
    rows = st.rows_for(ORIGIN)
    y = st.labels(rows, ORIGIN)["any_active_52"]
    X = tabular_features(st, rows, ORIGIN)
    lapsed = X[:, F["wk_since_active"]] >= 13
    none, many = lapsed & (X[:, F["n_returns_26"]] == 0), lapsed & (X[:, F["n_returns_26"]] >= 2)
    assert none.sum() > 100 and many.sum() > 100
    assert y[many].mean() - y[none].mean() > 0.25, "the planted effect is in the data"
    p2 = predict_hmm(b2[0], st, rows, ORIGIN).p_reengage
    p1 = predict_hmm(b1[0], st, rows, ORIGIN).p_reengage
    assert p2[many].mean() - p2[none].mean() > 0.15
    assert abs(p1[many].mean() - p1[none].mean()) < 0.08
    assert abs(p2[many].mean() - y[many].mean()) < abs(p1[many].mean() - y[many].mean())


def test_numpy_kernels_match_the_model(data, b2):
    """Week-by-week NumPy filtering equals the batched forward pass, and a quiet run collapses to one matrix."""
    _, st = data
    model, _ = b2
    params, buckets = model.export(), calendar_buckets(st)
    rows = st.rows_for(ORIGIN, "test")[:200]
    ref, _ = filter_users(model, st, rows, ORIGIN)
    alpha = np.zeros((len(rows), model.spec.n_states))
    for w in range(int(st.first_week[rows].min()), ORIGIN + 1):
        h, s = st.hours[rows, w].astype(float), st.spend[rows, w].astype(float)
        new, old = st.first_week[rows] == w, st.first_week[rows] < w
        if new.any():
            alpha[new] = kn.hmm_start(params, h[new], s[new])
        if old.any():
            alpha[old] = kn.hmm_step(alpha[old], params, int(buckets[w]), h[old], s[old])
    np.testing.assert_allclose(alpha, ref, rtol=1e-6, atol=1e-9)
    # A run of 30 weeks with no rows: one cached matrix equals 30 single steps.
    run = buckets[ORIGIN + 1:ORIGIN + 31]
    step = alpha.copy()
    zeros = np.zeros(len(rows))
    for b in run:
        step = kn.hmm_step(step, params, int(b), zeros, zeros)
    np.testing.assert_allclose(kn.hmm_advance_quiet(alpha, kn.hmm_quiet_matrix(params, run)), step,
                               rtol=1e-6, atol=1e-9)


def test_forecast_is_a_linear_map_of_the_state(data, b2):
    _, st = data
    model, _ = b2
    V = forecast_matrices(model.export(), future_buckets(st, ORIGIN))
    k = model.spec.n_states
    assert V["weekly_p_active"].shape == (k, 52) and V["cum_hours"].shape == (k, 5)
    assert ((V["p_no_active"] >= 0) & (V["p_no_active"] <= 1)).all()
    dormant_long = np.flatnonzero(model.spec.dormant)[2]
    active = np.flatnonzero(~model.spec.dormant)[0]
    assert V["p_no_active"][dormant_long] > V["p_no_active"][active]
