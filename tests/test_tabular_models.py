"""Baselines R1 and R2 and the tree suite, on synthetic data (no Spark)."""
import numpy as np
import pandas as pd
import pytest

from ltv.eval import metrics as M
from ltv.models.base import frame_from_store
from ltv.models.baselines import Persistence, SegmentLookup
from ltv.models.bgbb import BGBB
from ltv.models.trees import TreeSuite, ziln_quantiles
from ltv.synth.generator import generate, to_store
from ltv.train.timebox import TimeBox, random_search

TRAIN_ORIGINS, VAL_ORIGIN, TEST_ORIGIN = (120, 150, 180), 215, 267


@pytest.fixture(scope="module")
def frames():
    st = to_store(generate(5000, 320, seed=9))
    parts = [frame_from_store(st, st.rows_for(o, "train"), o) for o in TRAIN_ORIGINS]
    tr = pd.concat(parts, ignore_index=True)
    va = frame_from_store(st, st.rows_for(VAL_ORIGIN, "val"), VAL_ORIGIN)
    te = frame_from_store(st, st.rows_for(TEST_ORIGIN), TEST_ORIGIN)
    return tr, va, te


@pytest.fixture(scope="module")
def suite(frames):
    tr, va, _ = frames
    s = TreeSuite(max_rounds=120, early_stop=20, seed=1)
    s.fit(tr, va)
    return s


def test_segment_lookup_is_calibrated_in_sample_and_keys_on_past_returns(frames):
    tr, _, te = frames
    r1 = SegmentLookup().fit(tr)
    p = r1.predict(tr, 0)
    assert p.exp_active_weeks.mean() == pytest.approx(tr["active_weeks_52"].mean(), rel=0.02)
    assert p.p_reengage.mean() == pytest.approx(tr["any_active_52"].mean(), abs=0.01)
    pt = r1.predict(te, TEST_ORIGIN)
    lapsed = te["wk_since_active"].to_numpy() >= 13
    many, none = lapsed & (te["n_returns_26"].to_numpy() >= 2), lapsed & (te["n_returns_26"].to_numpy() == 0)
    assert pt.p_reengage[many].mean() > pt.p_reengage[none].mean() + 0.15
    assert (np.diff(pt.cum_hours, axis=1) >= -1e-9).all()


def test_persistence_repeats_the_last_year(frames):
    _, _, te = frames
    p = Persistence().predict(te, TEST_ORIGIN)
    np.testing.assert_allclose(p.exp_hours, te["hrs_52"])
    np.testing.assert_allclose(p.exp_active_weeks, te["aw_52"])
    assert set(np.unique(p.p_reengage)) <= {0.02, 0.98}


def test_bgbb_runs_on_features_and_is_ordered_sensibly(frames):
    tr, _, te = frames
    p = BGBB().fit(tr).predict(te, TEST_ORIGIN)
    assert np.isfinite(p.exp_active_weeks).all() and ((p.p_reengage >= 0) & (p.p_reengage <= 1)).all()
    assert M.auc(te["any_active_52"], p.p_reengage) > 0.7
    assert (np.diff(p.cum_hours, axis=1) >= -1e-9).all()
    np.testing.assert_allclose(p.weekly_p_active.sum(1), p.exp_active_weeks, rtol=1e-6)


def test_trees_beat_the_baselines_where_they_should(frames, suite):
    tr, _, te = frames
    y = te["any_active_52"].to_numpy()
    a1 = suite.predict_a1(te, TEST_ORIGIN)
    assert M.auc(y, a1.p_reengage) > M.auc(y, Persistence().predict(te, TEST_ORIGIN).p_reengage) + 0.1
    assert M.auc(y, a1.p_reengage) > 0.85
    assert M.aggregate_bias(te["active_weeks_52"], a1.exp_active_weeks) == pytest.approx(1.0, abs=0.15)
    assert M.normalized_gini(te["spend_52"].to_numpy(), a1.exp_spend) > 0.5


def test_a2_is_coherent_by_construction(frames, suite):
    _, _, te = frames
    p = suite.predict_a2(te, TEST_ORIGIN)
    assert (np.diff(p.cum_hours, axis=1) >= 0).all() and (np.diff(p.cum_spend, axis=1) >= 0).all()
    np.testing.assert_allclose(p.cum_hours[:, -1], p.exp_hours)
    np.testing.assert_allclose(p.weekly_p_active.sum(1), p.exp_active_weeks, rtol=1e-6)
    assert (p.exp_active_weeks <= 52 + 1e-9).all()
    assert (p.q_hours[:, 0] <= p.q_hours[:, 1] + 1e-9).all() and (p.q_hours[:, 1] <= p.q_hours[:, 2] + 1e-9).all()
    cover = M.coverage(te["hours_52"].to_numpy(), p.q_hours[:, 0], p.q_hours[:, 2])
    assert 0.6 < cover < 0.97, "the 10-90% band should cover very roughly 80%"


def test_final_fit_uses_the_frozen_tree_counts(frames, suite):
    tr, _, te = frames
    final = TreeSuite(params=suite.params, max_rounds=suite.max_rounds, seed=1, rounds=dict(suite.rounds))
    final.fit(tr, None)
    assert final.rounds == suite.rounds
    assert all(final.boosters[k].num_trees() == suite.rounds[k] for k in ("any_active", "hours", "spd_b4"))


def test_a_time_boxed_fit_resumes_model_by_model(frames, suite):
    """A fit stopped by its time box continues from the saved state and ends with the same models."""
    from ltv.train.timebox import TimeBox
    tr, _, te = frames
    kw = dict(params=suite.params, max_rounds=suite.max_rounds, seed=1)
    saved = []
    part = TreeSuite(rounds=dict(suite.rounds), **kw)

    class OneModelBox:                       # lets exactly one model through, then reports expiry
        def expired(self):
            return len(part.boosters) >= 1

    part.fit(tr, None, box=OneModelBox(), on_model=lambda: saved.append(part.state()))
    assert not part.complete and len(part.boosters) == 1 and len(saved) == 1
    resumed = TreeSuite.from_state(saved[-1])
    resumed.fit(tr, None, box=TimeBox(None), on_model=lambda: saved.append(resumed.state()))
    assert resumed.complete and resumed.rounds == suite.rounds and len(saved) == len(resumed.boosters)
    straight = TreeSuite(rounds=dict(suite.rounds), **kw)
    straight.fit(tr, None)
    assert straight.complete
    for make in (lambda s: s.predict_a1(te, TEST_ORIGIN), lambda s: s.predict_a2(te, TEST_ORIGIN)):
        a, b = make(resumed), make(straight)
        np.testing.assert_allclose(a.exp_hours, b.exp_hours, rtol=1e-6)
        np.testing.assert_allclose(a.exp_spend, b.exp_spend, rtol=1e-6)


def test_suite_survives_a_save_and_load(frames, suite):
    _, _, te = frames
    again = TreeSuite.from_state(suite.state())
    np.testing.assert_allclose(again.predict_a2(te, 0).exp_hours, suite.predict_a2(te, 0).exp_hours, rtol=1e-9)
    np.testing.assert_allclose(again.predict_a1(te, 0).exp_spend, suite.predict_a1(te, 0).exp_spend, rtol=1e-9)


def test_ziln_quantiles_against_sampling():
    rng = np.random.default_rng(0)
    p, mu, sg = 0.4, 2.0, 0.8
    draws = np.where(rng.random(400_000) < p, rng.lognormal(mu, sg, 400_000), 0.0)
    q = ziln_quantiles(np.array([p]), np.array([mu]), np.array([sg]), (0.5, 0.7, 0.9, 0.99))[0]
    np.testing.assert_allclose(q, np.quantile(draws, [0.5, 0.7, 0.9, 0.99]), rtol=0.03, atol=1e-9)
    assert q[0] == 0.0, "the median is zero when most of the mass is at zero"


def test_random_search_resumes_and_respects_the_time_box(tmp_path):
    calls = []

    def objective(params):
        calls.append(params)
        return (params["x"] - 3.0) ** 2, {}
    space = {"x": ("uniform", 0.0, 6.0), "k": [1, 2, 3]}
    log = tmp_path / "trials.jsonl"
    a = random_search(space, objective, 4, log, TimeBox(None), seed=7)
    assert a["n_done"] == 4 and not a["complete"] is False
    b = random_search(space, objective, 10, log, TimeBox(None), seed=7)
    assert b["n_done"] == 10 and b["complete"] and len(calls) == 10, "the first four trials were not repeated"
    straight = random_search(space, objective, 10, tmp_path / "other.jsonl", TimeBox(None), seed=7)
    assert straight["params"] == b["params"] and straight["loss"] == b["loss"]
    with pytest.raises(RuntimeError, match="time box expired"):
        random_search(space, objective, 3, tmp_path / "none.jsonl", TimeBox(0.0), seed=1)


def test_files_are_written_through_local_disk(tmp_path):
    """Checkpoints, shards and logs must not be written in place: Unity Catalog volumes reject
    random-access and append writes. Everything goes through ltv.io.write_via_local."""
    from ltv.io import write_via_local
    from ltv.train.timebox import TrialLog, load_pickle, save_pickle
    seen = []

    def writer(local):
        seen.append(local)
        open(local, "wb").write(b"abc")
    target = tmp_path / "deep" / "file.bin"
    write_via_local(target, writer)
    assert target.read_bytes() == b"abc" and not str(seen[0]).startswith(str(tmp_path))
    assert not (tmp_path / "deep" / "file.bin.tmp").exists()
    save_pickle(tmp_path / "x.pkl", {"a": 1})
    assert load_pickle(tmp_path / "x.pkl") == {"a": 1} and load_pickle(tmp_path / "none.pkl", 7) == 7
    log = TrialLog(tmp_path / "log.jsonl")
    log.append({"trial": 0, "loss": 1.0})
    log.append({"trial": 1, "loss": 0.5})
    assert [r["trial"] for r in log.read()] == [0, 1]
