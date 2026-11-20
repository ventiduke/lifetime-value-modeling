"""Example players for plotting (ltv.eval.examples). No Spark: models enter as plain functions."""
import copy

import numpy as np
import pandas as pd
import pytest

from ltv.data.sequences import CUM_HORIZONS
from ltv.eval import examples as ex
from ltv.eval.calibration import Recalibrator
from ltv.models.base import LABEL_OF, TARGETS, frame_from_store
from ltv.models.baselines import Persistence
from ltv.synth.generator import generate, to_store

ORIGIN = 240


@pytest.fixture(scope="module")
def store():
    return to_store(generate(1500, 320, seed=4))


def persistence(st, rows, origin):
    """A stand-in model: last year repeated. It reads the history up to the origin and nothing else."""
    return Persistence().predict(frame_from_store(st, rows, origin, with_labels=False), origin)


def test_weekly_path_keeps_the_totals_and_follows_weekly_activity():
    cum = np.array([[4.0, 13.0, 26.0, 39.0, 52.0], [0.0, 0.0, 10.0, 10.0, 30.0]])
    flat = ex.weekly_path(cum, None)
    np.testing.assert_allclose(flat[0], 1.0)                                    # one hour a week, spread evenly
    np.testing.assert_allclose(flat.sum(1), cum[:, -1])
    np.testing.assert_allclose(np.cumsum(flat, 1)[:, [h - 1 for h in CUM_HORIZONS]], cum)
    p = np.zeros((2, 52))
    p[:, 13] = 0.9                                                              # only week 14 is likely
    shaped = ex.weekly_path(cum, p)
    assert shaped[1, 13] == pytest.approx(10.0) and shaped[1, 14:26].sum() == pytest.approx(0.0)
    np.testing.assert_allclose(shaped.sum(1), cum[:, -1])                       # buckets with no activity spread evenly
    assert (shaped >= 0).all()


def test_pick_examples_is_stable_and_respects_groups(store):
    rows = store.rows_for(ORIGIN, "test")
    frame = frame_from_store(store, rows, ORIGIN)
    a = ex.pick_examples(frame, per_group=5, seed=1)
    b = ex.pick_examples(frame.sample(frac=1.0, random_state=3), per_group=5, seed=1)
    assert set(zip(a.user_id, a.example_group)) == set(zip(b.user_id, b.example_group)), "row order must not matter"
    assert a["user_id"].is_unique and a["example_group"].value_counts().max() <= 5
    assert {"serial returner, long dormant", "first long lapse", "playing, steady"} <= set(a["example_group"])
    f = frame.set_index("user_id").loc[a.set_index("example_group").loc[["serial returner, long dormant"], "user_id"]]
    assert (f["n_returns_26"] >= 2).all() and (f["wk_since_active"] >= 27).all()
    # players asked for by id, and the ones two models disagree on most
    ask = [frame["user_id"].iloc[7], "no-such-user"]
    gap = pd.Series(np.arange(len(frame), dtype=float), index=frame["user_id"].to_numpy())
    c = ex.pick_examples(frame, per_group=3, seed=1, user_ids=ask, disagreement=gap).set_index("user_id")
    assert c.loc[ask[0], "example_group"] == "requested" and "no-such-user" not in c.index
    top = list(gap.sort_values(ascending=False).index[:3])
    assert (c.loc[top, "example_group"] == "models disagree most").all()


def test_example_frames_hold_actuals_paths_and_the_rolling_view(store):
    rows = store.rows_for(ORIGIN, "test")[:40]
    groups = pd.DataFrame({"user_id": store.user_ids[rows], "example_group": "g"})
    as_of = range(ORIGIN - 20, ORIGIN + 11)
    raw = ex.example_frames(store, rows, {"R2": persistence}, ORIGIN, as_of, fit="final_1", last_train_origin=ORIGIN - 5)
    week_start = pd.Series(pd.date_range("2020-01-06", periods=store.n_weeks + 60, freq="7D").date)
    out = ex.finish(raw, groups, week_start)
    assert list(out.columns) == ex.WEEKLY_COLUMNS and set(out["view"]) == set(ex.VIEWS)
    assert out["week_start"].notna().all() and (out["fit"] == "final_1").all() and not out["calibrated"].any()

    u = store.user_ids[rows[0]]
    one = out[out.user_id == u]
    actual = one[one.view == "actual"].set_index("week_idx")
    first = int(store.first_week[rows[0]])
    assert actual.index.min() == first and actual.index.max() == store.last_week
    np.testing.assert_allclose(actual["hours"], store.hours[rows[0], first:], rtol=1e-6)
    after = actual[actual.index > ORIGIN]
    np.testing.assert_allclose(after["cum_hours"], np.cumsum(store.hours[rows[0], ORIGIN + 1:]), rtol=1e-5)
    assert actual.loc[actual.index <= ORIGIN, "cum_hours"].isna().all()

    p = persistence(store, rows[:1], ORIGIN)
    path = one[one.view == "path"].sort_values("week_idx")
    assert list(path["week_idx"]) == list(range(ORIGIN + 1, ORIGIN + 53)) and (path["weeks_from_origin"] > 0).all()
    assert path["hours"].sum() == pytest.approx(float(p.exp_hours[0]), rel=1e-6)
    assert path["cum_hours"].iloc[-1] == pytest.approx(float(p.exp_hours[0]), rel=1e-6)

    roll = one[one.view == "as_of"].set_index("week_idx")
    assert list(roll.index) == [t for t in as_of if t >= first]
    assert roll.loc[ORIGIN, "exp_hours_52"] == pytest.approx(float(p.exp_hours[0]), rel=1e-6)
    assert roll.loc[ORIGIN, "hours"] == pytest.approx(float(p.cum_hours[0, 0]) / 4, rel=1e-6)
    assert roll.loc[ORIGIN - 5, "after_training_period"] is False and roll.loc[ORIGIN - 4, "after_training_period"] is True
    assert one[one.view != "as_of"][list(ex.AS_OF_COLS)].isna().all().all()


def test_the_rolling_view_never_reads_ahead(store):
    """Changing every week after t must leave every as-of row at or before t exactly as it was."""
    rows = store.rows_for(ORIGIN, "test")[:25]
    t = ORIGIN - 6
    later = copy.deepcopy(store)
    later.hours[:, t + 1:] = 33.0
    later.spend[:, t + 1:] = 9.0
    as_of = range(ORIGIN - 12, ORIGIN + 1)
    a = ex.example_frames(store, rows, {"R2": persistence}, ORIGIN, as_of)
    b = ex.example_frames(later, rows, {"R2": persistence}, ORIGIN, as_of)
    keep = lambda d: d[(d.view == "as_of") & (d.week_idx <= t)].reset_index(drop=True)
    pd.testing.assert_frame_equal(keep(a), keep(b))
    assert not keep(a).empty
    changed = lambda d: d[(d.view == "as_of") & (d.week_idx == ORIGIN)]["exp_hours_52"].to_numpy()
    assert not np.allclose(changed(a), changed(b)), "the check has teeth: later weeks do change later opinions"


def test_calibration_rescales_totals_and_paths_together(store):
    rows = store.rows_for(ORIGIN, "test")[:60]
    val = store.rows_for(ORIGIN - 60, "val")
    p = persistence(store, val, ORIGIN - 60)
    y = store.labels(val, ORIGIN - 60)
    fit = pd.DataFrame({t: getattr(p, t) for t in TARGETS} | {LABEL_OF[t]: y[LABEL_OF[t]].astype(float) for t in TARGETS})
    rc = Recalibrator().fit(fit)
    out = ex.example_frames(store, rows, {"R2": persistence}, ORIGIN, [ORIGIN], recal={"R2": rc})
    assert out.loc[out.view != "actual", "calibrated"].all() and not out.loc[out.view == "actual", "calibrated"].any()
    raw = persistence(store, rows, ORIGIN)
    want = rc.maps["exp_hours"].predict(raw.exp_hours)
    path = out[out.view == "path"].groupby("user_id", sort=False)["hours"].sum()
    np.testing.assert_allclose(path.loc[raw.user_ids].to_numpy(), want, rtol=1e-6, atol=1e-9)
    roll = out[out.view == "as_of"].set_index("user_id").loc[raw.user_ids]
    np.testing.assert_allclose(roll["exp_hours_52"].to_numpy(), want, rtol=1e-6, atol=1e-9)
    assert roll["p_reengage_52"].between(0, 1).all()
