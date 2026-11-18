"""Split schedule and the future-calendar rule."""
import numpy as np
import pytest

from ltv.data.splits import make_schedule, median_season_length, project_calendar, season_bucket


def test_schedule_at_312_weeks():
    s = make_schedule(last_week=311)                 # 312 weeks of data, 0-based
    assert (s.test_1, s.test_2, s.validation) == (259, 233, 181)
    assert s.tune_train[-1] == s.validation - 52
    assert s.final_train_1[-1] == s.test_1 - 52 and s.final_train_2[-1] == s.test_2 - 52
    assert s.tune_train[0] >= 52 and all(b - a == 4 for a, b in zip(s.tune_train, s.tune_train[1:]))


def test_no_training_label_window_overlaps_the_scored_window():
    s = make_schedule(last_week=340)
    for fit in ("tune", "final_1", "final_2"):
        scored = s.score_origin(fit)
        assert max(s.train_origins(fit)) + s.horizon <= scored
    # every origin has a full label window
    assert max(s.all_origins()) + s.horizon <= s.last_week


def test_too_little_history_is_an_error():
    with pytest.raises(ValueError, match="not enough history"):
        make_schedule(last_week=150)


def _calendar(lengths):
    sw, si = [], []
    for i, n in enumerate(lengths):
        sw += list(range(1, n + 1)); si += [i] * n
    return np.array(sw), np.array(si)


def test_projection_uses_only_what_is_known_at_the_origin():
    sw, si = _calendar([8, 8, 8, 8, 11, 5, 9, 9, 9, 9, 9, 9])
    origin = 34                                       # week 3 of the 11-week season
    proj, known = project_calendar(sw, si, origin, 52, "current_season")
    k = int(known.sum())
    assert k == 8                                     # the rest of the current season is known
    assert (proj[:k] == sw[origin + 1:origin + 1 + k]).all()
    assert proj[k] == 1                               # the next season starts when this one ends
    # After the known window, seasons repeat at the median COMPLETED length (8), not the true lengths.
    assert (proj[k:k + 8] == np.arange(1, 9)).all() and proj[k + 8] == 1
    # Changing the future calendar beyond the known window must not change the projection.
    sw2, si2 = _calendar([8, 8, 8, 8, 11, 13, 4, 9, 9, 9, 9, 9])
    proj2, _ = project_calendar(sw2, si2, origin, 52, "current_season")
    assert (proj == proj2).all()


def test_projection_with_integer_window():
    sw, si = _calendar([8] * 12)
    proj, known = project_calendar(sw, si, 10, 52, 20)
    assert known.sum() == 20 and (proj[:20] == sw[11:31]).all()
    assert (proj == sw[11:63]).all()                  # regular seasons: projection equals truth


def test_median_season_length_ignores_the_unfinished_season():
    _, si = _calendar([8, 10, 9, 30])
    assert median_season_length(si, upto_week=40) == 9


def test_season_bucket():
    assert season_bucket(np.array([1, 2, 3, 4, 5, 9])).tolist() == [0, 1, 2, 2, 3, 3]
