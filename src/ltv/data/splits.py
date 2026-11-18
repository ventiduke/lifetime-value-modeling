"""Split schedule and the future-calendar rule.

Weeks are integers: ``week_idx`` 0 is the first week of data. An *origin* ``t`` is a scoring
week: features use weeks <= t, labels use weeks t+1 .. t+H (H = horizon_weeks = 52).

Split schedule (docs/DESIGN.md, "Splits"). With ``last`` = the last week index with data:

    test_1      = last - H                 latest origin with a full label window
    test_2      = test_1 - test_gap        a second test origin, in another season phase
    validation  = test_2 - H               tuning and early stopping
    tune_train  = ..., validation - H      every `step` weeks, from first_train_origin
    final_train_k = ..., test_k - H        refit with frozen settings, one per test origin

No training label window ever overlaps the window being scored: the latest training
origin is always exactly H weeks before the origin it is evaluated on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROLES = ("tune_train", "validation", "final_train_1", "final_train_2", "test_1", "test_2")


@dataclass
class Schedule:
    last_week: int
    horizon: int
    test_1: int
    test_2: int
    validation: int
    tune_train: list[int] = field(default_factory=list)
    final_train_1: list[int] = field(default_factory=list)
    final_train_2: list[int] = field(default_factory=list)

    def all_origins(self) -> list[int]:
        s = set(self.tune_train) | set(self.final_train_1) | set(self.final_train_2)
        s |= {self.validation, self.test_1, self.test_2}
        return sorted(s)

    def eval_origins(self) -> list[int]:
        return [self.validation, self.test_2, self.test_1]

    def train_origins(self, fit: str) -> list[int]:
        """``fit`` is 'tune', 'final_1' or 'final_2'."""
        return {"tune": self.tune_train, "final_1": self.final_train_1,
                "final_2": self.final_train_2}[fit]

    def score_origin(self, fit: str) -> int:
        """The origin a fit is evaluated on."""
        return {"tune": self.validation, "final_1": self.test_1, "final_2": self.test_2}[fit]

    def to_frame(self) -> pd.DataFrame:
        rows = [(o, "tune_train") for o in self.tune_train]
        rows += [(o, "final_train_1") for o in self.final_train_1]
        rows += [(o, "final_train_2") for o in self.final_train_2]
        rows += [(self.validation, "validation"), (self.test_1, "test_1"), (self.test_2, "test_2")]
        return pd.DataFrame(rows, columns=["origin_week", "role"]).astype(
            {"origin_week": "int32"})


def _back_from(end: int, first: int, step: int) -> list[int]:
    """Origins end, end-step, ... down to >= first, returned ascending."""
    if end < first:
        return []
    return list(range(end, first - 1, -step))[::-1]


def make_schedule(last_week: int, horizon: int = 52, step: int = 4, first_train_origin: int = 52,
                  test_gap: int = 26) -> Schedule:
    """Compute the schedule from the last week index that has data."""
    test_1 = last_week - horizon
    test_2 = test_1 - test_gap
    validation = test_2 - horizon
    sched = Schedule(
        last_week=last_week, horizon=horizon, test_1=test_1, test_2=test_2,
        validation=validation,
        tune_train=_back_from(validation - horizon, first_train_origin, step),
        final_train_1=_back_from(test_1 - horizon, first_train_origin, step),
        final_train_2=_back_from(test_2 - horizon, first_train_origin, step),
    )
    if not sched.tune_train:
        need = first_train_origin + 3 * horizon + test_gap + 1
        raise ValueError(
            f"not enough history for the split schedule: {last_week + 1} weeks of data, "
            f"but at least {need} are needed (first_train_origin + 3*horizon + test_gap). "
            "Stop at the Phase 0 gate and revisit the design.")
    return sched


def schedule_from_config(last_week: int, cfg) -> Schedule:
    s = cfg["splits"]
    return make_schedule(last_week, int(cfg["horizon_weeks"]), int(s["origin_step_weeks"]),
                         int(s["first_train_origin"]), int(s["test_gap_weeks"]))


# ---------------------------------------------------------------------------
# Future-calendar rule
# ---------------------------------------------------------------------------

def median_season_length(season_idx: np.ndarray, upto_week: int) -> int:
    """Median length of the seasons that had fully ended by ``upto_week`` (inclusive)."""
    s = np.asarray(season_idx)[: upto_week + 1]
    if len(s) == 0:
        return 8
    change = np.flatnonzero(np.diff(s) != 0)          # last week index of each completed season
    if len(change) == 0:
        return max(int(len(s)), 1)
    starts = np.concatenate([[0], change + 1])
    lengths = np.diff(starts)                         # completed seasons only
    if len(lengths) == 0:
        return max(int(change[0] + 1), 1)
    return max(int(round(float(np.median(lengths)))), 1)


def project_calendar(season_week: np.ndarray, season_idx: np.ndarray, origin: int, horizon: int,
                     known_window: str | int = "current_season") -> tuple[np.ndarray, np.ndarray]:
    """Season week for weeks origin+1 .. origin+horizon, using only what is known at origin.

    ``season_week``/``season_idx`` are the calendar's arrays indexed by week_idx (they may
    extend past the data). Inside the known window the real calendar is used. After it,
    seasons repeat at the median length of seasons completed by the origin.

    Returns ``(season_week_projected[horizon], known[horizon])``.
    """
    sw = np.asarray(season_week).astype(int)
    si = np.asarray(season_idx).astype(int)
    n = len(sw)
    if origin >= n:
        raise ValueError(f"origin {origin} is beyond the calendar ({n} weeks)")
    if known_window == "current_season":
        k = 0
        while origin + k + 1 < n and si[origin + k + 1] == si[origin] and k < horizon:
            k += 1
        # The calendar may simply stop mid-season; then only what it holds is known.
    else:
        k = min(int(known_window), horizon, n - 1 - origin)
    med = median_season_length(si, origin)
    out = np.zeros(horizon, dtype=np.int16)
    known = np.zeros(horizon, dtype=bool)
    cur = int(sw[origin])
    for h in range(1, horizon + 1):
        if h <= k:
            cur = int(sw[origin + h])
            known[h - 1] = True
        elif known_window == "current_season" and h == k + 1 and origin + k + 1 < n:
            cur = 1                                    # the current season is known to end here
        else:
            cur = cur + 1 if cur < med else 1
        out[h - 1] = cur
    return out, known


def season_bucket(season_week: np.ndarray, edges=(1, 2, 4)) -> np.ndarray:
    """Coarse calendar bucket used by the HMM transitions: 0 = launch week, 1 = week 2, ..."""
    return np.searchsorted(np.asarray(edges), np.asarray(season_week), side="left").astype(np.int16)


def projected_calendar_frame(season_week: np.ndarray, season_idx: np.ndarray, origins: list[int],
                             horizon: int, known_window: str | int) -> pd.DataFrame:
    """Long table (origin_week, h, season_week_proj, is_launch_proj, known) for SQL joins."""
    rows = []
    for o in origins:
        sw, known = project_calendar(season_week, season_idx, o, horizon, known_window)
        for h in range(horizon):
            rows.append((o, h + 1, int(sw[h]), bool(sw[h] == 1), bool(known[h])))
    return pd.DataFrame(rows, columns=["origin_week", "h", "season_week_proj", "is_launch_proj",
                                       "known"]).astype({"origin_week": "int32", "h": "int32",
                                                         "season_week_proj": "int32"})
