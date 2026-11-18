"""Time-consistency check on forecast revisions. Low priority: run for the finalist only.

A forecast of a FIXED quantity should move like a martingale: on average, new information
revises it by zero. A rolling 52-week forecast is not a fixed quantity (the window moves), so
the check holds the label window fixed at weeks t+1 .. t+52 and compares

    revision = actual(t+1 .. t+k)  +  forecast made at t+k for the rest of the window
               -  forecast made at t for the whole window.

Its mean should be about zero within each segment, where segments are defined from what was
known at t. Never select users by what they did after t: a quiet week is bad news and should
lower the forecast, so "users with an ordinary week" would have a negative mean revision
even under a perfect model.

Two spacings:
    weekly (k = 1)   for models that give weekly P(active)      -> ``weekly_activity_revision``
    13-week (k = 13) for models that give cumulative horizons   -> ``cumulative_revision``
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_activity_revision(weekly_t: np.ndarray, weekly_t1: np.ndarray, active_week_t1: np.ndarray) -> np.ndarray:
    """Revision of expected active weeks in the fixed window, one week on.

    ``weekly_t[:, h-1]`` = P(active in week t+h) forecast at t; ``weekly_t1`` the same at t+1;
    ``active_week_t1`` = 1 if the user was active in week t+1.
    """
    return active_week_t1 + weekly_t1[:, :-1].sum(axis=1) - weekly_t.sum(axis=1)


def cumulative_revision(cum_t: np.ndarray, cum_t13: np.ndarray, actual_first_13: np.ndarray) -> np.ndarray:
    """Revision of a 52-week total, thirteen weeks on, from cumulative horizons (4, 13, 26, 39, 52).

    The forecast made at t+13 for the remaining 39 weeks is its cumulative value at horizon 39.
    """
    return actual_first_13 + cum_t13[:, 3] - cum_t[:, 4]


def revision_summary(revision: np.ndarray, segment: np.ndarray, scale: np.ndarray | None = None) -> pd.DataFrame:
    """Mean revision, its standard error and a z-score, overall and by segment.

    ``scale`` (for example the forecast at t) gives a relative size: mean revision / mean scale.
    |z| above about 3 in a large segment means forecasts drift predictably there, so revisions
    cannot be read as the effect of new events.
    """
    rows = []
    for name, m in [("all", np.ones(len(revision), bool))] + [(str(s), segment == s) for s in np.unique(segment)]:
        r = revision[m]
        if len(r) < 30:
            continue
        se = float(r.std(ddof=1) / np.sqrt(len(r)))
        rows.append({"segment": name, "n": int(len(r)), "mean_revision": float(r.mean()), "std_error": se,
                     "z": float(r.mean() / se) if se > 0 else 0.0,
                     "relative": float(r.mean() / scale[m].mean()) if scale is not None and scale[m].mean() > 0 else np.nan,
                     "revision_sd": float(r.std(ddof=1))})
    return pd.DataFrame(rows)
