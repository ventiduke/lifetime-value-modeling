"""Tabular features computed in NumPy from dense user histories.

This is a second, independent implementation of ``sql/build/07_features_tabular.sql``:

* It is the **test oracle** for the SQL: tests/test_features.py checks that the sparse SQL
  path and this dense path agree on every feature. Change both together.
* It supplies the **static inputs** of the sequence models, which sample origins on the fly
  and therefore cannot join to a prebuilt feature table.
* It lets the **scoring** code build features for any origin without a feature table.

Definitions are in the SQL file's header (active week, gap, lapse, spell).
"""
from __future__ import annotations

import warnings

import numpy as np

from ltv.data.sequences import SequenceStore
from ltv.data.splits import project_calendar

WINDOWS = (4, 13, 26, 52, 104)
LAPSE_THRESHOLDS = (4, 13, 26)

FEATURE_NAMES = (
    ["tenure_weeks", "wk_since_active", "wk_since_spend"]
    + [f"aw_{w}" for w in WINDOWS] + [f"hrs_{w}" for w in WINDOWS]
    + [f"spd_{w}" for w in WINDOWS] + [f"sw_{w}" for w in WINDOWS]
    + ["aw_life", "hrs_life", "spd_life", "sw_life", "max_hrs_52",
       "ratio_aw_4_13", "hrs_slope_26",
       "share_aw_launch", "n_seasons_active", "aw_next13_1y", "aw_next13_2y", "origin_season_week",
       "n_returns_4", "n_returns_13", "n_returns_26", "median_lapse", "max_lapse", "cur_lapse_rel",
       "n_spells", "mean_spell_len", "wk_since_return", "share_returns_launch",
       "platform_id", "country_id", "n_platforms",
       "wk_to_next_launch", "launches_q1", "launches_q2", "launches_q3", "launches_q4"]
)
CATEGORICAL = ("platform_id", "country_id")
F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def calendar_features(store: SequenceStore, origin: int, horizon: int = 52,
                      known_window: str | int = "current_season") -> dict[str, float]:
    """Horizon-context features from the projected calendar (the same for every user)."""
    sw, _ = project_calendar(store.season_week, store.season_idx, origin, horizon, known_window)
    launch = sw == 1
    h = np.arange(1, horizon + 1)
    nxt = h[launch]
    return {
        "wk_to_next_launch": float(nxt[0]) if len(nxt) else np.nan,
        "launches_q1": float((launch & (h <= 13)).sum()),
        "launches_q2": float((launch & (h > 13) & (h <= 26)).sum()),
        "launches_q3": float((launch & (h > 26) & (h <= 39)).sum()),
        "launches_q4": float((launch & (h > 39)).sum()),
    }


def _chunk(store: SequenceStore, rows: np.ndarray, t: int, cal: dict[str, float]) -> np.ndarray:
    n = len(rows)
    H = store.hours[rows, :t + 1].astype(np.float64)
    S = store.spend[rows, :t + 1].astype(np.float64)
    P = store.platform[rows, :t + 1]
    C = store.country[rows, :t + 1]
    act, sp = H > 0, S > 0
    wk = np.arange(t + 1, dtype=np.int32)
    ar = np.arange(n)
    X = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float64)

    def win(a: np.ndarray, w: int) -> np.ndarray:              # sum over weeks (t - w, t]
        return a[:, max(t - w + 1, 0):].sum(axis=1)

    def span(a: np.ndarray, lo: int, hi: int) -> np.ndarray:   # sum over weeks (lo, hi]
        return a[:, max(lo + 1, 0):max(hi + 1, 0)].sum(axis=1)

    last_act = np.where(act, wk, -1).max(axis=1)
    last_sp = np.where(sp, wk, -1).max(axis=1)
    X[:, F["tenure_weeks"]] = t - store.first_week[rows]
    X[:, F["wk_since_active"]] = np.where(last_act >= 0, t - last_act, np.nan)
    X[:, F["wk_since_spend"]] = np.where(last_sp >= 0, t - last_sp, np.nan)
    for w in WINDOWS:
        X[:, F[f"aw_{w}"]] = win(act, w)
        X[:, F[f"hrs_{w}"]] = win(H, w)
        X[:, F[f"spd_{w}"]] = win(S, w)
        X[:, F[f"sw_{w}"]] = win(sp, w)
    aw_life = act.sum(axis=1)
    X[:, F["aw_life"]] = aw_life
    X[:, F["hrs_life"]] = H.sum(axis=1)
    X[:, F["spd_life"]] = S.sum(axis=1)
    X[:, F["sw_life"]] = sp.sum(axis=1)
    X[:, F["max_hrs_52"]] = H[:, max(t - 51, 0):].max(axis=1)

    # trend
    aw_prior13 = span(act, t - 17, t - 4)
    X[:, F["ratio_aw_4_13"]] = (X[:, F["aw_4"]] / 4.0) / ((aw_prior13 + 0.5) / 13.0)
    lo = max(t - 25, 0)
    coef = wk[lo:] - (t - 25) - 12.5
    X[:, F["hrs_slope_26"]] = (H[:, lo:] * coef).sum(axis=1) / 1462.5

    # seasonality
    sw_t = store.season_week[:t + 1]
    launch = sw_t <= 2
    with np.errstate(invalid="ignore", divide="ignore"):
        X[:, F["share_aw_launch"]] = np.where(aw_life > 0, (act & launch).sum(axis=1) / aw_life, np.nan)
    si = store.season_idx[:t + 1]
    starts = np.flatnonzero(np.r_[True, np.diff(si) != 0])
    X[:, F["n_seasons_active"]] = (np.add.reduceat(act.astype(np.int32), starts, axis=1) > 0).sum(axis=1)
    X[:, F["aw_next13_1y"]] = span(act, t - 52, t - 39)
    X[:, F["aw_next13_2y"]] = span(act, t - 104, t - 91)
    X[:, F["origin_season_week"]] = sw_t[t]

    # lapse and return
    idx = np.where(act, wk, -1).astype(np.int32)
    prev_incl = np.maximum.accumulate(idx, axis=1)                         # last active week <= w
    prev = np.concatenate([np.full((n, 1), -1, np.int32), prev_incl[:, :-1]], axis=1)   # < w
    gap = np.where(act & (prev >= 0), wk - prev - 1, -1)                   # -1 = not defined
    spell_start = act & ((prev < 0) | (gap >= 4))
    for thr in LAPSE_THRESHOLDS:
        X[:, F[f"n_returns_{thr}"]] = (gap >= thr).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(np.where(gap >= 4, gap, np.nan), axis=1)
    X[:, F["median_lapse"]] = med
    X[:, F["max_lapse"]] = np.maximum(gap.max(axis=1), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        X[:, F["cur_lapse_rel"]] = np.where(med > 0, X[:, F["wk_since_active"]] / med, np.nan)
    n_spells = spell_start.sum(axis=1)
    X[:, F["n_spells"]] = n_spells
    extra = np.where(act & ~spell_start, wk - prev, 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        X[:, F["mean_spell_len"]] = np.where(n_spells > 0, (extra + n_spells) / n_spells, np.nan)
    last_ret = np.where(spell_start, wk, -1).max(axis=1)
    X[:, F["wk_since_return"]] = np.where(last_ret >= 0, t - last_ret, np.nan)
    n_ret4 = X[:, F["n_returns_4"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        X[:, F["share_returns_launch"]] = np.where(
            n_ret4 > 0, ((gap >= 4) & launch).sum(axis=1) / n_ret4, np.nan)

    # context
    pidx = np.where(P > 0, wk, -1).max(axis=1)
    X[:, F["platform_id"]] = np.where(pidx >= 0, P[ar, np.maximum(pidx, 0)], 0)
    cidx = np.where(C > 0, wk, -1).max(axis=1)
    X[:, F["country_id"]] = np.where(cidx >= 0, C[ar, np.maximum(cidx, 0)], 0)
    nplat = np.zeros(n)
    for p in np.unique(P):
        if p > 0:
            nplat += (P == p).any(axis=1)
    X[:, F["n_platforms"]] = nplat

    for k, v in cal.items():
        X[:, F[k]] = v
    return X


def tabular_features(store: SequenceStore, rows: np.ndarray, origin: int, horizon: int = 52,
                     known_window: str | int = "current_season", chunk: int = 20_000) -> np.ndarray:
    """Feature matrix ``[len(rows), len(FEATURE_NAMES)]`` (float32, NaN = missing) at ``origin``.

    Only weeks <= origin are read, plus the projected calendar.
    """
    rows = np.asarray(rows)
    cal = calendar_features(store, origin, horizon, known_window)
    out = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    for a in range(0, len(rows), chunk):
        out[a:a + chunk] = _chunk(store, rows[a:a + chunk], origin, cal)
    return out


def returner_segment(X: np.ndarray) -> np.ndarray:
    """0, 1 or 2 (= 2+) past returns from a lapse of 26 weeks or more."""
    return np.minimum(np.nan_to_num(X[:, F["n_returns_26"]]), 2).astype(np.int8)
