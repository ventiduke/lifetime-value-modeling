"""Segments used for reporting, for the lookup baseline and for group-wise calibration.

Every segment is defined from features known at the origin, never from the outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DORMANCY_LABELS = ("active this week", "dormant 1-4", "dormant 5-26", "dormant 27-104",
                   "dormant over 104 or never active")
TENURE_LABELS = ("tenure under 8", "tenure 8-52", "tenure over 52")
RETURNER_LABELS = ("no past return", "1 past return", "2+ past returns")


def dormancy_bucket(wk_since_active) -> np.ndarray:
    w = np.asarray(wk_since_active, dtype=float)
    out = np.full(len(w), 4, dtype=np.int8)
    out[w <= 104] = 3
    out[w <= 26] = 2
    out[w <= 4] = 1
    out[w == 0] = 0
    return out


def tenure_bucket(tenure_weeks) -> np.ndarray:
    t = np.asarray(tenure_weeks, dtype=float)
    return np.where(t < 8, 0, np.where(t <= 52, 1, 2)).astype(np.int8)


def returner_bucket(n_returns_26) -> np.ndarray:
    """Past returns from a lapse of 26 weeks or more: 0, 1, or 2 (meaning 2+)."""
    return np.minimum(np.nan_to_num(np.asarray(n_returns_26, dtype=float)), 2).astype(np.int8)


def segment_frame(features: pd.DataFrame, top_countries: int = 10) -> pd.DataFrame:
    """One row per input row, one column per segment dimension (integer codes)."""
    seg = pd.DataFrame(index=features.index)
    seg["dormancy"] = dormancy_bucket(features["wk_since_active"])
    seg["tenure"] = tenure_bucket(features["tenure_weeks"])
    seg["prior_spender"] = (features["spd_life"].to_numpy() > 0).astype(np.int8)
    seg["returner"] = returner_bucket(features["n_returns_26"])
    seg["platform"] = features["platform_id"].fillna(0).astype(int)
    c = features["country_id"].fillna(0).astype(int)
    seg["country"] = np.where((c >= 1) & (c <= top_countries), c, 0)       # 0 = the rest
    # The headline cut for lapse-and-return behavior: returner history x current dormancy.
    seg["returner_x_dormancy"] = seg["returner"].astype(int) * 10 + seg["dormancy"].astype(int)
    return seg


def segment_label(dim: str, code: int) -> str:
    if dim == "dormancy":
        return DORMANCY_LABELS[code]
    if dim == "tenure":
        return TENURE_LABELS[code]
    if dim == "prior_spender":
        return "prior spender" if code else "never spent"
    if dim == "returner":
        return RETURNER_LABELS[code]
    if dim == "returner_x_dormancy":
        return f"{RETURNER_LABELS[code // 10]} / {DORMANCY_LABELS[code % 10]}"
    if dim == "country":
        return "other countries" if code == 0 else f"country_id {code}"
    return f"{dim} {code}"
