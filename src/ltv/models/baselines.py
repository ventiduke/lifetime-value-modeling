"""Reference baselines R1 and R2. No tuning. Every candidate is reported against them.

R1  segment lookup : the training mean of each target within dormancy x tenure x prior
                     spender x past-returns cells. It is also the default scorer for the
                     long-lapsed tier, where a table lookup is all the budget allows.
R2  persistence    : the next 52 weeks repeat the last 52.

R3 (discrete-time buy-till-you-die) lives in bgbb.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ltv.eval.segments import segment_frame
from ltv.models.base import LABEL_OF, TARGETS, Preds

_KEYS = ["dormancy", "tenure", "prior_spender", "returner"]


class SegmentLookup:
    """R1. Cell means, shrunk toward the dormancy-level mean when a cell is small."""

    model_id = "R1"

    def __init__(self, shrink: float = 20.0):
        self.shrink = shrink
        self.table: dict | None = None

    def fit(self, df: pd.DataFrame) -> "SegmentLookup":
        seg = segment_frame(df)
        Y = np.column_stack([df[LABEL_OF[t]].to_numpy(dtype=float) for t in TARGETS]
                            + [df.attrs["cum_hours"], df.attrs["cum_spend"], df.attrs["weekly_active"]])
        code = self._code(seg)
        dorm = seg["dormancy"].to_numpy()
        self.global_mean = Y.mean(axis=0)
        parent = {d: Y[dorm == d].mean(axis=0) for d in np.unique(dorm)}
        table = {}
        order = np.argsort(code, kind="stable")
        bounds = np.flatnonzero(np.r_[True, np.diff(code[order]) != 0, True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            rows = order[a:b]
            n = len(rows)
            p = parent[dorm[rows[0]]]
            table[int(code[rows[0]])] = (Y[rows].sum(axis=0) + self.shrink * p) / (n + self.shrink)
        self.table, self.parent = table, parent
        return self

    @staticmethod
    def _code(seg: pd.DataFrame) -> np.ndarray:
        return (seg["dormancy"].to_numpy(int) * 1000 + seg["tenure"].to_numpy(int) * 100
                + seg["prior_spender"].to_numpy(int) * 10 + seg["returner"].to_numpy(int))

    def predict(self, df: pd.DataFrame, origin: int) -> Preds:
        seg = segment_frame(df)
        code, dorm = self._code(seg), seg["dormancy"].to_numpy()
        M = np.stack([self.table.get(int(c), self.parent.get(int(d), self.global_mean))
                      for c, d in zip(code, dorm)])
        return Preds(df["user_id"].to_numpy(), origin, M[:, 0], M[:, 1], M[:, 2], M[:, 3], M[:, 4],
                     cum_hours=M[:, 5:10], cum_spend=M[:, 10:15], weekly_p_active=M[:, 15:67])


class Persistence:
    """R2. The last 52 weeks, repeated. Probabilities are 0 or 1, clipped for the log loss."""

    model_id = "R2"

    def fit(self, df: pd.DataFrame) -> "Persistence":
        return self

    def predict(self, df: pd.DataFrame, origin: int, eps: float = 0.02) -> Preds:
        aw = df["aw_52"].to_numpy(float)
        hrs, spd, sw = (df[c].to_numpy(float) for c in ("hrs_52", "spd_52", "sw_52"))
        share = np.array([4, 13, 26, 39, 52]) / 52.0
        return Preds(df["user_id"].to_numpy(), origin,
                     p_reengage=np.clip((aw > 0).astype(float), eps, 1 - eps), exp_active_weeks=aw,
                     exp_hours=hrs, p_spend=np.clip((sw > 0).astype(float), eps, 1 - eps), exp_spend=spd,
                     cum_hours=hrs[:, None] * share, cum_spend=spd[:, None] * share,
                     weekly_p_active=np.repeat((aw / 52.0)[:, None], 52, axis=1))
