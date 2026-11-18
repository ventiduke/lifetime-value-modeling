"""The shared post-hoc layer: the same recalibration and the same intervals for every candidate.

Fairness rule: nothing here is model-specific. Each candidate's validation-origin
predictions (from its 'tune' fit, 'val'-split users) fit the layer; it is then applied to
that candidate's test-origin predictions. Evaluation reports both raw and calibrated.

Recalibration. Isotonic regression of the outcome on the prediction, per target. It fixes
levels (a model that ranks well but is 30% high) without changing the order of users, so
rank metrics are unaffected. Cumulative horizons and weekly activity are rescaled by the
same factor as their 52-week total, which keeps them coherent.

Intervals. Split conformal by segment, on the ratio outcome / (prediction + c): for each
segment the validation quantiles of that ratio give multiplicative bands. Coverage must be
audited by segment (the calibration cohort is a year older than the cohort being scored,
and zero-heavy segments are hard); see ``interval_coverage`` in evaluate.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ltv.models.base import LABEL_OF, TARGETS

PROB = ("p_reengage", "p_spend")


class Recalibrator:
    """Per-target isotonic maps fitted on validation predictions."""

    def __init__(self) -> None:
        self.maps: dict[str, IsotonicRegression] = {}

    def fit(self, df) -> "Recalibrator":
        for t in TARGETS:
            y, p = df[LABEL_OF[t]].to_numpy(float), df[t].to_numpy(float)
            iso = IsotonicRegression(y_min=0.0, y_max=1.0 if t in PROB else None, out_of_bounds="clip")
            self.maps[t] = iso.fit(p, y)
        return self

    def apply(self, df):
        """Return a copy of a prediction frame with calibrated targets and rescaled arrays."""
        out = df.copy()
        for t in TARGETS:
            out[t] = self.maps[t].predict(df[t].to_numpy(float))
        for total, arr in (("exp_hours", "cum_hours"), ("exp_spend", "cum_spend"),
                           ("exp_active_weeks", "weekly_p_active")):
            if arr in df.columns:
                ratio = out[total].to_numpy() / np.maximum(df[total].to_numpy(float), 1e-9)
                cap = 1.0 if arr == "weekly_p_active" else None
                out[arr] = [None if a is None else np.clip(np.asarray(a, float) * r, 0.0, cap)
                            for a, r in zip(df[arr].to_numpy(), ratio)]
        return out


class SegmentConformal:
    """Multiplicative split-conformal intervals for one amount target, by segment."""

    def __init__(self, levels=(0.8, 0.95), offset: float | None = None, min_n: int = 50):
        self.levels, self.offset, self.min_n = tuple(levels), offset, min_n
        self.q: dict = {}

    def fit(self, pred, y, segment) -> "SegmentConformal":
        pred, y, segment = np.asarray(pred, float), np.asarray(y, float), np.asarray(segment)
        if self.offset is None:
            pos = y[y > 0]
            self.offset = float(np.median(pos)) * 0.05 if len(pos) else 1.0     # keeps tiny predictions sane
        ratio = y / (pred + self.offset)
        self.q = {"*": self._quantiles(ratio)}
        for s in np.unique(segment):
            m = segment == s
            if m.sum() >= self.min_n:
                self.q[s] = self._quantiles(ratio[m])
        return self

    def _quantiles(self, r: np.ndarray) -> dict:
        n = len(r)
        out = {}
        for lvl in self.levels:
            lo, hi = (1 - lvl) / 2, 1 - (1 - lvl) / 2
            hi_adj = min(1.0, np.ceil((n + 1) * hi) / max(n, 1))            # finite-sample correction
            out[lvl] = (float(np.quantile(r, lo)), float(np.quantile(r, hi_adj)))
        return out

    def interval(self, pred, segment, level: float):
        pred, segment = np.asarray(pred, float), np.asarray(segment)
        lo, hi = np.empty(len(pred)), np.empty(len(pred))
        for s in np.unique(segment):
            m = segment == s
            a, b = self.q.get(s, self.q["*"])[level]
            lo[m], hi[m] = a * (pred[m] + self.offset), b * (pred[m] + self.offset)
        return lo, hi
