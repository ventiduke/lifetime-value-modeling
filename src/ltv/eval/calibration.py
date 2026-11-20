"""The shared post-hoc layer: the same recalibration and the same intervals for every candidate.

Fairness rule: nothing here is model-specific. Each candidate's validation-origin
predictions (from its 'tune' fit, 'val'-split users) fit the layer; it is then applied to
that candidate's test-origin predictions. Evaluation reports both raw and calibrated.

Recalibration. Isotonic regression of the outcome on the prediction, per target. It fixes
levels (a model that ranks well but is 30% high) without changing the order of users, so
rank metrics are unaffected. Cumulative horizons and weekly activity are rescaled by the
same factor as their 52-week total, which keeps them coherent.

The map is fitted on binned validation predictions (``MonotoneMap``), not on single users.
Plain isotonic regression ends in steps that can rest on a handful of validation users, and
it clips everything outside the validation range onto those steps. The map is fitted on one
fit of a model (``tune``) and applied to another (``final_*``), so predictions move a
little; a model with lumpy predictions (an HMM gives every user of a type nearly the same
number) can then put thousands of users on an end step. In the 2M-user simulation that
turned 12,700 probabilities into exactly 0 and doubled the HMM's predicted hours
(``reports/01_simulation_study.md``). Bins of at least ``min_bin`` users, smoothed
probabilities and proportional extrapolation remove that failure for every candidate.

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


class MonotoneMap:
    """A non-decreasing map from prediction to outcome, fitted on binned validation data.

    Validation users are sorted by prediction and cut into equal-count bins (tied predictions
    stay together). Isotonic regression runs on the bin means, weighted by bin size, and the
    map interpolates linearly between them. No value of the map rests on fewer than
    ``min_bin`` users, unless the validation set itself is small.

    Probabilities: bin rates are smoothed, (positives + 0.5) / (n + 1), so the map never
    returns exactly 0 or 1, and predictions outside the validation range take the end value.
    Amounts: above the validation range the top bin's ratio (outcome / prediction) is carried
    on, so a larger prediction still gives a larger calibrated value. Below the range the
    bottom bin's value is held: a calibrated amount near zero for a user who then plays is
    what makes a deviance explode.

    50 bins is a compromise measured on simulated data: 200 bins left the end bins noisy
    enough to move a lumpy model's level by 10 to 20%; 10 to 20 bins were too coarse to
    correct a badly shaped baseline, or a skewed probability such as P(spend).
    """

    def __init__(self, prob: bool, max_bins: int = 50, min_bin: int = 200):
        self.prob, self.max_bins, self.min_bin = prob, max_bins, min_bin

    def fit(self, p: np.ndarray, y: np.ndarray) -> "MonotoneMap":
        p, y = np.asarray(p, float), np.asarray(y, float)
        order = np.argsort(p, kind="stable")
        p, y = p[order], y[order]
        n = len(p)
        size = max(20, min(self.min_bin, n // 20))            # small validation sets: smaller bins
        k = int(np.clip(n // size, 1, self.max_bins))
        bins = (np.searchsorted(p, p, side="left") * k) // n   # ties share the bin of their first member
        _, start, count = np.unique(bins, return_index=True, return_counts=True)
        x = np.add.reduceat(p, start) / count
        tot = np.add.reduceat(y, start)
        v = (tot + 0.5) / (count + 1.0) if self.prob else tot / count
        iso = IsotonicRegression(y_min=0.0, y_max=1.0 if self.prob else None, out_of_bounds="clip")
        self.x, self.v = x, iso.fit(x, v, sample_weight=count).predict(x)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        out = np.interp(p, self.x, self.v)                     # constant outside the fitted range
        if not self.prob and self.x[-1] > 0:
            out = np.where(p > self.x[-1], self.v[-1] * p / self.x[-1], out)
        return out


class Recalibrator:
    """Per-target monotone maps fitted on validation predictions."""

    def __init__(self) -> None:
        self.maps: dict[str, MonotoneMap] = {}

    def fit(self, df) -> "Recalibrator":
        for t in TARGETS:
            y, p = df[LABEL_OF[t]].to_numpy(float), df[t].to_numpy(float)
            self.maps[t] = MonotoneMap(prob=t in PROB).fit(p, y)
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
