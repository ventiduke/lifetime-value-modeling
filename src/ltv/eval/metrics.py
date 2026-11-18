"""Metrics. Plain NumPy, each small enough to check by hand (tests/test_metrics.py does).

Primary metric per target (docs/DESIGN.md, "Metrics"):
    p_reengage, p_spend   log loss                     lower is better
    exp_active_weeks      Poisson deviance             lower is better
    exp_hours             Tweedie deviance (p = 1.5)   lower is better
    exp_spend             normalized Gini              higher is better

Level metrics (deviance, bias) answer "is the amount right"; rank metrics (AUC, Gini, top-k
capture) answer "is the order right". A model can be good at one and poor at the other, so
both are always reported. MAE is reported but never used to rank: on a target that is mostly
zero it rewards predicting zero.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12

PRIMARY = {"p_reengage": ("log_loss", "lower"), "p_spend": ("log_loss", "lower"),
           "exp_active_weeks": ("poisson_deviance", "lower"), "exp_hours": ("tweedie_deviance", "lower"),
           "exp_spend": ("normalized_gini", "higher")}


# -- probabilities ----------------------------------------------------------------

def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, score) -> float:
    """Area under the ROC curve, by ranks (ties get the average rank). NaN if one class is missing."""
    y = np.asarray(y) > 0
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = np.asarray(score, float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]                                             # average the ranks of tied scores
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    ends = np.r_[starts[1:], len(ss)]
    avg = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    """Equal-count bins of the prediction: mean predicted and mean actual in each."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="mergesort")
    out = []
    for i, idx in enumerate(np.array_split(order, min(bins, max(len(y), 1)))):
        if len(idx):
            out.append({"bin": i, "n": int(len(idx)), "mean_pred": float(p[idx].mean()),
                        "mean_actual": float(y[idx].mean())})
    return out


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted mean |predicted - actual| over deciles."""
    t = calibration_table(y, p, bins)
    n = sum(r["n"] for r in t)
    return float(sum(r["n"] * abs(r["mean_pred"] - r["mean_actual"]) for r in t) / max(n, 1))


# -- counts and amounts -----------------------------------------------------------

def poisson_deviance(y, mu) -> float:
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), 1e-6)
    t = np.where(y > 0, y * np.log(np.maximum(y, EPS) / mu), 0.0)
    return float(2.0 * np.mean(t - (y - mu)))


def tweedie_deviance(y, mu, power: float = 1.5) -> float:
    """Tweedie deviance for 1 < power < 2: a proper score for the mean of a zero-inflated amount."""
    y, mu = np.asarray(y, float), np.maximum(np.asarray(mu, float), 1e-6)
    a, b = 1.0 - power, 2.0 - power
    return float(2.0 * np.mean(np.power(np.maximum(y, 0), b) / (a * b) - y * np.power(mu, a) / a
                               + np.power(mu, b) / b))


def mae(y, pred) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pred, float))))


def aggregate_bias(y, pred) -> float:
    """Predicted total / actual total. 1.0 is unbiased; 1.2 means 20% over-predicted."""
    s = float(np.sum(y))
    return float(np.sum(pred) / s) if s > 0 else float("nan")


# -- ranking ----------------------------------------------------------------------

def _gini(actual: np.ndarray, order_by: np.ndarray) -> float:
    n = len(actual)
    if n == 0 or actual.sum() <= 0:
        return float("nan")
    order = np.argsort(-order_by, kind="mergesort")
    cum = np.cumsum(actual[order]) / actual.sum()
    return float(cum.sum() / n - (n + 1) / (2.0 * n))


def normalized_gini(y, pred) -> float:
    """Gini of the actual values ordered by the prediction, divided by the best possible (ordered by
    the actual values). 1 = perfect ranking, 0 = random. The standard LTV ranking metric."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    best = _gini(y, y)
    return float(_gini(y, pred) / best) if best and not np.isnan(best) else float("nan")


def top_capture(y, pred, frac: float = 0.01) -> float:
    """Share of the actual total held by the top ``frac`` of users ranked by prediction."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    k = max(int(np.ceil(frac * len(y))), 1)
    top = np.argsort(-pred, kind="mergesort")[:k]
    return float(y[top].sum() / y.sum()) if y.sum() > 0 else float("nan")


def tail_bias(y, pred, frac: float = 0.01) -> float:
    """Predicted total / actual total among the top ``frac`` of users by PREDICTION.
    Decile charts hide the top; this is calibration where the money is."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    k = max(int(np.ceil(frac * len(y))), 1)
    top = np.argsort(-pred, kind="mergesort")[:k]
    return aggregate_bias(y[top], pred[top])


def bias_by_value_band(y, pred) -> list[dict]:
    """Predicted / actual total by band of the ACTUAL value: zero, then quantile bands of the positives.
    Models typically over-predict the low bands and under-predict the top one."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    pos = y > 0
    out = [{"band": "actual = 0", "n": int((~pos).sum()), "mean_pred": float(pred[~pos].mean()) if (~pos).any() else float("nan"),
            "mean_actual": 0.0, "bias": float("nan")}]
    if pos.sum() >= 20:
        q = np.quantile(y[pos], [0.5, 0.9, 0.99])
        edges = [0.0, q[0], q[1], q[2], np.inf]
        names = ["positive, bottom 50%", "positive, 50-90%", "positive, 90-99%", "positive, top 1%"]
        for name, lo, hi in zip(names, edges[:-1], edges[1:]):
            m = pos & (y > lo) & (y <= hi)
            if m.any():
                out.append({"band": name, "n": int(m.sum()), "mean_pred": float(pred[m].mean()),
                            "mean_actual": float(y[m].mean()), "bias": aggregate_bias(y[m], pred[m])})
    return out


# -- distributions ----------------------------------------------------------------

def pinball(y, q_pred, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def crps_from_quantiles(y, q_preds: np.ndarray, qs) -> float:
    """Approximate CRPS as twice the mean pinball loss over the available quantiles."""
    return float(2.0 * np.mean([pinball(y, q_preds[:, j], q) for j, q in enumerate(qs)]))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo, float) - 1e-9) & (y <= np.asarray(hi, float) + 1e-9)))


# -- one call per target ----------------------------------------------------------

def target_metrics(target: str, y, pred) -> dict[str, float]:
    """All metrics for one target on one group of users."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if len(y) == 0:
        return {}
    if target in ("p_reengage", "p_spend"):
        return {"log_loss": log_loss(y, pred), "auc": auc(y, pred), "ece": ece(y, pred),
                "mean_pred": float(pred.mean()), "mean_actual": float(y.mean())}
    out = {"mae": mae(y, pred), "aggregate_bias": aggregate_bias(y, pred),
           "mean_pred": float(pred.mean()), "mean_actual": float(y.mean())}
    if target == "exp_active_weeks":
        out["poisson_deviance"] = poisson_deviance(y, pred)
    else:
        out["tweedie_deviance"] = tweedie_deviance(y, pred)
        out["normalized_gini"] = normalized_gini(y, pred)
        out["gini_positives"] = normalized_gini(y[y > 0], pred[y > 0]) if (y > 0).sum() > 10 else float("nan")
        out["top1_capture"] = top_capture(y, pred, 0.01)
        out["top5_capture"] = top_capture(y, pred, 0.05)
        out["top1_bias"] = tail_bias(y, pred, 0.01)
        out["top01_bias"] = tail_bias(y, pred, 0.001)
    return out


def primary_value(target: str, y, pred) -> float:
    name, _ = PRIMARY[target]
    return {"log_loss": log_loss, "poisson_deviance": poisson_deviance, "tweedie_deviance": tweedie_deviance,
            "normalized_gini": normalized_gini}[name](y, pred)
