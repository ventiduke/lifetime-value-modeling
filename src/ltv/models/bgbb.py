"""R3: discrete-time buy-till-you-die baseline (BG/BB).

The beta-geometric / beta-Bernoulli model of Fader, Hardie and Shang (2010) is the
discrete-time member of the "buy till you die" family: each period a live user transacts
with probability p; at the start of each period a live user dies with probability theta;
death is permanent; p ~ Beta(alpha, beta) and theta ~ Beta(gamma, delta) across users.

Why it is here. It is the classical benchmark for this problem and costs almost nothing to
score (a function of three integers). It is expected to lose, because it cannot represent
a user who lapses and returns or anything about the season calendar. It is also the HMM
with two states, permanent churn and no calendar, so the gap between R3 and candidate B
shows what return-capable states and seasons add.

It is fitted twice: to weekly activity and to weekly spend incidence. Amounts come from
each user's own past mean, shrunk to the population mean.

Scored on its 52-week forecasts, never on P(alive): summing P(alive) badly overstates
returners (see reports/, literature review).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, logsumexp

from ltv.models.base import Preds

CUM = (4, 13, 26, 39, 52)


def _log_terms(params, x, tx, n):
    """log of the 'alive at n' term and of the full likelihood, per (x, tx, n) row."""
    a, b, g, d = params
    x, tx, n = (np.asarray(v, dtype=float) for v in (x, tx, n))
    base = -betaln(a, b) - betaln(g, d)
    alive = betaln(a + x, b + n - x) + betaln(g, d + n) + base
    k = int(np.max(n - tx)) if len(n) else 0
    if k > 0:
        i = np.arange(k)[None, :]
        died = betaln(a + x[:, None], b + tx[:, None] - x[:, None] + i) \
            + betaln(g + 1, d + tx[:, None] + i) + base
        died = np.where(i < (n - tx)[:, None], died, -np.inf)
        ll = logsumexp(np.column_stack([alive, died]), axis=1)
    else:
        ll = alive
    return alive, ll


def loglik(params, x, tx, n, w=None) -> float:
    _, ll = _log_terms(params, x, tx, n)
    return float(np.sum(ll if w is None else ll * w))


def expected_transactions(params, x, tx, n, m: int) -> np.ndarray:
    """E[transactions in periods n+1 .. n+m | x, tx, n]  (Fader, Hardie, Shang 2010, eq. 13)."""
    a, b, g, d = params
    x, tx, n = (np.asarray(v, dtype=float) for v in (x, tx, n))
    _, ll = _log_terms(params, x, tx, n)
    lead = betaln(a + x + 1, b + n - x) - betaln(a, b)
    c = gammaln(g + d) - gammaln(1 + d)
    t1 = np.exp(gammaln(1 + d + n) - gammaln(g + d + n) + c)
    t2 = np.exp(gammaln(1 + d + n + m) - gammaln(g + d + n + m) + c)
    return np.exp(lead - ll) * (d / (g - 1.0)) * (t1 - t2)


def prob_no_transaction(params, x, tx, n, m: int) -> np.ndarray:
    """P(no transaction in periods n+1 .. n+m | x, tx, n).

    The user was either already dead at n, or alive and then either died before transacting
    or survived all m periods without transacting.
    """
    a, b, g, d = params
    x, tx, n = (np.asarray(v, dtype=float) for v in (x, tx, n))
    alive, ll = _log_terms(params, x, tx, n)
    base = -betaln(a, b) - betaln(g, d)
    j = np.arange(m)[None, :]
    die_later = betaln(a + x[:, None], b + (n - x)[:, None] + j) + betaln(g + 1, d + n[:, None] + j) + base
    survive = betaln(a + x, b + n - x + m) + betaln(g, d + n + m) + base
    quiet_if_alive = logsumexp(np.column_stack([die_later, survive]), axis=1)
    dead = np.exp(ll) - np.exp(alive)
    return np.clip((np.maximum(dead, 0.0) + np.exp(quiet_if_alive)) / np.exp(ll), 0.0, 1.0)


def fit_params(x, tx, n, max_rows: int = 30_000, seed: int = 0) -> np.ndarray:
    """Maximum likelihood for (alpha, beta, gamma, delta), on the distinct (x, tx, n) triples.

    Four parameters do not need millions of users: above ``max_rows`` users a uniform sample
    is used. (Sampling the distinct triples by weight instead fails in pandas once one
    triple is common enough, which it is at scale.)
    """
    tri = pd.DataFrame({"x": x, "tx": tx, "n": n})
    if len(tri) > max_rows:
        tri = tri.sample(max_rows, random_state=seed)
    tri = tri.value_counts().reset_index(name="w")
    X, T, N, W = (tri[c].to_numpy(float) for c in ("x", "tx", "n", "w"))

    def nll(theta):
        p = np.exp(theta)
        p[2] = max(p[2], 1e-3)
        return -loglik(p, X, T, N, W) / W.sum()

    best = None
    for start in ([0.0, 0.0, 0.0, 0.0], [-0.5, 1.0, -1.0, 1.0], [0.5, 0.5, -2.0, 0.5]):
        res = minimize(nll, np.array(start), method="L-BFGS-B", bounds=[(-6, 6)] * 4)
        if best is None or res.fun < best.fun:
            best = res
    p = np.exp(best.x)
    if abs(p[2] - 1.0) < 1e-3:              # eq. 13 divides by (gamma - 1)
        p[2] = 1.0 + 1e-3
    return p


def _triples(df: pd.DataFrame, kind: str):
    """(x, tx, n) from tabular features.

    activity: the first week is the acquisition week, so x counts REPEAT active weeks over
              n = tenure opportunities.
    spend   : no acquisition purchase, so all tenure + 1 weeks are opportunities.
    """
    tenure = df["tenure_weeks"].to_numpy(float)
    if kind == "activity":
        n = tenure
        x = np.clip(df["aw_life"].to_numpy(float) - 1.0, 0.0, n)
        tx = np.where(x > 0, n - np.nan_to_num(df["wk_since_active"].to_numpy(float)), 0.0)
    else:
        n = tenure + 1.0
        x = np.minimum(df["sw_life"].to_numpy(float), n)
        tx = np.where(x > 0, n - np.nan_to_num(df["wk_since_spend"].to_numpy(float)), 0.0)
    tx = np.clip(tx, x, n)
    return x, tx, n


@dataclass
class BGBB:
    """R3 as a model with the common fit / predict interface."""

    model_id: str = "R3"
    shrink_hours: float = 4.0
    shrink_spend: float = 2.0

    def fit(self, df: pd.DataFrame) -> "BGBB":
        # One row per user: the latest origin each user appears at. Repeated origins of the
        # same user are not independent, and the latest carries the most information.
        last = df.sort_values("origin_week").groupby("user_id", sort=False).tail(1)
        self.p_act = fit_params(*_triples(last, "activity"))
        self.p_spd = fit_params(*_triples(last, "spend"))
        aw, sw = last["aw_life"].sum(), last["sw_life"].sum()
        self.mean_hours = float(last["hrs_life"].sum() / max(aw, 1.0))
        self.mean_spend = float(last["spd_life"].sum() / max(sw, 1.0))
        return self

    def predict(self, df: pd.DataFrame, origin: int) -> Preds:
        xa, ta, na = _triples(df, "activity")
        xs, ts, ns = _triples(df, "spend")
        cum_aw = np.column_stack([expected_transactions(self.p_act, xa, ta, na, m) for m in range(1, 53)])
        cum_sw = np.column_stack([expected_transactions(self.p_spd, xs, ts, ns, m) for m in CUM])
        weekly = np.diff(cum_aw, axis=1, prepend=0.0)
        hrs_pw = (df["hrs_life"].to_numpy(float) + self.shrink_hours * self.mean_hours) / \
                 (df["aw_life"].to_numpy(float) + self.shrink_hours)
        spd_pw = (df["spd_life"].to_numpy(float) + self.shrink_spend * self.mean_spend) / \
                 (df["sw_life"].to_numpy(float) + self.shrink_spend)
        idx = [m - 1 for m in CUM]
        return Preds(
            df["user_id"].to_numpy(), origin,
            p_reengage=1.0 - prob_no_transaction(self.p_act, xa, ta, na, 52),
            exp_active_weeks=cum_aw[:, -1], exp_hours=cum_aw[:, -1] * hrs_pw,
            p_spend=1.0 - prob_no_transaction(self.p_spd, xs, ts, ns, 52),
            exp_spend=cum_sw[:, -1] * spd_pw,
            cum_hours=cum_aw[:, idx] * hrs_pw[:, None], cum_spend=cum_sw * spd_pw[:, None],
            weekly_p_active=np.clip(weekly, 0.0, 1.0))
