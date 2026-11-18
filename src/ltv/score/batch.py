"""Scoring paths, and the timing harness behind notebooks/60_cost.

Weekly scoring must cover every user on serverless CPU with no command over one hour.
Scoring is written in Python. Three ways to run it are provided so their cost can be
MEASURED on the real platform (nothing here has been timed on Databricks):

1. A pandas function mapped over Spark partitions (``mapInPandas``), exchanging Arrow
   batches. Parallel across executors. On serverless each UDF sandbox has 1 GiB of memory,
   so what it loads must be small: tree models as text, or NumPy kernels.
2. Driver-side batches: pull a slice of users to the notebook, score with the full
   library, write back. Simple and always available, but single-machine.
3. SQL, where it is the simplest form (the R1 lookup join).

State. No model should reread a user's whole history each week. The trees need fresh
features (built in SQL from the sparse table); the HMM keeps a K-vector per user and
advances it with ``kernels_np.hmm_step``; the two-resolution GRU keeps its block-level
hidden state. ``hmm_weekly_update`` shows the pattern end to end.
"""
from __future__ import annotations

import time
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from ltv.data.np_features import FEATURE_NAMES

SCORE_SCHEMA = ("user_id string, p_reengage double, exp_active_weeks double, exp_hours double, "
                "p_spend double, exp_spend double")


def make_tree_scorer(state: dict, which: str = "A1") -> Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]:
    """A ``mapInPandas`` function that scores feature rows with the tree suite.

    ``state`` is ``TreeSuite.state()`` (model text, plain Python types), so the function ships
    to executors without needing shared files. The suite is rebuilt once per partition.
    """
    keep = {"A1": ("any_active", "active_weeks", "hours", "any_spend", "log_spend_pos"),
            "A1tw": ("any_active", "active_weeks", "hours", "any_spend", "spend_tw")}.get(which)
    slim = dict(state)
    if keep is not None:
        slim["models"] = {k: v for k, v in state["models"].items() if k in keep}

    def score(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        from ltv.models.trees import TreeSuite
        suite = TreeSuite.from_state(slim)
        for pdf in batches:
            if which == "A2":
                p = suite.predict_a2(pdf, 0)
            else:
                p = suite.predict_a1(pdf, 0, tweedie_spend=(which == "A1tw"))
            yield pd.DataFrame({"user_id": pdf["user_id"].to_numpy(), "p_reengage": p.p_reengage,
                                "exp_active_weeks": p.exp_active_weeks, "exp_hours": p.exp_hours,
                                "p_spend": p.p_spend, "exp_spend": p.exp_spend})

    return score


def score_trees_spark(features_sdf, state: dict, which: str = "A1"):
    """Score a Spark DataFrame of features in parallel. Returns a Spark DataFrame."""
    cols = ["user_id"] + list(FEATURE_NAMES)
    return features_sdf.select(*cols).mapInPandas(make_tree_scorer(state, which), schema=SCORE_SCHEMA)


def score_trees_driver(features_sdf, state: dict, which: str = "A1", n_slices: int = 1) -> pd.DataFrame:
    """Score on the driver, one hashed slice of users at a time (bounded memory)."""
    fn = make_tree_scorer(state, which)
    cols = ["user_id"] + list(FEATURE_NAMES)
    out = []
    for i in range(n_slices):
        sdf = features_sdf if n_slices == 1 else features_sdf.where(
            f"PMOD(XXHASH64(user_id, 'slice'), {int(n_slices)}) = {i}")
        out += list(fn(iter([sdf.select(*cols).toPandas()])))
    return pd.concat(out, ignore_index=True)


def make_hmm_updater(params: dict, bucket: int) -> Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]:
    """A ``mapInPandas`` function that advances HMM states by one week. NumPy only.

    Input rows: user_id, alpha (array<double>; null for a user's first week), hours, spend
    (0 for users with no row this week). Output: user_id, alpha.
    """
    def update(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        from ltv.models import kernels_np as kn
        for pdf in batches:
            h, s = pdf["hours"].to_numpy(float), pdf["spend"].to_numpy(float)
            new = pdf["alpha"].isna().to_numpy()
            out = np.zeros((len(pdf), len(params["p_active"])))
            if new.any():
                out[new] = kn.hmm_start(params, h[new], s[new])
            if (~new).any():
                prev = np.stack(pdf.loc[~new, "alpha"].to_numpy())
                out[~new] = kn.hmm_step(prev, params, bucket, h[~new], s[~new])
            yield pd.DataFrame({"user_id": pdf["user_id"].to_numpy(), "alpha": list(out)})

    return update


def hmm_weekly_update(state_sdf, week_activity_sdf, params: dict, bucket: int):
    """Advance every user's HMM state by one week.

    ``state_sdf``: (user_id, alpha) as of last week. ``week_activity_sdf``: (user_id, hours,
    spend) for this week's rows only. Users without a row this week get hours = spend = 0;
    users seen for the first time start from the model's initial distribution.
    """
    from pyspark.sql import functions as F
    j = (state_sdf.select("user_id", "alpha").join(week_activity_sdf.select("user_id", "hours", "spend"),
                                                   "user_id", "full_outer")
         .select("user_id", "alpha", F.coalesce("hours", F.lit(0.0)).alias("hours"),
                 F.coalesce("spend", F.lit(0.0)).alias("spend")))
    return j.mapInPandas(make_hmm_updater(params, bucket), schema="user_id string, alpha array<double>")


def hmm_forecast_frame(alpha: np.ndarray, user_ids: np.ndarray, V: dict) -> pd.DataFrame:
    """The 52-week forecast from state vectors: five matrix products."""
    a = alpha.astype(np.float64)
    return pd.DataFrame({"user_id": user_ids, "p_reengage": 1.0 - a @ V["p_no_active"],
                         "exp_active_weeks": a @ V["exp_active_weeks"], "exp_hours": a @ V["cum_hours"][:, -1],
                         "p_spend": 1.0 - a @ V["p_no_spend"], "exp_spend": a @ V["cum_spend"][:, -1]})


class Timer:
    """Collects (step, users, seconds) rows and extrapolates linearly to larger populations."""

    def __init__(self, targets=(30_000_000, 150_000_000)):
        self.rows: list[dict] = []
        self.targets = targets

    def run(self, step: str, model: str, users: int, fn: Callable[[], object], note: str = ""):
        t0 = time.time()
        out = fn()
        sec = time.time() - t0
        row = {"step": step, "model": model, "users": int(users), "seconds": round(sec, 2),
               "users_per_second": round(users / max(sec, 1e-9), 1), "note": note}
        for n in self.targets:
            row[f"minutes_for_{n // 1_000_000}M"] = round(n / max(users, 1) * sec / 60.0, 1)
        self.rows.append(row)
        return out

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
