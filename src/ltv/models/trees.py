"""Candidate A: gradient-boosted trees (LightGBM) on the tabular features.

A1  one model per target:
      any_active      binary           -> p_reengage
      active_weeks    Poisson          -> exp_active_weeks
      hours           Tweedie (1.5)    -> exp_hours
      any_spend       binary           -> p_spend
      log_spend_pos   L2 on log spend, spenders only; exp_spend = p_spend * exp(pred) * smearing factor
    A1tw is A1 with exp_spend from one Tweedie model instead of the hurdle.

A2  a distributional and more coherent version:
      for each of the five horizon buckets (weeks 1-4, 5-13, 14-26, 27-39, 40-52):
        active weeks (Poisson), hours (Tweedie), spend (Tweedie)
      52-week values are sums of the buckets, so cumulative horizons never decrease.
      Distribution of hours_52 and spend_52: a zero-inflated lognormal, fitted in its
      decomposed form (see below), giving quantiles.

Zero-inflated lognormal, decomposed. The ZILN negative log-likelihood is
    BCE(p; y > 0)  +  1[y > 0] * lognormal_nll(log y; mu, sigma).
With separate trees per output, p does not interact with (mu, sigma), so p is exactly the
binary model. mu and sigma are fitted in two stages: mu by least squares on log y among
positives (the MLE of mu for any fixed sigma), then log sigma^2 from the squared residuals.
This is the stable form the plan allows as the fallback to a joint custom objective.

Tuning. One shared hyperparameter set is chosen by time-boxed random search on the
validation origin (three representative models per trial). Each model's number of trees
comes from its own early stopping in the 'tune' fit and is then frozen for the final fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import ndtri

from ltv.data.np_features import CATEGORICAL, FEATURE_NAMES
from ltv.models.base import BUCKET_EDGES, Preds, bucket_active_weeks, cum_array, increments, weekly_array

N_BUCKETS = 5
BUCKET_LEN = np.diff(BUCKET_EDGES)

DEFAULT_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0)
SEARCH_SPACE = {
    "learning_rate": ("log", 0.02, 0.15),
    "num_leaves": [31, 63, 127, 255],
    "min_data_in_leaf": [50, 100, 300, 1000],
    "feature_fraction": ("uniform", 0.5, 1.0),
    "lambda_l2": ("log", 0.1, 30.0),
}
_OBJ = {
    "binary": dict(objective="binary", metric="binary_logloss"),
    "poisson": dict(objective="poisson", metric="poisson"),
    "tweedie": dict(objective="tweedie", tweedie_variance_power=1.5, metric="tweedie"),
    "l2": dict(objective="regression", metric="l2"),
}


def _specs() -> dict[str, tuple[str, str]]:
    """model name -> (objective kind, row filter). Labels come from ``_label``."""
    s = {"any_active": ("binary", "all"), "active_weeks": ("poisson", "all"), "hours": ("tweedie", "all"),
         "any_spend": ("binary", "all"), "log_spend_pos": ("l2", "spend"), "spend_tw": ("tweedie", "all"),
         "log_hours_pos": ("l2", "active")}
    for k in range(N_BUCKETS):
        s[f"aw_b{k}"] = ("poisson", "all")
        s[f"hrs_b{k}"] = ("tweedie", "all")
        s[f"spd_b{k}"] = ("tweedie", "all")
    return s


SPECS = _specs()
TUNING_MODELS = ("any_active", "hours", "any_spend")


def _label(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "any_active":
        return df["any_active_52"].to_numpy(float)
    if name == "active_weeks":
        return df["active_weeks_52"].to_numpy(float)
    if name == "hours":
        return df["hours_52"].to_numpy(float)
    if name == "any_spend":
        return df["any_spend_52"].to_numpy(float)
    if name == "spend_tw":
        return df["spend_52"].to_numpy(float)
    if name == "log_spend_pos":
        return np.log(np.maximum(df["spend_52"].to_numpy(float), 1e-9))
    if name == "log_hours_pos":
        return np.log(np.maximum(df["hours_52"].to_numpy(float), 1e-9))
    kind, k = name[:-3], int(name[-1])
    if kind == "aw":
        return bucket_active_weeks(weekly_array(df))[:, k].astype(float)
    src = cum_array(df, "hours" if kind == "hrs" else "spend")
    return np.maximum(increments(src)[:, k].astype(float), 0.0)


def _mask(df: pd.DataFrame, flt: str) -> np.ndarray:
    if flt == "spend":
        return df["spend_52"].to_numpy() > 0
    if flt == "active":
        return df["hours_52"].to_numpy() > 0
    return np.ones(len(df), dtype=bool)


def ziln_quantiles(p_pos: np.ndarray, mu: np.ndarray, sigma: np.ndarray, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
    """Quantiles of a zero-inflated lognormal: mass 1 - p_pos at zero, lognormal(mu, sigma) otherwise."""
    p = np.clip(p_pos, 1e-9, 1.0)
    out = np.zeros((len(p), len(qs)))
    for j, q in enumerate(qs):
        u = (q - (1.0 - p)) / p                       # quantile level within the positive part
        pos = u > 0
        out[pos, j] = np.exp(mu[pos] + sigma[pos] * ndtri(np.clip(u[pos], 1e-9, 1 - 1e-9)))
    return out


@dataclass
class TreeSuite:
    """All LightGBM models behind A1, A1tw and A2, fitted together on one training frame."""

    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    max_rounds: int = 2000
    early_stop: int = 50
    seed: int = 0
    rounds: dict = field(default_factory=dict)        # frozen number of trees per model
    boosters: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)        # smearing factors, validation losses
    complete: bool = False                            # set by fit(): false if a time box ended it early

    # -- fitting ------------------------------------------------------------------
    def _train(self, name: str, tr: pd.DataFrame, va: pd.DataFrame | None, fixed_rounds: int | None,
               y_tr: np.ndarray | None = None, y_va: np.ndarray | None = None):
        kind, flt = SPECS.get(name, ("l2", name.split(":")[1] if ":" in name else "all"))
        m_tr = _mask(tr, flt)
        y = _label(tr, name) if y_tr is None else y_tr
        prm = {**self.params, **_OBJ[kind], "verbose": -1, "seed": self.seed, "num_threads": 0}
        cats = [c for c in CATEGORICAL]
        dtr = lgb.Dataset(tr.loc[m_tr, list(FEATURE_NAMES)], y[m_tr], categorical_feature=cats,
                          free_raw_data=False)
        if fixed_rounds is not None or va is None:
            n = fixed_rounds or self.max_rounds
            return lgb.train(prm, dtr, num_boost_round=n), n, float("nan")
        m_va = _mask(va, flt)
        yv = _label(va, name) if y_va is None else y_va
        dva = lgb.Dataset(va.loc[m_va, list(FEATURE_NAMES)], yv[m_va], reference=dtr)
        b = lgb.train(prm, dtr, num_boost_round=self.max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(self.early_stop, verbose=False)])
        n = max(int(b.best_iteration or self.max_rounds), 10)
        return b, n, float(b.best_score["valid_0"][prm["metric"]])

    def fit(self, tr: pd.DataFrame, va: pd.DataFrame | None = None, names: tuple | None = None,
            box=None, on_model=None) -> dict:
        """Fit every model. With ``va`` the number of trees comes from early stopping; without
        it, ``self.rounds`` (frozen from the tune fit) is used. Returns validation losses.

        Resumable. A model that already has a booster is skipped, so a suite restored from a
        saved ``state()`` continues where it stopped. With ``box`` (a ``TimeBox``) the fit stops
        between models once the box has expired and leaves ``self.complete`` false.
        ``on_model()`` is called after each model, for the caller to save the state.
        """
        losses = self.extras.setdefault("val_losses", {})
        self.complete = False
        for name in (names or tuple(SPECS)):
            if name in self.boosters:
                continue
            if box is not None and box.expired():
                return dict(losses)
            b, n, loss = self._train(name, tr, va, None if va is not None else self.rounds.get(name))
            self.boosters[name], self.rounds[name], losses[name] = b, n, loss
            if on_model is not None:
                on_model()
        if names is None and not self._fit_variances(tr, va, box, on_model):
            return dict(losses)
        self.complete = True
        return dict(losses)

    def _fit_variances(self, tr: pd.DataFrame, va: pd.DataFrame | None, box=None, on_model=None) -> bool:
        """Second stage of the decomposed ZILN: log sigma^2 from squared log residuals."""
        for tgt, flt, base in (("hours", "active", "log_hours_pos"), ("spend", "spend", "log_spend_pos")):
            name = f"logvar_{tgt}:{flt}"
            if tgt == "spend" and "smear_spend" not in self.extras:
                m = _mask(tr, flt)
                # Duan smearing factor for the hurdle mean, from training residuals.
                self.extras["smear_spend"] = float(np.mean(np.exp(_label(tr, base)[m] - self._raw(tr, base)[m])))
            if name in self.boosters:
                continue
            if box is not None and box.expired():
                return False
            r_tr = (_label(tr, base) - self._raw(tr, base)) ** 2
            y_tr = np.log(np.maximum(r_tr, 1e-4))
            y_va = None if va is None else np.log(np.maximum((_label(va, base) - self._raw(va, base)) ** 2, 1e-4))
            b, n, _ = self._train(name, tr, va, None if va is not None else self.rounds.get(name), y_tr, y_va)
            self.boosters[name], self.rounds[name] = b, n
            if on_model is not None:
                on_model()
        return True

    def _raw(self, df: pd.DataFrame, name: str) -> np.ndarray:
        return self.boosters[name].predict(df[list(FEATURE_NAMES)], num_iteration=self.rounds[name])

    # -- prediction ---------------------------------------------------------------
    def predict_a1(self, df: pd.DataFrame, origin: int, tweedie_spend: bool = False) -> Preds:
        p_spend = self._raw(df, "any_spend")
        if tweedie_spend:
            spend = self._raw(df, "spend_tw")
        else:
            spend = p_spend * np.exp(self._raw(df, "log_spend_pos")) * self.extras.get("smear_spend", 1.0)
        return Preds(df["user_id"].to_numpy(), origin, self._raw(df, "any_active"),
                     self._raw(df, "active_weeks"), self._raw(df, "hours"), p_spend, spend)

    def _sigma(self, df: pd.DataFrame, tgt: str, flt: str) -> np.ndarray:
        # E[log r^2] under a normal residual is log sigma^2 - 1.27 (the mean of a log chi-square).
        return np.sqrt(np.exp(self._raw(df, f"logvar_{tgt}:{flt}") + 1.27))

    def predict_a2(self, df: pd.DataFrame, origin: int) -> Preds:
        p_act, p_spd = self._raw(df, "any_active"), self._raw(df, "any_spend")
        aw = np.column_stack([self._raw(df, f"aw_b{k}") for k in range(N_BUCKETS)])
        aw = np.minimum(aw, BUCKET_LEN)                                   # a bucket has only so many weeks
        hrs = np.cumsum(np.column_stack([self._raw(df, f"hrs_b{k}") for k in range(N_BUCKETS)]), axis=1)
        spd = np.cumsum(np.column_stack([self._raw(df, f"spd_b{k}") for k in range(N_BUCKETS)]), axis=1)
        weekly = np.repeat(aw / BUCKET_LEN, BUCKET_LEN, axis=1)
        return Preds(df["user_id"].to_numpy(), origin, p_act, aw.sum(axis=1), hrs[:, -1], p_spd, spd[:, -1],
                     cum_hours=hrs, cum_spend=spd, weekly_p_active=weekly,
                     q_hours=ziln_quantiles(p_act, self._raw(df, "log_hours_pos"), self._sigma(df, "hours", "active")),
                     q_spend=ziln_quantiles(p_spd, self._raw(df, "log_spend_pos"), self._sigma(df, "spend", "spend")))

    # -- persistence --------------------------------------------------------------
    def state(self) -> dict:
        return {"params": self.params, "rounds": self.rounds, "extras": self.extras,
                "max_rounds": self.max_rounds, "early_stop": self.early_stop, "seed": self.seed,
                "models": {k: b.model_to_string(num_iteration=self.rounds[k]) for k, b in self.boosters.items()}}

    @classmethod
    def from_state(cls, st: dict) -> "TreeSuite":
        s = cls(params=st["params"], max_rounds=st["max_rounds"], early_stop=st["early_stop"], seed=st["seed"],
                rounds=dict(st["rounds"]), extras=dict(st["extras"]))
        s.boosters = {k: lgb.Booster(model_str=v) for k, v in st["models"].items()}
        return s

    def importance(self, name: str = "any_active", top: int = 20) -> pd.DataFrame:
        b = self.boosters[name]
        return (pd.DataFrame({"feature": b.feature_name(), "gain": b.feature_importance("gain")})
                .sort_values("gain", ascending=False).head(top).reset_index(drop=True))


def tuning_objective(tr: pd.DataFrame, va: pd.DataFrame, max_rounds: int, early_stop: int, seed: int,
                     base: dict[str, float]):
    """Objective for random search: mean validation loss of three representative models,
    each expressed as a relative change from its loss under the default parameters."""

    def objective(params: dict):
        suite = TreeSuite(params={**DEFAULT_PARAMS, **params}, max_rounds=max_rounds,
                          early_stop=early_stop, seed=seed)
        losses = suite.fit(tr, va, names=TUNING_MODELS)
        for k, v in losses.items():
            base.setdefault(k, v)
        # Relative change against the default-parameter loss. Tweedie's metric can be negative,
        # so divide by the magnitude, not the value.
        score = float(np.mean([(losses[k] - base[k]) / abs(base[k]) for k in TUNING_MODELS]))
        return score, {"losses": losses, "rounds": {k: suite.rounds[k] for k in TUNING_MODELS}}

    return objective
