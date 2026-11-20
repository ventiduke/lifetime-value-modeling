"""NumPy scoring kernels: the per-week update of each stateful model, with no PyTorch.

Weekly scoring should never reread a user's whole history. Each stateful model keeps a small
per-user state and advances it by one week:

    HMM   the K-vector of state probabilities (``hmm_step``), with a shortcut for runs of
          weeks that have no row at all (``hmm_quiet_matrix``)
    GRU   the hidden state (``gru_step``)

These functions use NumPy only, so they run inside a Spark pandas UDF, whose sandbox on
Databricks serverless has 1 GiB of memory and may not have PyTorch. tests/test_kernels.py
checks each kernel against the PyTorch model it mirrors.
"""
from __future__ import annotations

import numpy as np

LOG_2PI_HALF = 0.9189385332046727
HMM_LEVEL_SHRINK = 4.0      # active weeks of prior that pull a user's hours level toward the population's


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

def hmm_emission(params: dict, hours: np.ndarray, spend: np.ndarray) -> np.ndarray:
    """Emission likelihood of one week, [N] hours and spend -> [N, K], rescaled per row."""
    p, q = params["p_active"], params["q_spend"]
    act, sp = (hours > 0)[:, None], (spend > 0)[:, None]
    lh = np.log(np.maximum(hours, 1e-6))[:, None]
    ls = np.log(np.maximum(spend, 1e-6))[:, None]
    ln_h = -0.5 * ((lh - params["h_mu"]) / params["h_sigma"]) ** 2 - np.log(params["h_sigma"]) - LOG_2PI_HALF
    ln_s = -0.5 * ((ls - params["s_mu"]) / params["s_sigma"]) ** 2 - np.log(params["s_sigma"]) - LOG_2PI_HALF
    log_em = (np.where(act, np.log(p) + ln_h, np.log1p(-p)) + np.where(sp, np.log(q) + ln_s, np.log1p(-q)))
    return np.exp(log_em - log_em.max(axis=1, keepdims=True))


def hmm_level(level_sum: np.ndarray, level_n: np.ndarray) -> np.ndarray:
    """A user's own hours level (log scale, 0 = the population's) from the two carried numbers:
    the sum of ``log(hours) - h_center`` over their active weeks so far, and the count of those weeks."""
    return level_sum / (level_n + HMM_LEVEL_SHRINK)


def hmm_level_update(params: dict, hours: np.ndarray, level_sum: np.ndarray, level_n: np.ndarray):
    """One week of a model with a per-user hours level (``'h_center' in params``).

    Returns (hours to pass to ``hmm_step``, new level_sum, new level_n). This week's hours are
    divided by the level known BEFORE this week, then the level takes this week in.
    """
    scaled = hours / np.exp(hmm_level(level_sum, level_n))
    act = hours > 0
    dev = np.where(act, np.log(np.maximum(hours, 1e-6)) - float(params["h_center"]), 0.0)
    return scaled, level_sum + dev, level_n + act


def hmm_step(alpha: np.ndarray, params: dict, bucket: int, hours: np.ndarray, spend: np.ndarray) -> np.ndarray:
    """Advance filtered state probabilities by one observed week. ``bucket`` is that week's calendar bucket."""
    new = (alpha @ params["trans"][bucket]) * hmm_emission(params, hours, spend)
    return new / np.maximum(new.sum(axis=1, keepdims=True), 1e-300)


def hmm_start(params: dict, hours: np.ndarray, spend: np.ndarray) -> np.ndarray:
    """State probabilities after a user's first week."""
    new = params["init"][None, :] * hmm_emission(params, hours, spend)
    return new / np.maximum(new.sum(axis=1, keepdims=True), 1e-300)


def hmm_quiet_matrix(params: dict, buckets: np.ndarray) -> np.ndarray:
    """Collapse a run of weeks with no row (no activity, no spend) into one K x K matrix.

    ``alpha_after = normalise(alpha_before @ Q)``. Q depends only on the calendar buckets of
    the run, so it can be cached per (start week, run length) and shared by every user: a
    user who has been gone for a year costs one matrix product, not 52.
    """
    quiet = (1.0 - params["p_active"]) * (1.0 - params["q_spend"])
    k = len(quiet)
    Q = np.eye(k)
    for b in buckets:
        Q = (Q @ params["trans"][int(b)]) * quiet[None, :]
        Q = Q / Q.max()                       # rescale; the caller normalises alpha anyway
    return Q


def hmm_advance_quiet(alpha: np.ndarray, Q: np.ndarray) -> np.ndarray:
    new = alpha @ Q
    return new / np.maximum(new.sum(axis=1, keepdims=True), 1e-300)


# ---------------------------------------------------------------------------
# GRU (PyTorch gate order: reset, update, new)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gru_step(h: np.ndarray, x: np.ndarray, w: dict) -> np.ndarray:
    """One GRU cell step. ``w`` holds weight_ih, weight_hh, bias_ih, bias_hh as exported from torch.nn.GRU."""
    gi = x @ w["weight_ih"].T + w["bias_ih"]
    gh = h @ w["weight_hh"].T + w["bias_hh"]
    hs = h.shape[1]
    r = _sigmoid(gi[:, :hs] + gh[:, :hs])
    z = _sigmoid(gi[:, hs:2 * hs] + gh[:, hs:2 * hs])
    n = np.tanh(gi[:, 2 * hs:] + r * gh[:, 2 * hs:])
    return (1.0 - z) * n + z * h


def gru_weights(gru, layer: int = 0) -> dict:
    """Export one layer of a torch.nn.GRU for ``gru_step``."""
    return {k: getattr(gru, f"{k}_l{layer}").detach().cpu().numpy().astype(np.float64)
            for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")}
