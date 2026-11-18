"""Output heads for the positive part of a zero-inflated, heavy-tailed target (hours, spend).

The probability of a positive value comes from a separate binary head. These heads model the
value GIVEN that it is positive, and all share one interface:

    params = head(z)                 # dict of tensors
    head.nll(params, y)              # per-row negative log-likelihood, for rows with y > 0
    head.mean(params)                # E[y | y > 0]
    head.quantiles(params_np, p_pos, qs)   # NumPy: quantiles of the zero-inflated distribution

ZILN     one lognormal. The default (Wang, Liu, Miao 2019). Known weakness: the mean
         exp(mu + sigma^2 / 2) is very sensitive to sigma, so levels and the top tail can be
         biased even when ranking is good.
Mixture  three lognormals. Fits multi-modal amounts (price ladders, ordinary players and
         very large spenders) that one lognormal cannot.
Ordinal  quantile buckets plus a position inside the bucket. Makes no shape assumption.

The plan keeps ZILN as the default and runs the others as ablations on C1.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import ndtr, ndtri
from torch import nn

SIGMA_MIN, SIGMA_MAX = 0.05, 2.5


def _sigma(raw: torch.Tensor) -> torch.Tensor:
    return nn.functional.softplus(raw).clamp(SIGMA_MIN, SIGMA_MAX)


class ZILNHead(nn.Module):
    kind = "ziln"

    def __init__(self, d: int, center: float = 0.0, **_):
        super().__init__()
        self.out = nn.Linear(d, 2)
        with torch.no_grad():
            self.out.bias[0] = center

    def forward(self, z):
        o = self.out(z)
        return {"mu": o[:, 0], "sigma": _sigma(o[:, 1])}

    def nll(self, p, y):
        ly = torch.log(y)
        return 0.5 * ((ly - p["mu"]) / p["sigma"]) ** 2 + torch.log(p["sigma"])

    def mean(self, p):
        return torch.exp(p["mu"] + 0.5 * p["sigma"] ** 2)

    @staticmethod
    def quantiles(p: dict, p_pos: np.ndarray, qs) -> np.ndarray:
        pp = np.clip(p_pos, 1e-9, 1.0)
        out = np.zeros((len(pp), len(qs)))
        for j, q in enumerate(qs):
            u = (q - (1.0 - pp)) / pp
            pos = u > 0
            out[pos, j] = np.exp(p["mu"][pos] + p["sigma"][pos] * ndtri(np.clip(u[pos], 1e-9, 1 - 1e-9)))
        return out


class MixtureHead(nn.Module):
    kind = "mixture"

    def __init__(self, d: int, center: float = 0.0, n_comp: int = 3, **_):
        super().__init__()
        self.k = n_comp
        self.out = nn.Linear(d, 3 * n_comp)
        with torch.no_grad():
            self.out.bias[n_comp:2 * n_comp] = center + torch.linspace(-1.0, 1.5, n_comp)

    def forward(self, z):
        o = self.out(z)
        k = self.k
        return {"logw": torch.log_softmax(o[:, :k], dim=-1), "mu": o[:, k:2 * k], "sigma": _sigma(o[:, 2 * k:])}

    def nll(self, p, y):
        ly = torch.log(y).unsqueeze(-1)
        comp = -0.5 * ((ly - p["mu"]) / p["sigma"]) ** 2 - torch.log(p["sigma"])
        return -torch.logsumexp(p["logw"] + comp, dim=-1)

    def mean(self, p):
        return (torch.exp(p["logw"]) * torch.exp(p["mu"] + 0.5 * p["sigma"] ** 2)).sum(-1)

    @staticmethod
    def quantiles(p: dict, p_pos: np.ndarray, qs) -> np.ndarray:
        """Bisection on the mixture CDF in log space."""
        w = np.exp(p["logw"])
        pp = np.clip(p_pos, 1e-9, 1.0)
        out = np.zeros((len(pp), len(qs)))
        for j, q in enumerate(qs):
            u = (q - (1.0 - pp)) / pp
            pos = u > 0
            lo = (p["mu"] - 6 * p["sigma"]).min(axis=1)
            hi = (p["mu"] + 6 * p["sigma"]).max(axis=1)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                cdf = (w * ndtr((mid[:, None] - p["mu"]) / p["sigma"])).sum(axis=1)
                lo, hi = np.where(cdf < u, mid, lo), np.where(cdf < u, hi, mid)
            out[pos, j] = np.exp(0.5 * (lo + hi))[pos]
        return out


class OrdinalHead(nn.Module):
    kind = "ordinal"

    def __init__(self, d: int, center: float = 0.0, edges: np.ndarray | None = None, **_):
        super().__init__()
        e = np.asarray(edges if edges is not None else center + np.linspace(-3, 3, 9), dtype=np.float32)
        self.register_buffer("edges", torch.tensor(e))               # log-space bucket edges, [B + 1]
        self.n = len(e) - 1
        self.cls = nn.Linear(d, self.n)
        self.pos = nn.Linear(d, self.n)

    def forward(self, z):
        return {"logp": torch.log_softmax(self.cls(z), dim=-1), "pos": torch.sigmoid(self.pos(z))}

    def _bucket(self, ly):
        b = torch.bucketize(ly, self.edges[1:-1])
        lo, hi = self.edges[b], self.edges[b + 1]
        return b, ((ly - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)

    def nll(self, p, y):
        b, where = self._bucket(torch.log(y).clamp(self.edges[0], self.edges[-1]))
        ce = -p["logp"].gather(1, b.unsqueeze(1)).squeeze(1)
        return ce + 2.0 * (p["pos"].gather(1, b.unsqueeze(1)).squeeze(1) - where) ** 2

    def mean(self, p):
        lo, hi = self.edges[:-1], self.edges[1:]
        return (torch.exp(p["logp"]) * torch.exp(lo + p["pos"] * (hi - lo))).sum(-1)

    def quantiles(self, p: dict, p_pos: np.ndarray, qs) -> np.ndarray:
        e = self.edges.cpu().numpy()
        prob = np.exp(p["logp"])
        cdf = np.concatenate([np.zeros((len(prob), 1)), np.cumsum(prob, axis=1)], axis=1)
        pp = np.clip(p_pos, 1e-9, 1.0)
        out = np.zeros((len(pp), len(qs)))
        for j, q in enumerate(qs):
            u = np.clip((q - (1.0 - pp)) / pp, 0, 1)
            b = np.clip((cdf[:, 1:] < u[:, None]).sum(axis=1), 0, self.n - 1)
            r = np.arange(len(u))
            frac = (u - cdf[r, b]) / np.maximum(prob[r, b], 1e-9)
            val = np.exp(e[b] + np.clip(frac, 0, 1) * (e[b + 1] - e[b]))
            out[:, j] = np.where(u > 0, val, 0.0)
        return out


HEADS = {"ziln": ZILNHead, "mixture": MixtureHead, "ordinal": OrdinalHead}


def quantile_edges(y_pos: np.ndarray, n_buckets: int = 8) -> np.ndarray:
    """Log-space bucket edges at quantiles of the positive training values, for OrdinalHead."""
    ly = np.log(y_pos[y_pos > 0])
    e = np.quantile(ly, np.linspace(0, 1, n_buckets + 1))
    e[0], e[-1] = e[0] - 1e-3, e[-1] + 1e-3
    return np.maximum.accumulate(e + np.arange(len(e)) * 1e-6).astype(np.float32)
