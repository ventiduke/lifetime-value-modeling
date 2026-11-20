"""Candidate B: hidden Markov models of weekly engagement.

Model. Each user is in one of K latent states each week. A state emits
    active     ~ Bernoulli(p_k)
    hours      ~ lognormal(mu_k, sigma_k)        given active
    spend > 0  ~ Bernoulli(q_k)
    spend      ~ lognormal(m_k, s_k)             given spend > 0
Expected hours and spend use each state's arithmetic mean, tracked alongside (see m_step).
The state moves by a transition matrix that depends on the calendar bucket of the week
being entered (launch week, season week 2, weeks 3-4, later). Transitions depend only on
the calendar, never on the user, so one set of matrices serves everyone.

B1  a plain HMM: any state can follow any state.
B2  a lapse-aware HMM, built for players who lapse for months and come back (spec_b2):
      * dormancy is a CHAIN of states (lapsed recently -> lapsed a while -> lapsed long), so
        the chance of returning depends on how long the lapse has run (a plain HMM is
        memoryless: it forgets how long a user has been away);
      * states are split into user TYPES that never mix (block-diagonal transitions), so the
        filter infers from the whole history whether this is, say, a serial returner;
      * types are crossed with SPEND CLASSES, with tied parameters, because how a player
        engages and how much they spend are different axes;
      * optionally (``user_level``), hours are modeled relative to each user's own level,
        so the types are not spent on telling heavy players from light ones.
    B2 is still an HMM: it is B1 with some transitions forbidden and dormant states that
    cannot emit activity. Fitting, forecasting and scoring are identical.

Fitting. Expectation-maximisation (Baum-Welch), streamed over batches of users: each
iteration is one pass that accumulates sufficient statistics, then closed-form updates.
EM maximises the marginal likelihood, never decreases it, needs no learning rate, and is
deterministic, so a fit stopped at the time box resumes exactly. A fit for origin t reads
weeks <= t only. It needs no labels, so unlike the supervised candidates it can use the
most recent year of data; that is a real property of the model, not leakage.

Forecasting. Filter to the state probabilities alpha at the origin, then push them forward
52 weeks under the PROJECTED calendar. Every expected value is linear in alpha:

    prediction = alpha @ V        V depends only on the origin's calendar

so scoring is a matrix product over a per-user state table (see ``forecast_matrices``).
P(re-engage) is 1 minus the probability of 52 inactive weeks in a row, which is also linear.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ltv.data.sequences import CUM_HORIZONS, SequenceStore
from ltv.data.splits import project_calendar, season_bucket
from ltv.models.base import Preds
from ltv.models.kernels_np import HMM_LEVEL_SHRINK
from ltv.train.timebox import TimeBox, load_pickle, save_pickle

P_INACTIVE = 1e-6                        # p_active of a dormant state: effectively 0
LOG_2PI_HALF = 0.9189385332046727
DT = torch.float64


@dataclass
class HMMSpec:
    name: str
    n_states: int
    trans_mask: np.ndarray               # [K, K] bool, allowed transitions
    dormant: np.ndarray                  # [K] bool, states that cannot emit activity
    labels: list[str] = field(default_factory=list)
    n_buckets: int = 4
    n_types: int = 1                     # user types whose dynamics never mix (B2)
    n_spend_classes: int = 1             # spend classes crossed with the types (B2)
    n_roles: int = 0                     # states per (type, spend class) block (B2)
    user_level: bool = False             # hours are modeled relative to each user's own level


def spec_b1(k: int, n_buckets: int = 4) -> HMMSpec:
    return HMMSpec(f"B1-K{k}", k, np.ones((k, k), bool), np.zeros(k, bool),
                   [f"s{i}" for i in range(k)], n_buckets)


def spec_b2(n_types: int = 4, n_active: int = 2, n_dormant: int = 3, n_spend_classes: int = 3,
            n_buckets: int = 4, user_level: bool = False) -> HMMSpec:
    """Lapse-aware HMM: user types x spend classes, each block a set of active states plus a
    chain of dormant states.

    A user belongs to one (type, spend class) block for life. Engagement behavior and spend
    propensity are different things, so they are separate axes, with parameters TIED:
      * transitions, activity and hours depend on the type and the role, not the spend class;
      * spend probability and amount depend on the spend class and the role, not the type.
    Tying keeps the parameter count small while the filter still learns, per user, both what
    kind of player they are and how much they tend to spend.
    State index = spend_class * (n_types * n_roles) + type * n_roles + role.

    ``user_level`` adds a third axis without adding states: each user's hours are divided by
    their own level before the model sees them (``hours_level``). Players differ a lot, and
    persistently, in how many hours they play. Left to the types, that difference is the
    largest likelihood gain on offer, and EM spends the types on it: in simulation three of
    four types became "heavy", "medium" and "light" players with identical return behavior,
    and the model could no longer tell a serial returner from a churned player.
    """
    s = n_active + n_dormant
    k0 = n_types * s
    mask0 = np.zeros((k0, k0), bool)
    dormant0 = np.zeros(k0, bool)
    for t in range(n_types):
        o = t * s
        for a in range(n_active):
            mask0[o + a, o:o + n_active] = True         # active -> any active state of the type
            mask0[o + a, o + n_active] = True           # active -> start of a lapse
        for d in range(n_dormant):
            i = o + n_active + d
            dormant0[i] = True
            mask0[i, o:o + n_active] = True             # return
            mask0[i, i] = True                          # stay
            if d + 1 < n_dormant:
                mask0[i, i + 1] = True                  # lapse deepens
    c = n_spend_classes
    mask = np.kron(np.eye(c, dtype=bool), mask0)
    roles = [f"active{a}" for a in range(n_active)] + [f"dormant{d}" for d in range(n_dormant)]
    labels = [f"spend{ci}/type{t}/{r}" for ci in range(c) for t in range(n_types) for r in roles]
    return HMMSpec(f"B2-T{n_types}xC{c}" + ("L" if user_level else ""), k0 * c, mask, np.tile(dormant0, c),
                   labels, n_buckets, n_types, c, s, user_level)


def init_params(spec: HMMSpec, seed: int, hours_center: float, spend_center: float) -> dict[str, np.ndarray]:
    """A deliberately asymmetric start: EM cannot separate states that begin identical."""
    rng = np.random.default_rng(seed)
    k, b = spec.n_states, spec.n_buckets
    live = ~spec.dormant
    n_live = int(live.sum())
    n_blocks = spec.n_types * spec.n_spend_classes
    per_type = max(n_live // n_blocks, 1)
    level = np.tile(np.linspace(1.0, -1.0, per_type) if per_type > 1 else np.zeros(1), n_blocks)[:n_live]
    p_act = np.full(k, P_INACTIVE)
    p_act[live] = np.clip(0.6 + 0.3 * level + 0.05 * rng.normal(size=n_live), 0.05, 0.97)
    h_mu = np.full(k, hours_center)
    h_mu[live] = hours_center + 0.7 * level + 0.1 * rng.normal(size=n_live)
    q = np.full(k, 0.002)
    q[live] = np.clip(0.03 * np.exp(level + 0.3 * rng.normal(size=n_live)), 1e-4, 0.5)
    trans = np.zeros((k, k))
    s = spec.n_roles if spec.n_roles else k
    for i in range(k):
        allowed = np.flatnonzero(spec.trans_mask[i])
        t = (i // s) % spec.n_types if spec.n_roles else 0
        if spec.dormant[i]:
            # Types start with different return rates (the thing that distinguishes them), and
            # deeper dormancy starts with a lower one.
            blk = (i // s) * s
            depth = i - (blk + (~spec.dormant[blk:blk + s]).sum())
            ret = min(0.15 * (0.25 ** t) * (0.6 ** depth) * np.exp(0.2 * rng.normal()), 0.5)
            to_active = allowed[~spec.dormant[allowed]]
            deeper = allowed[(spec.dormant[allowed]) & (allowed != i)]
            trans[i, to_active] = ret / max(len(to_active), 1)
            if len(deeper):
                trans[i, deeper] = 0.08
            trans[i, i] = 1.0 - trans[i].sum()
        else:
            others = allowed[allowed != i]
            w = rng.dirichlet(np.ones(len(others))) if len(others) else np.zeros(0)
            leave = 0.12 * (1.0 + 0.5 * t)
            trans[i, others] = leave * w
            trans[i, i] = 1.0 - leave
    init = np.where(live, 1.0, 0.0) + 0.05 * rng.random(k) * live
    s_mu = np.full(k, spend_center) + 0.1 * rng.normal(size=k)
    if spec.n_spend_classes > 1:
        # Spend classes start from "hardly ever spends" up to "spends often and a lot".
        c = spec.n_spend_classes
        cls = np.repeat(np.arange(c), k // c)
        q = np.clip(q * np.geomspace(0.05, 4.0, c)[cls], 1e-6, 0.6)
        s_mu = s_mu + np.linspace(-0.3, 1.0, c)[cls]
        init = init * np.geomspace(1.0, 0.15, c)[cls]
    params = {"trans": np.repeat(trans[None], b, axis=0), "init": init / init.sum(), "p_active": p_act,
              "h_mu": h_mu, "h_sigma": np.full(k, 0.8), "q_spend": q, "s_mu": s_mu, "s_sigma": np.full(k, 1.0),
              "h_mean": np.exp(h_mu + 0.32), "s_mean": np.exp(s_mu + 0.5)}
    if spec.user_level:
        params["h_center"] = np.float64(hours_center)     # fixed for the life of the model
    if spec.n_spend_classes > 1:                      # start exactly tied: copy the first block's dynamics
        k0 = k // spec.n_spend_classes
        blk = np.zeros((b, k, k))
        for ci in range(spec.n_spend_classes):
            blk[:, ci * k0:(ci + 1) * k0, ci * k0:(ci + 1) * k0] = params["trans"][:, :k0, :k0]
        params["trans"] = blk
        for name in ("p_active", "h_mu", "h_mean"):
            params[name] = np.tile(params[name][:k0], spec.n_spend_classes)
    return params


class HMM:
    """Parameters plus the forward and forward-backward passes (probability space, scaled)."""

    def __init__(self, spec: HMMSpec, params: dict[str, np.ndarray]):
        self.spec = spec
        self.params = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}

    def _t(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.params[name])

    def emission(self, hours: torch.Tensor, spend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[N, T] -> (em [N, T, K] rescaled by its row maximum, log of that maximum [N, T])."""
        p, q = self._t("p_active"), self._t("q_spend")
        act = (hours > 0).to(DT).unsqueeze(-1)
        sp = (spend > 0).to(DT).unsqueeze(-1)
        lh = torch.log(hours.to(DT).clamp_min(1e-6)).unsqueeze(-1)
        ls = torch.log(spend.to(DT).clamp_min(1e-6)).unsqueeze(-1)
        hs, ss = self._t("h_sigma"), self._t("s_sigma")
        ln_h = -0.5 * ((lh - self._t("h_mu")) / hs) ** 2 - torch.log(hs) - LOG_2PI_HALF
        ln_s = -0.5 * ((ls - self._t("s_mu")) / ss) ** 2 - torch.log(ss) - LOG_2PI_HALF
        log_em = (act * (torch.log(p) + ln_h) + (1 - act) * torch.log1p(-p)
                  + sp * (torch.log(q) + ln_s) + (1 - sp) * torch.log1p(-q))
        mx = log_em.max(dim=-1, keepdim=True).values
        return torch.exp(log_em - mx), mx.squeeze(-1)

    def forward(self, hours, spend, first, bucket, keep: bool = False):
        """Scaled forward pass over globally aligned weeks 0 .. T-1.

        Weeks before ``first[i]`` are skipped. Returns (loglik [N], alpha_last [N, K]) and,
        with ``keep``, also (alpha [N, T, K], em [N, T, K], c [N, T]) for the backward pass.
        """
        n, t_max = hours.shape
        em, mx = self.emission(hours, spend)
        trans, pi = self._t("trans"), self._t("init")
        k = self.spec.n_states
        alpha = torch.zeros(n, k, dtype=DT)
        ll = torch.zeros(n, dtype=DT)
        A = torch.zeros(n, t_max, k, dtype=DT) if keep else None
        C = torch.ones(n, t_max, dtype=DT) if keep else None
        for t in range(int(first.min()) if n else t_max, t_max):
            starting = (first == t).unsqueeze(-1)
            live = (first <= t).unsqueeze(-1)
            new = torch.where(starting, pi.unsqueeze(0), alpha @ trans[bucket[t]]) * em[:, t]
            c = new.sum(dim=-1, keepdim=True).clamp_min(1e-300)
            alpha = torch.where(live, new / c, alpha)
            ll = ll + torch.where(live.squeeze(-1), torch.log(c.squeeze(-1)) + mx[:, t], torch.zeros_like(ll))
            if keep:
                A[:, t], C[:, t] = alpha, torch.where(live, c, torch.ones_like(c)).squeeze(-1)
        return (ll, alpha, A, em, C) if keep else (ll, alpha)

    def e_step(self, hours, spend, first, bucket) -> dict[str, torch.Tensor]:
        """Sufficient statistics of one batch (forward-backward)."""
        n, t_max = hours.shape
        k, nb = self.spec.n_states, self.spec.n_buckets
        ll, _, A, em, C = self.forward(hours, spend, first, bucket, keep=True)
        trans = self._t("trans")
        act = (hours > 0).to(DT)
        sp = (spend > 0).to(DT)
        lh = torch.log(hours.to(DT).clamp_min(1e-6)) * act
        ls = torch.log(spend.to(DT).clamp_min(1e-6)) * sp
        st = {"xi": torch.zeros(nb, k, k, dtype=DT), "init": torch.zeros(k, dtype=DT),
              "g": torch.zeros(k, dtype=DT), "g_act": torch.zeros(k, dtype=DT),
              "g_lh": torch.zeros(k, dtype=DT), "g_lh2": torch.zeros(k, dtype=DT),
              "g_sp": torch.zeros(k, dtype=DT), "g_ls": torch.zeros(k, dtype=DT),
              "g_ls2": torch.zeros(k, dtype=DT), "g_h": torch.zeros(k, dtype=DT),
              "g_s": torch.zeros(k, dtype=DT), "ll": ll.sum(), "weeks": (t_max - first).sum().to(DT)}
        beta = torch.ones(n, k, dtype=DT)
        for t in range(t_max - 1, int(first.min()) - 1, -1):
            live = (first <= t).to(DT).unsqueeze(-1)
            gamma = A[:, t] * beta * live
            st["g"] += gamma.sum(0)
            st["g_act"] += (gamma * act[:, t:t + 1]).sum(0)
            st["g_lh"] += (gamma * lh[:, t:t + 1]).sum(0)
            st["g_lh2"] += (gamma * (lh[:, t:t + 1] ** 2)).sum(0)
            st["g_sp"] += (gamma * sp[:, t:t + 1]).sum(0)
            st["g_ls"] += (gamma * ls[:, t:t + 1]).sum(0)
            st["g_ls2"] += (gamma * (ls[:, t:t + 1] ** 2)).sum(0)
            st["g_h"] += (gamma * hours[:, t:t + 1].to(DT)).sum(0)
            st["g_s"] += (gamma * spend[:, t:t + 1].to(DT)).sum(0)
            st["init"] += (gamma * (first == t).to(DT).unsqueeze(-1)).sum(0)
            if t > 0:
                w = em[:, t] * beta / C[:, t:t + 1]                      # [N, K], for the move into week t
                moved = (first < t).to(DT).unsqueeze(-1)                 # users who made a transition into t
                st["xi"][bucket[t]] += (A[:, t - 1] * moved).T @ w * trans[bucket[t]]
                beta = torch.where(moved.bool(), w @ trans[bucket[t]].T, beta)
        return st

    def m_step(self, st: dict[str, torch.Tensor]) -> None:
        """Closed-form updates. Small pseudo-counts keep allowed transitions from locking at 0."""
        spec, p = self.spec, self.params
        mask = spec.trans_mask
        st = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in st.items()}
        if spec.n_spend_classes > 1:
            st = self._tie(st)
        st = {k: torch.as_tensor(v) for k, v in st.items()}
        xi = st["xi"].numpy() + 1e-3 * mask[None]
        xi = xi * mask[None]
        p["trans"] = xi / xi.sum(-1, keepdims=True)
        init = st["init"].numpy() * (~spec.dormant) + 1e-6 * (~spec.dormant)
        p["init"] = init / init.sum()
        g, ga, gs = (st[k].numpy() for k in ("g", "g_act", "g_sp"))
        g_a = st["g_actden"].numpy() if "g_actden" in st else g       # tied denominators (B2)
        g_s = st["g_spden"].numpy() if "g_spden" in st else g
        pa = (ga + 0.5) / (g_a + 1.0)
        p["p_active"] = np.where(spec.dormant, P_INACTIVE, np.clip(pa, 1e-4, 1 - 1e-4))
        p["q_spend"] = np.clip((gs + 0.01) / (g_s + 1.0), 1e-6, 1 - 1e-4)
        for mu, sg, s1, s2, n in (("h_mu", "h_sigma", "g_lh", "g_lh2", ga), ("s_mu", "s_sigma", "g_ls", "g_ls2", gs)):
            n_safe = np.maximum(n, 1e-9)
            m = st[s1].numpy() / n_safe
            v = np.maximum(st[s2].numpy() / n_safe - m ** 2, 0.0)
            enough = n > 5.0                                   # leave a state alone until it has data
            p[mu] = np.where(enough, m, p[mu])
            p[sg] = np.where(enough, np.sqrt(v + 0.05 ** 2), p[sg])
        # Arithmetic means, used for expected values. A lognormal fitted on logs gets the
        # MEAN wrong when a state mixes ordinary and very large spenders; the plain weighted
        # mean does not. (The lognormal still describes the shape, for likelihood.)
        p["h_mean"] = np.where(ga > 5.0, st["g_h"].numpy() / np.maximum(ga, 1e-9), p["h_mean"])
        p["s_mean"] = np.where(gs > 5.0, st["g_s"].numpy() / np.maximum(gs, 1e-9), p["s_mean"])

    def _tie(self, st: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Pool sufficient statistics across the tied dimensions of a B2 model (see spec_b2)."""
        spec = self.spec
        c, r, s = spec.n_spend_classes, spec.n_types, spec.n_roles
        k0, nb = r * s, spec.n_buckets
        out = dict(st)
        # Transitions: pooled over spend classes (one set of dynamics per type).
        x = st["xi"].reshape(nb, c, k0, c, k0)
        base = sum(x[:, ci, :, ci, :] for ci in range(c))
        full = np.zeros_like(st["xi"])
        for ci in range(c):
            full[:, ci * k0:(ci + 1) * k0, ci * k0:(ci + 1) * k0] = base
        out["xi"] = full
        # Activity and hours: pooled over spend classes.
        for name in ("g_act", "g_lh", "g_lh2", "g_h"):
            out[name] = np.tile(st[name].reshape(c, k0).sum(0), c)
        out["g_actden"] = np.tile(st["g"].reshape(c, k0).sum(0), c)
        # Spend: pooled over types (one spend profile per spend class and role).
        for name in ("g_sp", "g_ls", "g_ls2", "g_s"):
            out[name] = np.repeat(st[name].reshape(c, r, s).sum(1), r, axis=0).reshape(-1)
        out["g_spden"] = np.repeat(st["g"].reshape(c, r, s).sum(1), r, axis=0).reshape(-1)
        return out

    def export(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def hours_level(hours: np.ndarray, center: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each user's own hours level, week by week, from EARLIER weeks only.

    ``hours`` is [N, T]. The level before week w is the mean of ``log(hours) - center`` over
    the user's active weeks before w, shrunk toward 0 by ``HMM_LEVEL_SHRINK`` pseudo-weeks.
    Returns (level before each week [N, T], sum [N], active weeks [N]); the last two are the
    state after week T-1, which weekly scoring carries forward (``kernels_np.hmm_level``).

    Because week w is scaled by what was known before w, the scaled model is still a proper
    model of the data one week at a time: nothing from the future is used, the likelihood
    stays comparable with an unscaled model's (``filter_users`` adds the change of variable),
    and a carried state reproduces a full-history filter exactly.
    """
    h = hours.astype(np.float64)
    act = h > 0
    dev = np.where(act, np.log(np.maximum(h, 1e-6)) - center, 0.0)
    cs, cn = np.cumsum(dev, axis=1), np.cumsum(act, axis=1)
    after = cs / (cn + HMM_LEVEL_SHRINK)
    before = np.concatenate([np.zeros((len(h), 1)), after[:, :-1]], axis=1) if h.shape[1] else after
    last = (cs[:, -1], cn[:, -1]) if h.shape[1] else (np.zeros(len(h)), np.zeros(len(h), int))
    return before, last[0], last[1]


def level_state(model: "HMM", store: SequenceStore, rows: np.ndarray, upto: int) -> tuple[np.ndarray, np.ndarray]:
    """(level_sum, level_n) after week ``upto``: zeros for a model without a user level."""
    if not model.spec.user_level:
        return np.zeros(len(rows)), np.zeros(len(rows), np.int64)
    _, s, n = hours_level(store.hours[rows, :upto + 1], float(model.params["h_center"]))
    return s, n.astype(np.int64)


def _tensors(model: "HMM", store: SequenceStore, rows: np.ndarray, upto: int):
    """(hours, spend, first week, change-of-variable term per user) for the forward pass."""
    hours = store.hours[rows, :upto + 1]
    jac = np.zeros(len(rows))
    if model.spec.user_level:
        before, _, _ = hours_level(hours, float(model.params["h_center"]))
        jac = (before * (hours > 0)).sum(axis=1)
        hours = hours / np.exp(before)
    return (torch.from_numpy(hours), torch.from_numpy(store.spend[rows, :upto + 1]),
            torch.from_numpy(store.first_week[rows].astype(np.int64)), jac)


def calendar_buckets(store: SequenceStore, edges=(1, 2, 4)) -> np.ndarray:
    return season_bucket(store.season_week, edges)


def _open(shard) -> SequenceStore:
    return shard if isinstance(shard, SequenceStore) else SequenceStore.load(shard)


def fit_hmm(spec: HMMSpec, shards: list, upto: int, box: TimeBox, ckpt: str | Path | None = None,
            split: str | None = "train", edges=(1, 2, 4), batch: int = 4096, max_iters: int = 100,
            tol: float = 1e-5, seed: int = 0, start: dict | None = None) -> tuple[HMM, dict]:
    """EM using weeks <= ``upto``. Time-boxed; resumes exactly from ``ckpt``.

    ``start`` optionally gives the starting parameters (for example the tune fit's, to warm
    start a final fit). Without it the start is drawn from ``seed`` and the first shard.

    ``shards`` is a list of SequenceStore objects or paths to saved shards; one EM iteration
    is one pass over all of them, so memory stays at one shard. Only users in ``split`` are
    used. ``info['complete']`` is true once the log-likelihood has converged or max_iters is
    reached; false means the time box ended first and the call should be repeated.
    """
    saved = load_pickle(ckpt) if ckpt is not None else None
    if saved and saved.get("spec") == spec.name and saved.get("upto") == upto and saved.get("seed") == seed:
        model, hist, done = HMM(spec, saved["params"]), saved["history"], saved["complete"]
    elif start is not None:
        if spec.user_level and "h_center" not in start:
            raise ValueError(f"{spec.name} models hours relative to a user level; the start has no 'h_center'")
        model, hist, done = HMM(spec, start), [], False
    else:
        st0 = _open(shards[0])
        r0 = _rows(st0, upto, split)
        h, s = st0.hours[r0, :upto + 1], st0.spend[r0, :upto + 1]
        model = HMM(spec, init_params(spec, seed, float(np.log(h[h > 0]).mean()) if (h > 0).any() else 1.5,
                                      float(np.log(s[s > 0]).mean()) if (s > 0).any() else 2.5))
        hist, done = [], False
    t0, n_users = time.time(), 0
    with torch.no_grad():
        while not done and len(hist) < max_iters and not box.expired():
            total, stopped, n_users = None, False, 0
            for shard in shards:
                store = _open(shard)
                rows = _rows(store, upto, split)
                n_users += len(rows)
                bucket = torch.from_numpy(calendar_buckets(store, edges)[: upto + 1].astype(np.int64))
                for a in range(0, len(rows), batch):
                    if box.expired():
                        stopped = True
                        break
                    st = model.e_step(*_tensors(model, store, rows[a:a + batch], upto)[:3], bucket)
                    total = st if total is None else {k: total[k] + st[k] for k in st}
                if stopped:
                    break
            if stopped or total is None:                      # a partial pass is discarded and redone
                break
            hist.append(float(total["ll"] / total["weeks"]))  # likelihood of the parameters BEFORE this update
            model.m_step(total)
            done = len(hist) >= 2 and abs(hist[-1] - hist[-2]) < tol * abs(hist[-2])
            if ckpt is not None:
                save_pickle(ckpt, {"spec": spec.name, "upto": upto, "seed": seed, "params": model.export(),
                                   "history": hist, "complete": done})     # converged, not merely stopped
    done = done or len(hist) >= max_iters
    return model, {"spec": spec.name, "iters": len(hist), "complete": bool(done),
                   "train_ll_per_week": hist[-1] if hist else float("nan"), "history": list(hist),
                   "seconds": round(time.time() - t0, 1), "n_users": int(n_users)}


def _rows(store: SequenceStore, upto: int, split: str | None) -> np.ndarray:
    keep = store.first_week <= upto
    if split is not None:
        keep &= store.split == split
    return np.flatnonzero(keep)


@torch.no_grad()
def filter_users(model: HMM, store: SequenceStore, rows: np.ndarray, upto: int, edges=(1, 2, 4),
                 batch: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Filtered state probabilities at week ``upto`` and log-likelihood per user.

    For a model with a user level the likelihood is of the observed hours, not the scaled
    ones (the change of variable is subtracted), so models with and without it compare."""
    bucket = torch.from_numpy(calendar_buckets(store, edges)[: upto + 1].astype(np.int64))
    alphas, lls = [], []
    for a in range(0, len(rows), batch):
        h, s, first, jac = _tensors(model, store, rows[a:a + batch], upto)
        ll, alpha = model.forward(h, s, first, bucket)
        alphas.append(alpha.numpy())
        lls.append(ll.numpy() - jac)
    k = model.spec.n_states
    return (np.concatenate(alphas) if alphas else np.zeros((0, k)),
            np.concatenate(lls) if lls else np.zeros(0))


def loglik_per_week(model: HMM, shards: list, upto: int, split: str | None = "val", edges=(1, 2, 4)) -> float:
    """Mean log-likelihood per user-week for users in ``split``, on weeks <= ``upto``."""
    ll_sum, weeks = 0.0, 0
    for shard in shards:
        store = _open(shard)
        rows = _rows(store, upto, split)
        _, ll = filter_users(model, store, rows, upto, edges)
        ll_sum += float(ll.sum())
        weeks += int((upto + 1 - store.first_week[rows]).sum())
    return ll_sum / max(weeks, 1)


# ---------------------------------------------------------------------------
# Forecasting: everything is linear in the filtered state probabilities
# ---------------------------------------------------------------------------

def forecast_matrices(params: dict[str, np.ndarray], future_bucket: np.ndarray,
                      cum_horizons=CUM_HORIZONS) -> dict[str, np.ndarray]:
    """Matrices V such that each forecast is ``alpha @ V``.

    ``future_bucket[h-1]`` is the calendar bucket of week origin + h under the projected
    calendar. Returns V for: weekly_p_active [K, H], exp_active_weeks [K], cum_hours [K, 5],
    cum_spend [K, 5], p_no_active [K], p_no_spend [K].
    """
    A = params["trans"]
    p, q = params["p_active"], params["q_spend"]
    hrs = p * params["h_mean"]          # expected hours per week in each state
    spd = q * params["s_mean"]
    k, horizon = len(p), len(future_bucket)
    M = np.eye(k)
    quiet_a, quiet_s = np.eye(k), np.eye(k)
    weekly = np.zeros((k, horizon))
    ch, cs = np.zeros(k), np.zeros(k)
    cum_h, cum_s = [], []
    for h in range(1, horizon + 1):
        Ah = A[future_bucket[h - 1]]
        M = M @ Ah
        quiet_a = (quiet_a @ Ah) * (1 - p)[None, :]
        quiet_s = (quiet_s @ Ah) * (1 - q)[None, :]
        weekly[:, h - 1] = M @ p
        ch = ch + M @ hrs
        cs = cs + M @ spd
        if h in cum_horizons:
            cum_h.append(ch.copy())
            cum_s.append(cs.copy())
    return {"weekly_p_active": weekly, "exp_active_weeks": weekly.sum(axis=1),
            "cum_hours": np.stack(cum_h, axis=1), "cum_spend": np.stack(cum_s, axis=1),
            "p_no_active": quiet_a.sum(axis=1), "p_no_spend": quiet_s.sum(axis=1)}


def forecast(alpha: np.ndarray, V: dict[str, np.ndarray], user_ids: np.ndarray, origin: int) -> Preds:
    a = alpha.astype(np.float64)
    cum_h, cum_s = a @ V["cum_hours"], a @ V["cum_spend"]
    return Preds(user_ids, origin, p_reengage=1.0 - a @ V["p_no_active"],
                 exp_active_weeks=a @ V["exp_active_weeks"], exp_hours=cum_h[:, -1],
                 p_spend=1.0 - a @ V["p_no_spend"], exp_spend=cum_s[:, -1],
                 cum_hours=cum_h, cum_spend=cum_s,
                 weekly_p_active=(a @ V["weekly_p_active"]).astype(np.float32),
                 state=alpha.astype(np.float32))


def future_buckets(store: SequenceStore, origin: int, horizon: int = 52,
                   known_window: str | int = "current_season", edges=(1, 2, 4)) -> np.ndarray:
    sw, _ = project_calendar(store.season_week, store.season_idx, origin, horizon, known_window)
    return season_bucket(sw, edges)


def predict_hmm(model: HMM, store: SequenceStore, rows: np.ndarray, origin: int, horizon: int = 52,
                known_window: str | int = "current_season", edges=(1, 2, 4)) -> Preds:
    """Filter each user to the origin, then forecast under the projected calendar."""
    alpha, _ = filter_users(model, store, rows, origin, edges)
    V = forecast_matrices(model.export(), future_buckets(store, origin, horizon, known_window, edges))
    p = forecast(alpha, V, store.user_ids[rows], origin)
    if model.spec.user_level:                      # hours come back on each user's own scale
        s, n = level_state(model, store, rows, origin)
        f = np.exp(s / (n + HMM_LEVEL_SHRINK))
        p.exp_hours, p.cum_hours = p.exp_hours * f, p.cum_hours * f[:, None]
    return p


def state_profile(model: HMM) -> list[dict]:
    """Per-state emission summary, for describing states by what they emit."""
    p = model.params
    return [{"state": lab, "p_active": float(p["p_active"][i]),
             "mean_hours_if_active": float(p["h_mean"][i]), "p_spend": float(p["q_spend"][i]),
             "mean_spend_if_spend": float(p["s_mean"][i]),
             "p_stay_ordinary_week": float(p["trans"][-1, i, i]),
             "p_return_ordinary_week": float(p["trans"][-1, i, ~model.spec.dormant].sum()) if model.spec.dormant[i] else None,
             "p_return_launch_week": float(p["trans"][0, i, ~model.spec.dormant].sum()) if model.spec.dormant[i] else None}
            for i, lab in enumerate(model.spec.labels)]
