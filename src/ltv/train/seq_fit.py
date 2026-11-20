"""Time-boxed, resumable training for the sequence models.

Training is counted in steps, not epochs. Each step draws one training origin and a batch
of users in the population at that origin, builds their inputs and labels on the fly from
the user histories, and takes one optimiser step. The batch for step s is a function of
(seed, s) only, so a run that stops at the time box and resumes from its checkpoint follows
the same path as an uninterrupted run.

Shards. ``shards`` is a list of SequenceStore objects or saved shard paths. Training works
through them a block of steps at a time, so memory holds one shard.

Tune fit   : evaluates on the validation set every ``eval_every`` steps, keeps the best
             weights, and stops after ``patience`` evaluations without improvement.
Final fits : run exactly the number of steps the tune fit chose (``fixed_steps``).
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ltv.data import seq_inputs as si
from ltv.data.sequences import SequenceStore
from ltv.io import write_via_local
from ltv.models.heads import quantile_edges
from ltv.models.seq import SeqConfig, SeqModel, label_tensors, to_tensors
from ltv.train.timebox import TimeBox


def _open(shard) -> SequenceStore:
    return shard if isinstance(shard, SequenceStore) else SequenceStore.load(shard)


def prepare_config(base: SeqConfig, store: SequenceStore, origins: list[int], known_window="current_season",
                   seed: int = 0, sample: int = 20_000, lookback: int = 104) -> SeqConfig:
    """Fill in the data-dependent parts of a config from TRAINING users only: input scaling,
    head centres, ordinal bucket edges and the high-spender threshold."""
    rng = np.random.default_rng(seed)
    stat, hrs, spd = [], [], []
    for o in rng.choice(origins, size=min(4, len(origins)), replace=False):
        rows = store.rows_for(int(o), "train", lookback)
        rows = rng.choice(rows, size=min(sample // 4, len(rows)), replace=False)
        s, _ = si.static_inputs(store, rows, int(o), known_window)
        y = store.labels(rows, int(o))
        stat.append(s); hrs.append(y["hours_52"]); spd.append(y["spend_52"])
    stat, hrs, spd = np.concatenate(stat), np.concatenate(hrs), np.concatenate(spd)
    std = stat.std(axis=0)
    cfg = replace(base, static_mean=stat.mean(axis=0).tolist(), static_std=np.where(std < 1e-6, 1.0, std).tolist(),
                  hours_center=float(np.log(hrs[hrs > 0]).mean()) if (hrs > 0).any() else 3.0,
                  spend_center=float(np.log(spd[spd > 0]).mean()) if (spd > 0).any() else 3.0,
                  whale_threshold=float(np.quantile(spd, 0.99)) if len(spd) else 0.0)
    if base.head == "ordinal":
        cfg = replace(cfg, hours_edges=quantile_edges(hrs).tolist(), spend_edges=quantile_edges(spd).tolist())
    return cfg


class BatchSampler:
    """Deterministic (seed, step) -> (store, rows, origin)."""

    def __init__(self, shards: list, origins: list[int], batch: int, seed: int, steps_per_shard: int = 200,
                 lookback: int = 104, max_users: int | None = None):
        self.shards, self.origins, self.batch, self.seed = shards, list(origins), batch, seed
        self.steps_per_shard, self.lookback, self.max_users = steps_per_shard, lookback, max_users
        self._idx, self._store, self._pop = -1, None, {}

    def _load(self, i: int) -> None:
        if i != self._idx:
            self._idx, self._store, self._pop = i, _open(self.shards[i]), {}

    def population(self, origin: int) -> np.ndarray:
        if origin not in self._pop:
            rows = self._store.rows_for(origin, "train", self.lookback)
            if self.max_users is not None:                      # learning curve: a fixed subset of users
                rows = rows[rows < self.max_users]
            self._pop[origin] = rows
        return self._pop[origin]

    def get(self, step: int):
        self._load((step // self.steps_per_shard) % len(self.shards))
        rng = np.random.default_rng([self.seed, step])
        for _ in range(20):
            origin = int(self.origins[rng.integers(len(self.origins))])
            pop = self.population(origin)
            if len(pop):
                return self._store, rng.choice(pop, size=min(self.batch, len(pop)), replace=False), origin
        raise RuntimeError("no training users found at any training origin in this shard")


def make_batch(store: SequenceStore, rows: np.ndarray, origin: int, cfg: SeqConfig, known_window):
    inp = to_tensors(si.build_inputs(store, rows, origin, known_window, tokens=cfg.uses_tokens))
    return inp, label_tensors(store.labels(rows, origin, cfg.horizon))


@torch.no_grad()
def evaluate(model: SeqModel, batches: list) -> dict[str, float]:
    """Mean of each head's loss over prebuilt validation batches, plus the scaled total."""
    model.eval()
    tot, n = {}, 0
    for inp, y in batches:
        L = model.losses(model(inp), y)
        k = len(y["hours_52"])
        for name, v in L.items():
            tot[name] = tot.get(name, 0.0) + float(v) * k
        n += k
    out = {k: v / max(n, 1) for k, v in tot.items()}
    out["total"] = float(sum(v / model.cfg.loss_scale.get(k, 1.0) for k, v in out.items()))
    return out


def validation_batches(store: SequenceStore, origin: int, cfg: SeqConfig, known_window, split: str = "val",
                       max_users: int = 20_000, batch: int = 4096, lookback: int = 104) -> list:
    rows = store.rows_for(origin, split, lookback)[:max_users]
    return [make_batch(store, rows[a:a + batch], origin, cfg, known_window) for a in range(0, len(rows), batch)]


def fit_seq(cfg: SeqConfig, shards: list, origins: list[int], box: TimeBox, ckpt: str | Path | None = None,
            val: list | None = None, max_steps: int = 20_000, eval_every: int = 200, patience: int = 5,
            fixed_steps: int | None = None, batch: int = 512, lr: float = 2e-3, seed: int = 0,
            known_window="current_season", lookback: int = 104, max_users: int | None = None,
            threads: int | None = None, lr_drops: int = 0, lr_factor: float = 0.3,
            lr_steps: list[int] | None = None) -> tuple[SeqModel, dict]:
    """Train one sequence model. Returns the model (best weights for a tune fit) and an info dict.

    ``info['complete']`` false means the time box ended first: call again with the same
    arguments to continue.

    Learning rate. Constant by default. With ``lr_drops`` > 0 a tune fit does not stop the
    first time patience runs out: it multiplies the rate by ``lr_factor`` and carries on, up
    to ``lr_drops`` times, and records the steps in ``info['lr_steps']``. A final fit is given
    those steps as ``lr_steps`` and drops the rate at the same points.
    """
    if threads:
        torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    model = SeqModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    st = {"step": 0, "best": None, "best_val": float("inf"), "best_step": 0, "bad": 0, "history": [],
          "done": False, "loss_scale": {}}
    if ckpt is not None and Path(ckpt).exists():
        saved = torch.load(ckpt, weights_only=False)
        if saved["cfg_key"] == _key(cfg, origins, seed, max_users):
            model.cfg.loss_scale = saved["st"]["loss_scale"]
            model.load_state_dict(saved["model"])
            opt.load_state_dict(saved["opt"])
            st = saved["st"]
    model.cfg.loss_scale = st["loss_scale"]
    st.setdefault("lr_steps", [])
    replay = set(int(x) for x in (lr_steps or [])) if fixed_steps is not None else set()

    def drop_lr() -> None:
        for group in opt.param_groups:
            group["lr"] *= lr_factor

    sampler = BatchSampler(shards, origins, batch, seed, lookback=lookback, max_users=max_users)
    target = fixed_steps if fixed_steps is not None else max_steps
    t0 = time.time()

    def save() -> None:
        if ckpt is not None:
            payload = {"cfg_key": _key(cfg, origins, seed, max_users), "model": model.state_dict(),
                       "opt": opt.state_dict(), "st": st}
            write_via_local(ckpt, lambda p: torch.save(payload, p))

    while not st["done"] and st["step"] < target and not box.expired():
        step = st["step"]
        store, rows, origin = sampler.get(step)
        inp, y = make_batch(store, rows, origin, cfg, known_window)
        model.train()
        torch.manual_seed(seed * 1_000_003 + step)                 # dropout masks follow the step too
        L = model.losses(model(inp), y)
        if not st["loss_scale"]:                                   # first step: fix the per-head scales
            st["loss_scale"] = {k: max(float(v.detach()), 1e-3) for k, v in L.items()}
            model.cfg.loss_scale = st["loss_scale"]
        loss = model.total_loss(L)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        st["step"] = step + 1
        if st["step"] in replay:
            drop_lr()
        if fixed_steps is None and val is not None and st["step"] % eval_every == 0:
            v = evaluate(model, val)
            st["history"].append({"step": st["step"], **v})
            if v["total"] < st["best_val"] - 1e-5:
                st["best_val"], st["best_step"], st["bad"] = v["total"], st["step"], 0
                st["best"] = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                st["bad"] += 1
            if st["bad"] >= patience and len(st["lr_steps"]) < lr_drops:
                st["lr_steps"].append(st["step"])
                st["bad"] = 0
                drop_lr()
            st["done"] = st["bad"] >= patience
            save()
    save()
    complete = st["done"] or st["step"] >= target       # early-stopped, or ran all the steps asked for
    if fixed_steps is None and st["best"] is not None:
        model.load_state_dict(st["best"])
    model.eval()
    return model, {"encoder": cfg.encoder, "head": cfg.head, "steps": st["step"], "complete": bool(complete),
                   "best_step": st["best_step"] if fixed_steps is None else st["step"],
                   "best_val": st["best_val"], "history": st["history"], "seconds": round(time.time() - t0, 1),
                   "lr_steps": [x for x in (st["lr_steps"] if fixed_steps is None else sorted(replay))
                                if fixed_steps is None or x <= st["step"]],
                   "lr": float(opt.param_groups[0]["lr"]),
                   "steps_per_second": round((st["step"]) / max(time.time() - t0, 1e-9), 2)}


def _key(cfg: SeqConfig, origins: list[int], seed: int, max_users) -> str:
    return f"{cfg.encoder}|{cfg.head}|{cfg.d}|{cfg.layers}|{cfg.whale_aux}|{cfg.use_tabular}|" \
           f"{min(origins)}-{max(origins)}-{len(origins)}|{seed}|{max_users}"
