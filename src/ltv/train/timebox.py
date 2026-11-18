"""Time-boxed, resumable training utilities.

The run environment allows no command over one hour. So every long step here:

* takes a time limit and stops cleanly before it,
* checkpoints to a file (a Unity Catalog volume in Databricks) as it goes, and
* resumes from the checkpoint when called again.

A step that runs out of time is not an error: call it again in the next command.
tests/test_training.py checks that a stopped-and-resumed run matches an uninterrupted one.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ltv.io import write_bytes, write_text


class TimeBox:
    """A deadline. ``margin`` seconds are kept back so the caller can save and return."""

    def __init__(self, minutes: float | None, margin: float = 20.0):
        self.start = time.time()
        self.limit = None if minutes is None else minutes * 60.0
        self.margin = margin

    def elapsed(self) -> float:
        return time.time() - self.start

    def left(self) -> float:
        return float("inf") if self.limit is None else self.limit - self.elapsed()

    def expired(self) -> bool:
        return self.left() <= self.margin


def save_pickle(path: str | Path, obj: Any) -> None:
    """Pickle to ``path`` through local disk (safe on Unity Catalog volumes; see ltv.io)."""
    write_bytes(path, pickle.dumps(obj, protocol=4))


def load_pickle(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "rb") as fh:
        return pickle.load(fh)


class TrialLog:
    """Append-only JSON-lines log of tuning trials. Survives restarts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:      # a line cut off by a killed command
                    continue
        return out

    def append(self, rec: dict) -> None:
        # Volumes do not support append writes, so the (small) log is rewritten whole.
        lines = [json.dumps(r, default=float) for r in self.read()] + [json.dumps(rec, default=float)]
        write_text(self.path, "\n".join(lines) + "\n")


def sample_params(space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Draw one configuration. Values: a list (choice), ('log', lo, hi), ('uniform', lo, hi) or ('int', lo, hi)."""
    out = {}
    for k, spec in space.items():
        if isinstance(spec, list):
            out[k] = spec[int(rng.integers(len(spec)))]
        elif spec[0] == "log":
            out[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif spec[0] == "uniform":
            out[k] = float(rng.uniform(spec[1], spec[2]))
        elif spec[0] == "int":
            out[k] = int(rng.integers(spec[1], spec[2] + 1))
        else:
            raise ValueError(f"bad search-space entry for {k}: {spec!r}")
    return out


def random_search(space: dict[str, Any], objective: Callable[[dict], tuple[float, dict]],
                  n_trials: int, log_path: str | Path, box: TimeBox, seed: int = 0,
                  defaults: dict | None = None) -> dict:
    """Seeded random search that resumes from its log and stops at the time box.

    ``objective(params)`` returns ``(loss, info)``; lower is better. Trial ``i`` always draws
    the same configuration for a given seed, so a resumed search continues exactly where it
    stopped. Trial 0 is ``defaults`` if given. Returns the best record so far, with
    ``complete`` = whether all ``n_trials`` have run.
    """
    log = TrialLog(log_path)
    done = {r["trial"]: r for r in log.read()}
    for i in range(n_trials):
        if i in done:
            continue
        if box.expired():
            break
        rng = np.random.default_rng([seed, i])
        params = dict(defaults) if (i == 0 and defaults is not None) else sample_params(space, rng)
        t0 = time.time()
        loss, info = objective(params)
        rec = {"trial": i, "params": params, "loss": float(loss), "seconds": round(time.time() - t0, 1),
               "info": info}
        log.append(rec)
        done[i] = rec
    if not done:
        raise RuntimeError("the time box expired before the first tuning trial finished; "
                           "raise budget.step_minutes or shrink the training sample")
    best = min(done.values(), key=lambda r: r["loss"])
    return {**best, "n_done": len(done), "complete": len(done) >= n_trials}
