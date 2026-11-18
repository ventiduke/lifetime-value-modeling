"""Run tracking: every pipeline step is logged with its parameters, results and wall-clock time.

Two destinations, always both when possible:

* a JSON-lines file, ``<artifact_path>/run_log.jsonl``, which works everywhere;
* MLflow, when it is importable (it is on Databricks). In a Databricks notebook a run is
  attached to that notebook's experiment unless ``mlflow_experiment`` is set in the config.

DBUs cannot be read from inside a run. Each record keeps the Databricks job and run IDs
when they can be found, so cost can be joined later from the billing system table
(``system.billing.usage``).

Tracking must never break a run: every failure here is caught and printed.
"""
from __future__ import annotations

import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

from ltv.config import Config


def _numeric(result: Any, prefix: str = "") -> dict[str, float]:
    """Flatten the numbers (and booleans) in a result dict, one level of nesting deep."""
    out: dict[str, float] = {}
    if not isinstance(result, dict):
        return out
    for k, v in result.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)) and v == v and abs(v) != float("inf"):
            out[key] = float(v)
        elif isinstance(v, dict) and not prefix:
            out.update(_numeric(v, f"{key}."))
    return out


def _context() -> dict[str, str]:
    """Where this ran. Best effort: every lookup may be unavailable."""
    ctx = {"python": platform.python_version(), "machine": platform.machine(), "cpu_count": str(os.cpu_count())}
    for var in ("DATABRICKS_RUNTIME_VERSION", "DB_CLUSTER_ID", "DATABRICKS_JOB_ID", "DATABRICKS_JOB_RUN_ID"):
        if os.environ.get(var):
            ctx[var.lower()] = os.environ[var]
    return ctx


def log_run(cfg: Config, name: str, params: dict, result: Any, seconds: float) -> dict:
    """Record one pipeline step. Returns the record that was written."""
    rec = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "name": name,
           "profile": cfg["budget"]["profile"], "schema": cfg["schema"],
           "params": {k: str(v) for k, v in params.items()}, "metrics": _numeric(result),
           "seconds": round(float(seconds), 1), "context": _context()}
    try:
        from ltv.train.timebox import TrialLog
        TrialLog(cfg.artifact_dir() / "run_log.jsonl").append(rec)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not write run_log.jsonl: {exc}")
    try:
        import mlflow
    except ImportError:
        return rec
    try:
        if cfg.get("mlflow_experiment"):
            mlflow.set_experiment(cfg["mlflow_experiment"])
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"profile": rec["profile"], "schema": rec["schema"], **rec["params"]})
            metrics = {k.replace(" ", "_"): v for k, v in rec["metrics"].items()}
            metrics["wall_clock_seconds"] = rec["seconds"]
            mlflow.log_metrics(metrics)
            mlflow.set_tags(rec["context"])
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] MLflow logging failed for {name}: {exc}")
    return rec


class Tracked:
    """Context manager: time a step and log it on exit.

        with Tracked(cfg, "trees", step="tune") as t:
            t.result = do_the_work()
    """

    def __init__(self, cfg: Config, name: str, **params: Any):
        self.cfg, self.name, self.params, self.result = cfg, name, params, None

    def __enter__(self) -> "Tracked":
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            log_run(self.cfg, self.name, self.params, self.result, time.time() - self.t0)
        return False
