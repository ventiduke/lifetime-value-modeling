"""The prediction contract every candidate writes, and helpers to read training frames.

Every model, whatever its internals, ends by producing a ``Preds`` for a set of users at one
origin. ``Preds.frame`` turns it into rows of the ``predictions`` table (see
docs/DATA_CONTRACT.md). One evaluation notebook then scores all models the same way.

Fits. Each candidate is fitted three times (docs/DESIGN.md, "Splits"):
    'tune'     trains on tune_train origins, scored at the validation origin; all tuning happens here
    'final_1'  refit with frozen settings on final_train_1 origins, scored at test_1
    'final_2'  the same for test_2
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ltv.config import Config
from ltv.data.np_features import FEATURE_NAMES

FITS = ("tune", "final_1", "final_2")
TARGETS = ("p_reengage", "exp_active_weeks", "exp_hours", "p_spend", "exp_spend")
LABEL_OF = {"p_reengage": "any_active_52", "exp_active_weeks": "active_weeks_52",
            "exp_hours": "hours_52", "p_spend": "any_spend_52", "exp_spend": "spend_52"}
QUANTILES = (0.1, 0.5, 0.9)

PRED_SCHEMA = (
    "model_id string, fit string, origin_week int, user_id string, "
    "p_reengage double, exp_active_weeks double, exp_hours double, p_spend double, exp_spend double, "
    "cum_hours array<double>, cum_spend array<double>, weekly_p_active array<float>, "
    "q_hours array<double>, q_spend array<double>, state array<float>"
)


@dataclass
class Preds:
    user_ids: np.ndarray
    origin: int
    p_reengage: np.ndarray
    exp_active_weeks: np.ndarray
    exp_hours: np.ndarray
    p_spend: np.ndarray
    exp_spend: np.ndarray
    cum_hours: np.ndarray | None = None        # [n, 5] cumulative at weeks 4, 13, 26, 39, 52
    cum_spend: np.ndarray | None = None
    weekly_p_active: np.ndarray | None = None  # [n, 52]
    q_hours: np.ndarray | None = None          # [n, 3] quantiles 0.1, 0.5, 0.9 of hours_52
    q_spend: np.ndarray | None = None
    state: np.ndarray | None = None            # [n, K] journey-state probabilities or an embedding

    def __post_init__(self) -> None:
        self.p_reengage = np.clip(np.asarray(self.p_reengage, float), 0.0, 1.0)
        self.p_spend = np.clip(np.asarray(self.p_spend, float), 0.0, 1.0)
        for k in ("exp_active_weeks", "exp_hours", "exp_spend"):
            setattr(self, k, np.maximum(np.asarray(getattr(self, k), float), 0.0))

    def frame(self, model_id: str, fit: str) -> pd.DataFrame:
        n = len(self.user_ids)

        def arr(a, dtype):
            return [None] * n if a is None else [np.asarray(r, dtype).tolist() for r in a]

        return pd.DataFrame({
            "model_id": model_id, "fit": fit, "origin_week": np.int32(self.origin),
            "user_id": np.asarray(self.user_ids, dtype=object),
            "p_reengage": self.p_reengage, "exp_active_weeks": self.exp_active_weeks,
            "exp_hours": self.exp_hours, "p_spend": self.p_spend, "exp_spend": self.exp_spend,
            "cum_hours": arr(self.cum_hours, float), "cum_spend": arr(self.cum_spend, float),
            "weekly_p_active": arr(self.weekly_p_active, np.float32),
            "q_hours": arr(self.q_hours, float), "q_spend": arr(self.q_spend, float),
            "state": arr(self.state, np.float32),
        })


def write_predictions(spark, cfg: Config, frames: list[pd.DataFrame]) -> int:
    """Write prediction frames, replacing any earlier rows for the same model and origin.

    On Delta this is one atomic ``replaceWhere`` overwrite, so rerunning one model touches
    only its own rows. Other formats (not used in Databricks) rewrite the table through a
    temporary copy.
    """
    pdf = pd.concat(frames, ignore_index=True)
    sdf = spark.createDataFrame(pdf, schema=PRED_SCHEMA)
    table = cfg.table("predictions")
    fmt = cfg.get("table_format", "delta")
    keys = pdf[["model_id", "origin_week"]].drop_duplicates()
    pred = " OR ".join(f"(model_id = '{m}' AND origin_week = {int(o)})" for m, o in keys.to_numpy())
    if not spark.catalog.tableExists(table):
        sdf.write.mode("overwrite").format(fmt).saveAsTable(table)
        if fmt == "delta" and cfg.get("cluster_tables"):
            try:
                spark.sql(f"ALTER TABLE {table} CLUSTER BY (model_id, origin_week)")
            except Exception as exc:  # noqa: BLE001  (an optimisation; never fail the run for it)
                print(f"[warn] could not set clustering on {table}: {exc}")
    elif fmt == "delta":
        sdf.write.mode("overwrite").format("delta").option("replaceWhere", pred).saveAsTable(table)
    else:
        tmp = table + "__tmp"
        spark.table(table).where(f"NOT ({pred})").unionByName(sdf).write.mode("overwrite") \
            .format(fmt).saveAsTable(tmp)
        spark.table(tmp).write.mode("overwrite").format(fmt).saveAsTable(table)
        spark.sql(f"DROP TABLE {tmp}")
    return len(pdf)


def load_xy(spark, cfg: Config, origins: list[int], split: str | None = None,
            max_rows: int | None = None) -> pd.DataFrame:
    """Features joined to labels for the given origins, as one pandas frame.

    Columns: user_id, origin_week, split, the features (float32), the label columns, and
    cum_hours / cum_spend / weekly_active as stacked NumPy arrays in ``frame.attrs``.
    """
    f = spark.table(cfg.table("features_tabular")).alias("f")
    lab = spark.table(cfg.table("labels")).alias("l")
    df = f.join(lab, ["user_id", "origin_week"]).where(f"origin_week IN ({', '.join(map(str, origins))})")
    if split:
        df = df.where(f"split = '{split}'")
    if max_rows:
        n = df.count()
        if n > max_rows:
            df = df.sample(fraction=max_rows / n, seed=int(cfg["training"]["seed"]))
    pdf = df.toPandas()
    for c in FEATURE_NAMES:
        pdf[c] = pdf[c].astype(np.float32)
    arrays = {}
    for c, width in (("cum_hours", 5), ("cum_spend", 5), ("weekly_active", 52)):
        arrays[c] = (np.stack(pdf[c].to_numpy()).astype(np.float32) if len(pdf)
                     else np.zeros((0, width), np.float32))
        pdf = pdf.drop(columns=c)
    pdf.attrs.update(arrays)
    return pdf


def increments(cum: np.ndarray) -> np.ndarray:
    """Per-bucket amounts from cumulative totals: [n, 5] -> [n, 5]."""
    return np.diff(cum, axis=1, prepend=0.0)


BUCKET_EDGES = (0, 4, 13, 26, 39, 52)          # weeks (lo, hi] of the five horizon buckets


def bucket_active_weeks(weekly_active: np.ndarray) -> np.ndarray:
    """Active weeks inside each horizon bucket: [n, 52] -> [n, 5]."""
    return np.stack([weekly_active[:, lo:hi].sum(axis=1)
                     for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])], axis=1)
