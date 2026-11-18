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


def write_predictions(spark, cfg: Config, frames: list[pd.DataFrame], upload_rows: int = 50_000) -> int:
    """Write prediction frames, replacing any earlier rows for the same model and origin.

    On Delta this is one atomic ``replaceWhere`` overwrite, so rerunning one model touches
    only its own rows. Other formats (not used in Databricks) rewrite the table through a
    temporary copy.
    """
    pdf = pd.concat(frames, ignore_index=True)
    table = cfg.table("predictions")
    fmt = cfg.get("table_format", "delta")
    keys = pdf[["model_id", "origin_week"]].drop_duplicates()
    pred = " OR ".join(f"(model_id = '{m}' AND origin_week = {int(o)})" for m, o in keys.to_numpy())
    # Upload in pieces: Spark Connect limits the size of local data sent in one message, and
    # each row carries arrays. The first piece replaces the old rows; the rest append.
    for i in range(0, max(len(pdf), 1), upload_rows):
        sdf = spark.createDataFrame(pdf.iloc[i:i + upload_rows], schema=PRED_SCHEMA)
        if i > 0:
            sdf.write.mode("append").format(fmt).saveAsTable(table)
        elif not spark.catalog.tableExists(table):
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

    Columns: user_id, origin_week, split, the features (float32), the five label columns, and
    the label arrays flattened into plain columns: ``cum_hours_4`` .. ``cum_hours_52``,
    ``cum_spend_4`` .. ``cum_spend_52`` and ``wa_1`` .. ``wa_52`` (weekly activity). Plain
    columns stay aligned through any pandas filter or concat; use ``cum_array`` and
    ``weekly_array`` to get them back as matrices.
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
    # Cast features to FLOAT in Spark, before collecting: collecting DOUBLE columns and
    # converting afterwards would hold the frame twice at double width.
    keys = ["user_id", "origin_week", "split"]
    other = [c for c in df.columns if c not in keys and c not in FEATURE_NAMES]
    df = df.select(*keys, *[df[c].cast("float").alias(c) for c in FEATURE_NAMES], *other)
    pdf = df.toPandas()
    for c in FEATURE_NAMES:
        pdf[c] = pdf[c].astype(np.float32)
    return _flatten_label_arrays(pdf)


CUM_COLS = {"hours": [f"cum_hours_{h}" for h in (4, 13, 26, 39, 52)],
            "spend": [f"cum_spend_{h}" for h in (4, 13, 26, 39, 52)]}
WEEKLY_COLS = [f"wa_{h}" for h in range(1, 53)]


def _flatten_label_arrays(pdf: pd.DataFrame) -> pd.DataFrame:
    """Replace the array columns cum_hours, cum_spend and weekly_active with plain columns."""
    parts = [pdf.drop(columns=["cum_hours", "cum_spend", "weekly_active"])]
    for src, cols, dtype in (("cum_hours", CUM_COLS["hours"], np.float32),
                             ("cum_spend", CUM_COLS["spend"], np.float32),
                             ("weekly_active", WEEKLY_COLS, np.uint8)):
        mat = np.stack(pdf[src].to_numpy()).astype(dtype) if len(pdf) else np.zeros((0, len(cols)), dtype)
        parts.append(pd.DataFrame(mat, columns=cols, index=pdf.index))
    return pd.concat(parts, axis=1)


def cum_array(df: pd.DataFrame, target: str) -> np.ndarray:
    """Cumulative labels at weeks 4, 13, 26, 39, 52 as a matrix; ``target`` is 'hours' or 'spend'."""
    return df[CUM_COLS[target]].to_numpy(dtype=np.float64)


def weekly_array(df: pd.DataFrame) -> np.ndarray:
    """Weekly activity labels as an [n, 52] matrix."""
    return df[WEEKLY_COLS].to_numpy(dtype=np.float32)


def increments(cum: np.ndarray) -> np.ndarray:
    """Per-bucket amounts from cumulative totals: [n, 5] -> [n, 5]."""
    return np.diff(cum, axis=1, prepend=0.0)


BUCKET_EDGES = (0, 4, 13, 26, 39, 52)          # weeks (lo, hi] of the five horizon buckets


def bucket_active_weeks(weekly_active: np.ndarray) -> np.ndarray:
    """Active weeks inside each horizon bucket: [n, 52] -> [n, 5]."""
    return np.stack([weekly_active[:, lo:hi].sum(axis=1)
                     for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])], axis=1)


def frame_from_store(store, rows: np.ndarray, origin: int, known_window="current_season",
                     with_labels: bool = True) -> pd.DataFrame:
    """The same frame ``load_xy`` returns, built from a SequenceStore instead of the tables.

    Used where there is no feature table: scoring at an arbitrary origin, and unit tests.
    Features come from the NumPy implementation, which the tests keep equal to the SQL.
    """
    from ltv.data.np_features import tabular_features
    X = tabular_features(store, rows, origin, known_window=known_window)
    df = pd.DataFrame(X, columns=list(FEATURE_NAMES))
    df.insert(0, "split", store.split[rows])
    df.insert(0, "origin_week", np.int32(origin))
    df.insert(0, "user_id", store.user_ids[rows])
    if with_labels:
        y = store.labels(rows, origin)
        for k in ("any_active_52", "active_weeks_52", "hours_52", "any_spend_52", "spend_52"):
            df[k] = y[k].astype(float)
        df = pd.concat([df, pd.DataFrame(y["cum_hours"], columns=CUM_COLS["hours"]),
                        pd.DataFrame(y["cum_spend"], columns=CUM_COLS["spend"]),
                        pd.DataFrame(y["weekly_active"].astype(np.uint8), columns=WEEKLY_COLS)], axis=1)
    return df
