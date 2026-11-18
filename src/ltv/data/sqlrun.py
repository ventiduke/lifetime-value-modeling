"""Run the SQL files in sql/ against a config.

Design rules (see docs/DESIGN.md, "Data layer"):

* Every SQL file is a single SELECT. Writing the result is done here, in one place, so
  the same SELECT runs on local Spark in tests and on Databricks serverless.
* Table and column names never appear literally in SQL. Files use ``{{name}}``
  placeholders, filled from the config after validation. Only identifiers and numbers
  from the config are substituted; no user input reaches this path.
* Only Spark Connect-safe features are used: no RDDs, no CACHE TABLE, no temp-view tricks
  that depend on session state.
"""
from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Any

from ltv.config import Config

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def sql_dir() -> Path:
    """Locate sql/: the installed ``ltv_sql`` data package, else the repo checkout."""
    try:
        root = Path(str(resources.files("ltv_sql")))
        if (root / "build").is_dir():
            return root
    except (ModuleNotFoundError, TypeError):
        pass
    root = Path(__file__).resolve().parents[3] / "sql"
    if not root.is_dir():
        raise FileNotFoundError("cannot find the sql/ directory; is the project installed?")
    return root


def base_params(cfg: Config) -> dict[str, Any]:
    """Everything a SQL file may reference: table names, column names and constants."""
    src = cfg["source"]
    buckets = int(cfg["sample"]["hash_buckets"])
    tr, va, _ = cfg["splits"]["user_split_pct"]
    p: dict[str, Any] = {f"t_{name}": fq for name, fq in cfg.tables().items()}
    p.update(
        activity_table=src["activity_table"],
        calendar_table=src["calendar_table"],
        horizon=int(cfg["horizon_weeks"]),
        lookback=int(cfg["population_lookback_weeks"]),
        max_weekly_hours=float(cfg["cleaning"]["max_weekly_hours"]),
        hash_buckets=buckets,
        sample_threshold=int(round(float(cfg["sample"]["fraction"]) * buckets)),
        split_train_pct=int(tr),
        split_val_pct=int(tr + va),
    )
    for logical, real in src["activity_columns"].items():
        p[f"act_{logical}"] = real
    for logical, real in src["calendar_columns"].items():
        p[f"cal_{logical}"] = real
    return p


def render(text: str, params: dict[str, Any]) -> str:
    """Fill ``{{name}}`` placeholders. Unknown names and unsafe values are errors."""

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in params:
            raise KeyError(f"SQL placeholder {{{{{key}}}}} has no value")
        val = params[key]
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"
        if isinstance(val, (int, float)):
            return repr(val)
        sval = str(val)
        if key.startswith("sql_"):          # pre-built SQL fragments from this package
            return sval
        if not _SAFE_VALUE.match(sval):
            raise ValueError(f"unsafe value for SQL placeholder {key!r}: {sval!r}")
        return sval

    return _PLACEHOLDER.sub(sub, text)


def load_sql(name: str, params: dict[str, Any]) -> str:
    """``name`` is like ``build/07_features_tabular`` (no extension)."""
    path = sql_dir() / f"{name}.sql"
    return render(path.read_text(encoding="utf-8"), params)


def select(spark, cfg: Config, name: str, **extra):
    """Run one SQL file and return its DataFrame."""
    params = base_params(cfg)
    params.update(extra)
    return spark.sql(load_sql(name, params))


def ensure_schema(spark, cfg: Config) -> None:
    schema = ".".join(p for p in (cfg.get("catalog"), cfg["schema"]) if p)
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")


def write_table(spark, cfg: Config, df, table: str, cluster_by: list[str] | None = None) -> str:
    """Overwrite a pipeline table with ``df`` and return its qualified name."""
    fq = cfg.table(table)
    (df.write.mode("overwrite").option("overwriteSchema", "true")
       .format(cfg.get("table_format", "delta")).saveAsTable(fq))
    if cluster_by and cfg.get("cluster_tables") and cfg.get("table_format", "delta") == "delta":
        # Liquid clustering. Failure here must not fail the build: it is an optimisation.
        try:
            spark.sql(f"ALTER TABLE {fq} CLUSTER BY ({', '.join(cluster_by)})")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not set clustering on {fq}: {exc}")
    return fq


def build(spark, cfg: Config, name: str, table: str, cluster_by: list[str] | None = None,
          **extra) -> str:
    """Run ``sql/<name>.sql`` and write the result to pipeline table ``table``."""
    return write_table(spark, cfg, select(spark, cfg, name, **extra), table, cluster_by)
