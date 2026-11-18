"""Shared fixtures: a local Spark session and a small synthetic dataset built once per run.

Local tests use classic PySpark in local mode with ANSI SQL on and the UTC time zone, which
is as close to Databricks serverless as a laptop gets. They do NOT exercise Spark Connect,
Delta or Unity Catalog; docs/VERIFICATION.md lists what that leaves unverified.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

N_USERS = int(os.environ.get("LTV_TEST_USERS", "4000"))
N_WEEKS = 320


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession

    wh = tmp_path_factory.mktemp("warehouse")
    s = (SparkSession.builder.master("local[4]").appName("ltv-tests")
         .config("spark.sql.warehouse.dir", str(wh))
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "true")
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.ui.enabled", "false")
         .config("spark.driver.memory", "4g")
         .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={wh}")
         .getOrCreate())
    yield s
    s.stop()


@pytest.fixture(scope="session")
def synth():
    from ltv.synth.generator import generate

    return generate(n_users=N_USERS, n_weeks=N_WEEKS, seed=11, plant=True)


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    from ltv.config import load_config

    art = tmp_path_factory.mktemp("artifacts")
    return load_config(
        ROOT / "conf" / "bakeoff.yml", ROOT / "conf" / "synth.yml",
        overrides={
            "catalog": None, "schema": "ltv_test", "table_format": "parquet",
            "cluster_tables": False, "artifact_path": str(art),
            "source": {"activity_table": "ltv_test_src.weekly_activity",
                       "calendar_table": "ltv_test_src.calendar"},
            "budget": {"profile": "quick", "step_minutes": 2},
            "evaluation": {"bootstrap_draws": 30},
        })


@pytest.fixture(scope="session")
def built(spark, synth, cfg):
    """Source tables written and the whole data layer built once."""
    from ltv import pipeline

    spark.sql("CREATE SCHEMA IF NOT EXISTS ltv_test_src")
    spark.createDataFrame(synth.activity).write.mode("overwrite").format("parquet").saveAsTable(
        cfg["source"]["activity_table"])
    spark.createDataFrame(synth.calendar).write.mode("overwrite").format("parquet").saveAsTable(
        cfg["source"]["calendar_table"])
    info = pipeline.build_all(spark, cfg, with_panel=True)
    return info


@pytest.fixture(scope="session")
def store(spark, cfg, built):
    from ltv.data.sequences import load_store

    return load_store(spark, cfg)
