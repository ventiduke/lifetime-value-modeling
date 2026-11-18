"""Shared fixtures: a local Spark session and a small synthetic dataset built once per run.

Local tests use classic PySpark in local mode with Delta Lake tables, ANSI SQL on and the
UTC time zone, which is as close to Databricks serverless as a laptop gets. They do NOT
exercise Spark Connect or Unity Catalog; docs/VERIFICATION.md lists what that leaves unverified.
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


def make_spark(warehouse: str, memory: str = "4g"):
    """A local Spark session with Delta Lake, ANSI SQL and UTC, like Databricks serverless."""
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    b = (SparkSession.builder.master("local[4]").appName("ltv-tests")
         .config("spark.sql.warehouse.dir", f"{warehouse}/wh")
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.sql.ansi.enabled", "true")
         .config("spark.sql.session.timeZone", "UTC")
         .config("spark.ui.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.driver.memory", memory)
         .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={warehouse}"))
    s = configure_spark_with_delta_pip(b).getOrCreate()
    s.sparkContext.setLogLevel("ERROR")
    return s


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    s = make_spark(str(tmp_path_factory.mktemp("warehouse")))
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
            "catalog": None, "schema": "ltv_test", "table_format": "delta",
            "cluster_tables": True, "artifact_path": str(art),
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
    spark.createDataFrame(synth.activity).write.mode("overwrite").format("delta").saveAsTable(
        cfg["source"]["activity_table"])
    spark.createDataFrame(synth.calendar).write.mode("overwrite").format("delta").saveAsTable(
        cfg["source"]["calendar_table"])
    info = pipeline.build_all(spark, cfg, with_panel=True)
    return info


@pytest.fixture(scope="session")
def store(spark, cfg, built):
    from ltv.data.sequences import load_store

    return load_store(spark, cfg)
