# Databricks notebook source
# MAGIC %md
# MAGIC # 60 · Phase 3: speed and cost
# MAGIC
# MAGIC **Requirement.** Weekly scoring of all users runs on serverless CPU with no command over 1 hour; batching is allowed.
# MAGIC Total time and DBUs per weekly run are the main tiebreaker after accuracy. **Measure, don't estimate.**
# MAGIC
# MAGIC This notebook times each scoring path on the user sample and extrapolates linearly. Treat the extrapolation as a
# MAGIC first estimate, then confirm the finalist with one full run over all users. Writes `cost_timings` and
# MAGIC `reports/02_cost.md`.
# MAGIC
# MAGIC **The `mapInPandas` path needs the project importable inside Spark's Python workers.** On serverless that means
# MAGIC adding this repo as a dependency in the Environment panel (path to the repo root; it has a `pyproject.toml`), then
# MAGIC applying. If that step fails, record it and use the driver-batch timings.
# MAGIC
# MAGIC **DBUs.** Not readable from inside a run. After the runs, join their job or notebook IDs to
# MAGIC `system.billing.usage` and add the cost per weekly run to the report.

# COMMAND ----------

# MAGIC %md
# MAGIC Skip the next cell if the notebook uses the **ML** base environment.

# COMMAND ----------

# MAGIC %pip install -q lightgbm torch scikit-learn scipy pyyaml tabulate

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

print(pipeline.environment_report(cfg))

# COMMAND ----------

timings = pipeline.run_cost(spark, cfg, report_path=f"{REPORTS}/02_cost.md")
display(timings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the table
# MAGIC - **Trees** pay for features every week (`build features (SQL)`) plus scoring.
# MAGIC - **HMM**: `advance state one week` plus `forecast from state` is the weekly cost; `filter whole history` is the
# MAGIC   one-off cold start. A run of weeks with no rows collapses to one cached matrix (`kernels_np.hmm_quiet_matrix`).
# MAGIC - **Sequence models** are timed recomputing from the whole history. The two-resolution GRU can instead carry its
# MAGIC   block-level state (`SeqModel.encode_blocks`); that weekly path is designed and unit-tested but not built into a job.
# MAGIC - **Long-lapsed tier**: for users lapsed over N weeks, compare the model's scores with the R1 lookup
# MAGIC   (`eval_metrics`, dormancy segments). If they match within tolerance, score that tier by lookup.
