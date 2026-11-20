# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Synthetic smoke test (run this first)
# MAGIC
# MAGIC **Purpose.** Prove that this workspace can run the whole pipeline before any real table is read: libraries,
# MAGIC permissions, the output schema, the artifact volume, Spark Connect behaviour, and every step end to end: build,
# MAGIC baselines, trees, the GRU, evaluation, the example players and the cost harness.
# MAGIC It also measures the CPU and training throughput that size the real runs.
# MAGIC
# MAGIC **Data.** Generated in this notebook, with planted defects, into its own schema (`conf/synth.yml`).
# MAGIC **Nothing here says anything about which model is better.** A generated world rewards the model that resembles it.
# MAGIC
# MAGIC **Set the widgets** to `config = synth`, `profile = quick`. First set `catalog` in `conf/bakeoff.yml` and
# MAGIC `artifact_path` in `conf/synth.yml` to places you can write.
# MAGIC
# MAGIC **Time.** One long cell. Locally (4 cores, 4,000 users) it takes about 10 minutes; with the default 50,000 users
# MAGIC expect more. If it nears an hour, lower `synthetic.n_users` in `conf/synth.yml`. Every step inside it is resumable.

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

assert dbutils.widgets.get("config") == "synth", "set the config widget to 'synth' for the smoke test"
print(pipeline.environment_report(cfg))

# COMMAND ----------

result = pipeline.smoke_test(spark, cfg, seq_variants=("C1",), report_dir=REPORTS)
print(f"finished in {result['seconds']} s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to check
# MAGIC 1. **Audit findings** below: each planted defect should show as WARN or FAIL (week alignment FAILs on purpose).
# MAGIC 2. **Predictions**: every model has rows for all three fits.
# MAGIC 3. **Scorecard**: numbers exist and are finite. Do not read them as a ranking.
# MAGIC 4. **Example players**: the step ran and wrote `example_weekly` (see `55_examples` for what it holds).
# MAGIC 5. **Timings**: these and the environment report go into the Phase 0 report; they size the real runs.

# COMMAND ----------

display(spark.createDataFrame(result["audit_findings"]))
display(spark.table(cfg.table("predictions")).groupBy("model_id").pivot("fit").count().orderBy("model_id"))

# COMMAND ----------

display(result["eval"]["scorecard"].reset_index())
print(result["examples"])
display(result["cost"])
print(result["environment"])
