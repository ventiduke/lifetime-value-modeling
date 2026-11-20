# Databricks notebook source
# MAGIC %md
# MAGIC # 15 · Reference baselines R1, R2, R3
# MAGIC
# MAGIC **What this does.** Scores three simple reference models that every candidate is reported against. No tuning.
# MAGIC
# MAGIC | Model | What it is |
# MAGIC | --- | --- |
# MAGIC | `R1` | Segment lookup: the average outcome of past players in the same segment (recency, past returns, tenure, spend) |
# MAGIC | `R2` | Persistence: the last 52 weeks, repeated |
# MAGIC | `R3` | Buy-till-you-die (BG/BB): a classic two-parameter-pair model of activity and of spending |
# MAGIC
# MAGIC **Reads** the shared tables built by `10_build` (`features_tabular`, `labels`). **Writes** rows to `predictions`
# MAGIC for the three fits (`tune`, `final_1`, `final_2`).
# MAGIC
# MAGIC **Time.** About 20 minutes; nearly all of it is R3's three fits.

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

# MAGIC %md
# MAGIC ## The shared inputs
# MAGIC Every model notebook prints this same table: the scored players at each evaluation origin, by split, with a
# MAGIC fingerprint of exactly which players they are. If two notebooks show the same fingerprints, their models are
# MAGIC being compared on the same people, the same weeks and the same outcomes.

# COMMAND ----------

display(pipeline.describe_inputs(spark, cfg))

# COMMAND ----------

info = pipeline.run_baselines(spark, cfg)
print(info)

# COMMAND ----------

MODELS = ["R1", "R2", "R3"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## What this notebook wrote
# MAGIC One row per model, fit and origin. `same_users` must be true everywhere: the model scored exactly the shared
# MAGIC players. `50_eval` refuses to compare a model for which it is false.

# COMMAND ----------

check = pipeline.check_same_users(spark, cfg)
display(check[check["model_id"].isin(MODELS)])
