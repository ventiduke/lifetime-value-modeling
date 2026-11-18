# Databricks notebook source
# MAGIC %md
# MAGIC # 15 · Reference baselines R1, R2, R3
# MAGIC
# MAGIC No tuning. **R1** segment lookup, **R2** persistence, **R3** discrete-time buy-till-you-die (BG/BB).
# MAGIC Every candidate is reported against all three. Writes rows to `predictions` for the three fits.
# MAGIC
# MAGIC **Time.** Minutes. R3 fits four parameters twice per fit by maximum likelihood.

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

info = pipeline.run_baselines(spark, cfg)
print(info)
