# Databricks notebook source
# MAGIC %md
# MAGIC # 40 · Candidate C: sequence models (CPU)
# MAGIC
# MAGIC Run once per variant; set the `variant` widget.
# MAGIC
# MAGIC | Variant | What it is |
# MAGIC | --- | --- |
# MAGIC | `C1` | Two-resolution GRU over the whole history (the main sequence candidate) |
# MAGIC | `C1e` | GRU over play-spell tokens |
# MAGIC | `C2` | Small Transformer; about 5 times slower per step than C1. Run only if C1 works and the budget allows |
# MAGIC | `C1mix`, `C1ord`, `C1whale` | Head ablations: lognormal mixture, ordinal buckets, high-spender auxiliary task |
# MAGIC | `C1notab` | Input ablation: without the trees' features |
# MAGIC
# MAGIC | Step | What it does | Rerun? |
# MAGIC | --- | --- | --- |
# MAGIC | `tune` | Trains with early stopping on the validation origin | **Rerun until `complete: True`** |
# MAGIC | `score` | Scores the validation origin with the best weights | once |
# MAGIC | `final_1`, `final_2` | Retrains for the number of steps `tune` chose; scores the test origins | **rerun until complete** |
# MAGIC
# MAGIC Keep an ablation only if it lowers aggregate bias or top-1% bias on validation without losing Gini; ZILN is the default.

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

dbutils.widgets.dropdown("variant", "C1", sorted(pipeline.SEQ_VARIANTS), "variant")
VARIANT = dbutils.widgets.get("variant")

# COMMAND ----------

print(pipeline.run_seq(spark, cfg, "tune", VARIANT))      # rerun until complete is True

# COMMAND ----------

print(pipeline.run_seq(spark, cfg, "score", VARIANT))

# COMMAND ----------

print(pipeline.run_seq(spark, cfg, "final_1", VARIANT))   # rerun until complete

# COMMAND ----------

print(pipeline.run_seq(spark, cfg, "final_2", VARIANT))   # rerun until complete

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning curve (C1 only)
# MAGIC Validation loss as the number of training users grows. **If it is still falling at the largest size, CPU
# MAGIC capacity, not the model, set the result. Say so in the final report.** Rerun the cell until every size is present.

# COMMAND ----------

curve = pipeline.run_seq_learning_curve(spark, cfg, VARIANT)
display(curve)
