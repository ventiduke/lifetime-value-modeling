# Databricks notebook source
# MAGIC %md
# MAGIC # 50 · Phase 3: evaluation
# MAGIC
# MAGIC One evaluation for every model in `predictions`. Writes `eval_metrics`, `eval_pairwise` and
# MAGIC `reports/03_bakeoff_results.md`.
# MAGIC
# MAGIC - Headline numbers: `test`-split users at the two test origins, after the shared recalibration layer.
# MAGIC - A model wins a target only if it leads at **both** test origins and the 95% interval of its paired difference
# MAGIC   from the runner-up excludes zero at both. The runner-up is the best model of another family (A, B, C).
# MAGIC - The **test origins are for final numbers only**. Do not go back and tune after looking at this notebook.

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

# Stack diagnostic (not a candidate): how much signal do the trees and the sequence model each miss?
# Needs A2 and C1 predictions. Skip this line if either is absent.
print(pipeline.run_stack(spark, cfg, "A2", "C1"))

# COMMAND ----------

res = pipeline.run_eval(spark, cfg, report_path=f"{REPORTS}/03_bakeoff_results.md")
display(res["winners"])

# COMMAND ----------

display(res["scorecard"].reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lapse-and-return (the headline cut for long-term behavior)
# MAGIC Mean predicted against actual P(re-engage), by past returns from a 26-week lapse and current dormancy.

# COMMAND ----------

m = res["metrics"]
cut = m[(m.segment_dim == "returner_x_dormancy") & (m.target == "p_reengage") & (m.split == "test") & m.calibrated
        & m.metric.isin(["mean_pred", "mean_actual", "log_loss"])]
display(cut.pivot_table(index=["segment", "origin_week", "n"], columns=["model_id", "metric"], values="value").reset_index())

# COMMAND ----------

display(res["coherence"])
display(res["intervals"])
display(res["pairwise"])
