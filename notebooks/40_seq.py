# Databricks notebook source
# MAGIC %md
# MAGIC # 40 · Candidate C: sequence models (GRU), on CPU
# MAGIC
# MAGIC **What this does.** Trains and scores a sequence model. It reads each player's weekly history directly (hours,
# MAGIC spend, platform, the season calendar) from the sequence shards exported by `10_build`, the same weeks the tree
# MAGIC features were computed from. Adding another weekly signal later means adding an input column per week here.
# MAGIC
# MAGIC **Run once per variant**; set the `variant` widget. Start with `C1`.
# MAGIC
# MAGIC | Variant | What it is |
# MAGIC | --- | --- |
# MAGIC | `C1` | Two-resolution GRU over the whole history: the main sequence candidate |
# MAGIC | `C1e` | GRU over play-spell tokens (one token per spell of play and per lapse) |
# MAGIC | `C2` | Small Transformer; about 5 times slower per step than C1. Run only if C1 works and the budget allows |
# MAGIC | `C1mix`, `C1ord`, `C1whale` | Output ablations: lognormal mixture, ordinal buckets, high-spender auxiliary task |
# MAGIC | `C1notab` | Input ablation: without the trees' features |
# MAGIC
# MAGIC **Reads** the sequence shards (and `labels` through them). **Writes** rows to `predictions` for the variant, and
# MAGIC the fitted models under `<artifact_path>/seq/<variant>`.
# MAGIC
# MAGIC | Step | What it does | Rerun? |
# MAGIC | --- | --- | --- |
# MAGIC | `tune` | Trains with early stopping on the validation origin | **Rerun until `complete: True`** |
# MAGIC | `score` | Scores the validation origin with the best weights | once |
# MAGIC | `final_1`, `final_2` | Retrains for the number of steps `tune` chose; scores the test origins | **rerun until complete** |
# MAGIC
# MAGIC Keep an ablation only if it lowers aggregate bias or top-1% bias on validation without losing Gini.

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

# MAGIC %md
# MAGIC ## The shared inputs
# MAGIC Every model notebook prints this same table: the scored players at each evaluation origin, by split, with a
# MAGIC fingerprint of exactly which players they are. If two notebooks show the same fingerprints, their models are
# MAGIC being compared on the same people, the same weeks and the same outcomes.

# COMMAND ----------

display(pipeline.describe_inputs(spark, cfg))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: tune (train with early stopping)

# COMMAND ----------

print(pipeline.run_seq(spark, cfg, "tune", VARIANT))      # rerun until complete is True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: score the validation origin

# COMMAND ----------

print(pipeline.run_seq(spark, cfg, "score", VARIANT))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Steps 3 and 4: the two final fits, scored at the test origins

# COMMAND ----------

print(pipeline.run_seq(spark, cfg, "final_1", VARIANT))   # rerun until complete

# COMMAND ----------

print(pipeline.run_seq(spark, cfg, "final_2", VARIANT))   # rerun until complete

# COMMAND ----------

MODELS = [VARIANT]

# COMMAND ----------

# MAGIC %md
# MAGIC ## What this notebook wrote
# MAGIC One row per model, fit and origin. `same_users` must be true everywhere: the model scored exactly the shared
# MAGIC players. `50_eval` refuses to compare a model for which it is false.

# COMMAND ----------

check = pipeline.check_same_users(spark, cfg)
display(check[check["model_id"].isin(MODELS)])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning curve (C1 only)
# MAGIC Validation loss as the number of training users grows. **If it is still falling at the largest size, CPU
# MAGIC capacity, not the model, set the result. Say so in the final report.** Rerun the cell until every size is present.

# COMMAND ----------

curve = pipeline.run_seq_learning_curve(spark, cfg, VARIANT)
display(curve)

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `50_eval` compares every model in `predictions`; `55_examples` builds the player-level plots.
