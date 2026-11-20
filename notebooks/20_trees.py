# Databricks notebook source
# MAGIC %md
# MAGIC # 20 · Candidate A: gradient-boosted trees
# MAGIC
# MAGIC **What this does.** Trains and scores the tree models. They read a table of about 50 features summarising each
# MAGIC player's history (`features_tabular`), built in SQL by `10_build`.
# MAGIC
# MAGIC | Model | What it is |
# MAGIC | --- | --- |
# MAGIC | `A1` | One model per target, each predicting its 52-week total directly |
# MAGIC | `A1tw` | A1 with a different spend model (Tweedie) |
# MAGIC | `A2` | The coherent variant: predicts five time buckets and adds them up, so its outputs agree with each other and have a shape over time |
# MAGIC
# MAGIC **Reads** `features_tabular` and `labels`. **Writes** rows to `predictions` for `A1`, `A1tw`, `A2`, and the fitted
# MAGIC models under `<artifact_path>/trees`.
# MAGIC
# MAGIC **How to run.** Four steps, one cell each. Each stays under the step limit and saves its progress.
# MAGIC
# MAGIC | Step | What it does | Rerun? |
# MAGIC | --- | --- | --- |
# MAGIC | `tune` | Time-boxed random search for one shared set of settings, judged on the validation origin | **Rerun until it prints `complete: True`** |
# MAGIC | `fit` | Fits every model at the tuned settings, fixes the number of trees, scores the validation origin | **Rerun until `complete: True`** |
# MAGIC | `final_1`, `final_2` | Refits with those settings on later data and scores the two test origins | **Rerun each until `complete: True`** |
# MAGIC
# MAGIC Training data is capped at `training.max_tabular_rows` rows held in memory. If a cell runs out of memory, lower it.

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

# MAGIC %md
# MAGIC ## Step 1: tune

# COMMAND ----------

print(pipeline.run_trees(spark, cfg, "tune"))      # rerun this cell until complete is True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: fit at the tuned settings and score the validation origin

# COMMAND ----------

r = pipeline.run_trees(spark, cfg, "fit")         # saves after each model; rerun until complete is True
print({k: v for k, v in r.items() if k != "rounds"})
print("trees per model:", r.get("rounds"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Steps 3 and 4: the two final fits, scored at the test origins

# COMMAND ----------

print({k: v for k, v in pipeline.run_trees(spark, cfg, "final_1").items() if k in ("rows", "train_rows", "complete", "models_done")})

# COMMAND ----------

print({k: v for k, v in pipeline.run_trees(spark, cfg, "final_2").items() if k in ("rows", "train_rows", "complete", "models_done")})

# COMMAND ----------

MODELS = ["A1", "A1tw", "A2"]

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
# MAGIC ## What the trees rely on, and a leakage check
# MAGIC Top features by gain. A calendar or carried-forward feature at the very top deserves a second look.

# COMMAND ----------

from ltv.models.trees import TreeSuite
from ltv.train.timebox import load_pickle

suite = TreeSuite.from_state(load_pickle(f"{cfg['artifact_path']}/trees/suite_tune.pkl"))
display(spark.createDataFrame(suite.importance("any_active")))
display(spark.createDataFrame(suite.importance("spend_tw")))

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `40_seq` for the sequence models, then `50_eval`. The first time through, stop after this notebook,
# MAGIC run `50_eval` once and review A1 end to end before training anything else.
