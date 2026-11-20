# Databricks notebook source
# MAGIC %md
# MAGIC # 20 · Candidate A: gradient-boosted trees
# MAGIC
# MAGIC Four steps, **one cell each, each under the step limit**. See `src/ltv/models/trees.py` for what A1, A1tw and A2 are.
# MAGIC
# MAGIC | Step | What it does | Rerun? |
# MAGIC | --- | --- | --- |
# MAGIC | `tune` | Time-boxed random search for one shared hyperparameter set | **Rerun until it prints `complete: True`** |
# MAGIC | `fit` | Fits every model at the tuned settings, freezes tree counts, scores the validation origin | **Rerun until `complete: True`** |
# MAGIC | `final_1`, `final_2` | Refit with frozen settings, score the two test origins | **Rerun each until `complete: True`** |
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

print(pipeline.run_trees(spark, cfg, "tune"))      # rerun this cell until complete is True

# COMMAND ----------

r = pipeline.run_trees(spark, cfg, "fit")         # saves after each model; rerun until complete is True
print({k: v for k, v in r.items() if k != "rounds"})
print("trees per model:", r.get("rounds"))

# COMMAND ----------

print({k: v for k, v in pipeline.run_trees(spark, cfg, "final_1").items() if k in ("rows", "train_rows", "complete", "models_done")})

# COMMAND ----------

print({k: v for k, v in pipeline.run_trees(spark, cfg, "final_2").items() if k in ("rows", "train_rows", "complete", "models_done")})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explainability and a leakage check
# MAGIC Top features by gain. A calendar or carried-forward feature at the very top deserves a second look.

# COMMAND ----------

from ltv.models.trees import TreeSuite
from ltv.train.timebox import load_pickle

suite = TreeSuite.from_state(load_pickle(f"{cfg['artifact_path']}/trees/suite_tune.pkl"))
display(spark.createDataFrame(suite.importance("any_active")))
display(spark.createDataFrame(suite.importance("spend_tw")))
