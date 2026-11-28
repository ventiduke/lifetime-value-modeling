# Databricks notebook source
# MAGIC %md
# MAGIC # 70 · Register a model in MLflow (no retraining)
# MAGIC
# MAGIC **What this does.** Takes a fit the bake-off already trained and saved, packages it as one MLflow model, and
# MAGIC registers it in Unity Catalog. **Nothing is trained here.** The registered model is the saved fit, with its
# MAGIC calibration layer and the code it needs.
# MAGIC
# MAGIC **Run after** the model's notebook (`20_trees` or `40_seq`) has finished its `final_1` step, and normally after
# MAGIC `50_eval`, so the choice of model is made on the evaluation.
# MAGIC
# MAGIC **What the registered model takes and returns** (the same for every model, so the caller does not change when
# MAGIC the winner does):
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | Input | Rows of `user_sequences`: `user_id`, `first_week`, `hours[]`, `spend[]` (one array element per week) |
# MAGIC | Parameters | `origin_week` (the scoring week; default: the last week of data), `calibrated`, and the current season calendar |
# MAGIC | Output | `user_id`, `origin_week`, `in_population`, and the five 52-week outputs |
# MAGIC
# MAGIC **Widgets**
# MAGIC - `model_id`: `A1`, `A2`, `C1`, ... One model per run of this notebook.
# MAGIC - `fit`: which saved fit. `final_1` is the one scored at the latest test week.
# MAGIC - `registered_name`: `catalog.schema.model_name` in Unity Catalog. You need `CREATE MODEL` on the schema.
# MAGIC   Leave blank to log the model to the experiment without registering it.
# MAGIC - `calibration`: `tune_fit` ships the layer the evaluation used, so served numbers are the evaluated ones.
# MAGIC   `same_fit` fits the layer on this fit itself (`docs/DECISIONS.md`, open question 8). `none` ships raw outputs.

# COMMAND ----------

# MAGIC %md
# MAGIC Skip the next cell if the notebook uses the **ML** base environment.

# COMMAND ----------

# MAGIC %pip install -q lightgbm torch scikit-learn scipy pyyaml tabulate mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

dbutils.widgets.dropdown("model_id", "A2", ["A1", "A1tw", "A2"] + sorted(pipeline.SEQ_VARIANTS), "model")
dbutils.widgets.dropdown("fit", "final_1", ["final_1", "final_2", "tune"], "fit")
dbutils.widgets.text("registered_name", "", "registered name (catalog.schema.model)")
dbutils.widgets.dropdown("calibration", "tune_fit", ["tune_fit", "same_fit", "none"], "calibration")
MODEL_ID, FIT = dbutils.widgets.get("model_id"), dbutils.widgets.get("fit")
NAME = dbutils.widgets.get("registered_name").strip() or None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: log and register
# MAGIC The new version gets the alias `candidate`. The step ends by loading the model back and scoring a sample of
# MAGIC test players, then comparing with their rows in `predictions`.

# COMMAND ----------

info = pipeline.register_model(spark, cfg, MODEL_ID, FIT, name=NAME, calibration=dbutils.widgets.get("calibration"))
print({k: v for k, v in info.items() if k != "verify"})
print("verify:", info["verify"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check: the registered model is the model that was evaluated
# MAGIC `matches` must be true: the registered model reproduces the bake-off's predictions for the same players to within
# MAGIC the tolerance. For the trees this also confirms that features rebuilt from a player's history equal the SQL
# MAGIC features. If it is false, stop and look at which output differs before using the model.

# COMMAND ----------

assert info["verify"]["matches"], f"the registered model does not reproduce the evaluated predictions: {info['verify']}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: score with the registered model
# MAGIC Loads the model by its alias and scores the sampled players at the latest week of data. `n_slices` splits the
# MAGIC players into hashed slices so the driver holds one slice at a time.

# COMMAND ----------

from ltv import serve

URI = f"models:/{NAME}@candidate" if NAME else info["model_uri"]
calendar = spark.table(cfg.table("calendar_weeks")).select("week_idx", "season_week", "season_idx").toPandas()
scores = serve.score_sequences(spark, cfg.table("user_sequences"), URI, origin_week=None, calendar=calendar,
                               n_slices=8, where="split = 'test'")
display(spark.createDataFrame(scores.head(1000)))
print(len(scores), "players scored at week", int(scores["origin_week"].iloc[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 (the owner's decision): name the champion
# MAGIC Run this only once the winner is chosen. It points the alias `champion` at the version registered above, and
# MAGIC production loads `models:/<name>@champion`. Nothing else changes.

# COMMAND ----------

# serve.set_alias(NAME, "champion", info["version"])

# COMMAND ----------

# MAGIC %md
# MAGIC **About freshness.** The bake-off fits are trained on the player sample, with training data that stops a year
# MAGIC before the test week. The registered model is exactly that fit. Refitting once on the latest data, with the
# MAGIC settings frozen, is a separate step and a separate decision.
