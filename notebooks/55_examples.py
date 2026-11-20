# Databricks notebook source
# MAGIC %md
# MAGIC # 55 · Example players, ready to plot
# MAGIC
# MAGIC **What this does.** Picks example players and writes one long table, `example_weekly`, with each player's weekly
# MAGIC history overlaid with what each model said, week by week. It is built for a plotting tool such as Tableau: one
# MAGIC row per player, week and series, with a single `hours` column to overlay.
# MAGIC
# MAGIC **Run after** at least one model notebook has finished its `final_1` step. Models without a saved fit are skipped.
# MAGIC
# MAGIC | `view` | `series` | What a row is |
# MAGIC | --- | --- | --- |
# MAGIC | `actual` | `actual` | The player's real week: `hours`, `active` (0 or 1), `spend` |
# MAGIC | `path` | a model | The forecast made at the test origin for that week: expected `hours`, P(`active`), expected `spend` |
# MAGIC | `as_of` | a model | The model's opinion at the end of that week, scored fresh from the history so far: `p_reengage_52`, the 52-week totals, and `hours` = expected hours per week over the next 4 weeks |
# MAGIC
# MAGIC **Who is in it.** `examples.per_group` players from the `test` split in each of about ten groups (serial returners in
# MAGIC a long lapse, first long lapse, hours falling, new players, high spenders, the players the models disagree on most,
# MAGIC ...), plus any ids in the `user_ids` widget. Test players were never used to train or tune a model.
# MAGIC
# MAGIC **Writes** tables `example_weekly` and `example_users`, and the same as CSV under `<artifact_path>/examples`.
# MAGIC Outputs are after the shared calibration layer unless `examples.calibrated` is false. Full column list:
# MAGIC `docs/DATA_CONTRACT.md`.

# COMMAND ----------

# MAGIC %md
# MAGIC Skip the next cell if the notebook uses the **ML** base environment.

# COMMAND ----------

# MAGIC %pip install -q lightgbm torch scikit-learn scipy pyyaml tabulate matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_setup

# COMMAND ----------

dbutils.widgets.text("user_ids", "", "extra user ids (comma separated)")
EXTRA = [u.strip() for u in dbutils.widgets.get("user_ids").split(",") if u.strip()]

# COMMAND ----------

info = pipeline.build_examples(spark, cfg, user_ids=EXTRA)
print(info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The players

# COMMAND ----------

display(spark.table(cfg.table("example_users")).orderBy("example_group", "user_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## A quick look at one player
# MAGIC Top: actual weekly hours and each model's forecast from the origin. Bottom: each model's P(re-engage within 52
# MAGIC weeks) as it stood at the end of each week. Change `GROUP` or set `PLAYER` to an id.

# COMMAND ----------

from ltv.eval import examples as ex

weekly = spark.table(cfg.table("example_weekly")).toPandas()
GROUP = "serial returner, long dormant"
PLAYER = None
pick = PLAYER or weekly.loc[weekly["example_group"] == GROUP, "user_id"].iloc[0]
display(ex.plot_player(weekly, pick))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plotting it yourself
# MAGIC Connect the plotting tool to `example_weekly` (or open `example_weekly.csv`). Three plots cover most questions:
# MAGIC
# MAGIC | Question | Filter | Columns (x) | Rows (y) | Color |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | How did the forecast compare with what happened? | one `user_id`; `view` in (`actual`, `path`) | `week_start` | `hours` | `series` |
# MAGIC | What did each model think during a lapse? | one `user_id`; `view` = `as_of` | `week_start` | `p_reengage_52` | `series` |
# MAGIC | Did the totals add up over the year? | one `user_id`; `view` in (`actual`, `path`); `weeks_from_origin` > 0 | `weeks_from_origin` | `cum_hours` | `series` |
# MAGIC
# MAGIC Put the first two on one dashboard, one above the other, sharing the date axis. A1 predicts 52-week totals only, so
# MAGIC its `path` is a flat line; A2 is the tree variant with a shape over time.
