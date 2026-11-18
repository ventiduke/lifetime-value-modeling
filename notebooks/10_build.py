# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Phase 1: build the modeling tables
# MAGIC
# MAGIC **Run after** the Phase 0 gate is signed off and the config holds the agreed sample size and known-calendar window.
# MAGIC
# MAGIC **Writes** (all in the config's schema; see `docs/DATA_CONTRACT.md`): `origins`, `calendar_projected`,
# MAGIC `user_origins`, `features_tabular`, `labels`, `user_sequences`, `panel_weekly`, and the sequence shards
# MAGIC under the artifact path.
# MAGIC
# MAGIC **Time.** One SQL statement per table. `features_tabular` is the largest: it joins the sampled activity rows to
# MAGIC the user-origin pairs. If it runs long, lower `splits.max_train_origins_per_user`. Cells are safe to rerun.

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

info = pipeline.build_clean(spark, cfg)       # safe to repeat; the audit already built these
sched = pipeline.build_splits(spark, cfg)
print(sched)

# COMMAND ----------

print(pipeline.build_features(spark, cfg, with_panel=True))

# COMMAND ----------

shards = pipeline.export_sequences(spark, cfg)
print(f"{len(shards)} shard(s) under {cfg['artifact_path']}/shards")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks before modeling
# MAGIC Row counts per table; label base rates by origin role (they should be stable across origins); and the leakage
# MAGIC test, which is part of the test suite (`tests/test_data_layer.py`) and should be rerun here if the SQL was edited.

# COMMAND ----------

for t in ("activity_clean", "users", "user_origins", "features_tabular", "labels", "user_sequences", "panel_weekly"):
    print(f"{t:18s} {spark.table(cfg.table(t)).count():>14,}")

# COMMAND ----------

display(spark.sql(f"""
  SELECT o.role, l.origin_week, COUNT(*) AS n, AVG(any_active_52) AS p_reengage, AVG(active_weeks_52) AS active_weeks,
         AVG(hours_52) AS hours, AVG(any_spend_52) AS p_spend, AVG(spend_52) AS spend
  FROM {cfg.table('labels')} l JOIN {cfg.table('origins')} o USING (origin_week)
  GROUP BY o.role, l.origin_week ORDER BY l.origin_week, o.role"""))
