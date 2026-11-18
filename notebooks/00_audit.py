# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Phase 0 data audit
# MAGIC
# MAGIC **Purpose.** Check the real source tables before anything is modeled. **Stop for review when this notebook finishes.**
# MAGIC
# MAGIC **Before running.** Set the PLACEHOLDER values in `conf/bakeoff.yml`: catalog, schema, artifact path, the two
# MAGIC source tables and their column mapping. Run `05_synth_smoke_test` first; it proves the environment works.
# MAGIC
# MAGIC **Reads** the raw activity and calendar tables. **Writes** `calendar_weeks`, `activity_clean`, `dim_*`,
# MAGIC `users` (the clean, sampled copy the later checks need) and `reports/00_data_audit.md`.
# MAGIC
# MAGIC **Time.** Each cell is a few full scans of the activity table; all are well under the 1-hour limit on a sample,
# MAGIC but the raw-table scans read every row. If one runs long, check that the table is clustered or partitioned by week.

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
# MAGIC ## 1. Checks on the raw tables (a01 to a08)

# COMMAND ----------

pipeline.ensure_storage(spark, cfg)
audit = pipeline.run_audit(spark, cfg, include_clean=False)
findings = pipeline.audit_findings(audit)
display(spark.createDataFrame(findings))

# COMMAND ----------

# MAGIC %md
# MAGIC **Read the findings before going on.**
# MAGIC - `FAIL` on *week alignment*: the week definition differs between the two tables. Fix the mapping; do not continue.
# MAGIC - `FAIL` on *weeks of history*: under 312 weeks. The split schedule needs about 235; below that, stop and escalate.
# MAGIC - `WARN` rows are handled by the cleaning rules in `sql/build/02_activity_clean.sql`. Confirm each rule is right
# MAGIC   for this data (negative spend, duplicate user-weeks, suspect accounts) and change `cleaning:` in the config if not.
# MAGIC
# MAGIC Things the data cannot tell you; ask the table owners: the time zone of `week_start`; whether spend is net of
# MAGIC refunds and in one currency; which accounts are bots, banned or merged (supply them through
# MAGIC `source.exclude_users_table`); how far ahead season launch dates are known (sets `calendar.known_window`).

# COMMAND ----------

for name in ("a06_history_and_users", "a02_row_semantics", "a03_spend_distribution", "a04_duplicate_user_weeks",
             "a05_week_alignment", "a08_seasons"):
    print(name)
    display(spark.createDataFrame(audit[name]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build the clean sampled tables, then the checks that need them (a09, a10)

# COMMAND ----------

info = pipeline.build_clean(spark, cfg)
print(info)
sched = pipeline.get_schedule(spark, cfg)     # raises with a clear message if history is too short
print(sched)

# COMMAND ----------

audit = pipeline.run_audit(spark, cfg, include_clean=True)
findings = pipeline.audit_findings(audit)
display(spark.createDataFrame(audit["a09_dormancy_base_rates"]))
display(spark.createDataFrame(audit["a10_lapse_and_return"]))

# COMMAND ----------

path = pipeline.write_audit_report(audit, findings, f"{REPORTS}/00_data_audit.md")
print(f"wrote {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate: stop here
# MAGIC
# MAGIC Add to `reports/00_data_audit.md`, under a "Proposed settings" heading, and wait for sign-off:
# MAGIC 1. the user sample size (`sample.fraction`), given the base rates in a09 and the returner counts in a10;
# MAGIC 2. the split schedule printed above;
# MAGIC 3. `calendar.known_window`, from what the calendar's owners say about launch announcements;
# MAGIC 4. any change to the cleaning rules.
