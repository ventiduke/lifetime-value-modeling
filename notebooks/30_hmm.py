# Databricks notebook source
# MAGIC %md
# MAGIC # 30 · Candidate B: hidden Markov models
# MAGIC
# MAGIC **B1** is a plain HMM. **B2** is the lapse-aware HMM: a chain of dormancy states, and user types crossed with
# MAGIC spend classes (see `src/ltv/models/hmm.py`). Fitting is EM, streamed over the sequence shards.
# MAGIC
# MAGIC | Step | What it does | Rerun? |
# MAGIC | --- | --- | --- |
# MAGIC | `tune` | EM for every candidate size in `hmm:` of the config, on weeks up to the validation origin | **Rerun until `complete: True`** |
# MAGIC | `score` | Picks the best B1 and best B2 by validation log-likelihood; scores the validation origin | once |
# MAGIC | `final_1`, `final_2` | Refits the chosen models on weeks up to each test origin (warm-started); scores it; writes `hmm_user_state` | **rerun until complete** |
# MAGIC
# MAGIC EM resumes exactly from its checkpoint, so rerunning a cell continues where it stopped.

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

r = pipeline.run_hmm(spark, cfg, "tune")           # rerun until complete is True
print("complete:", r["complete"])
display(spark.createDataFrame([{"model": k, **v} for k, v in r["candidates"].items()]))

# COMMAND ----------

r = pipeline.run_hmm(spark, cfg, "score")
print(r["chosen"], r["rows"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the states are
# MAGIC Describe each state by what it does, not by its label: how often it is active, how much it plays and spends,
# MAGIC and for dormant states how likely a return is in an ordinary week and in a launch week.

# COMMAND ----------

import pandas as pd
for fam in ("B1", "B2"):
    if f"profile_{fam}" in r:
        display(spark.createDataFrame(pd.DataFrame(r[f"profile_{fam}"])))

# COMMAND ----------

print({k: v for k, v in pipeline.run_hmm(spark, cfg, "final_1").items() if not k.startswith("profile")})   # rerun until complete

# COMMAND ----------

print({k: v for k, v in pipeline.run_hmm(spark, cfg, "final_2").items() if not k.startswith("profile")})   # rerun until complete
