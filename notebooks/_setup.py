# Databricks notebook source
# MAGIC %md
# MAGIC # Shared setup
# MAGIC
# MAGIC Every notebook starts with `%run ./_setup`. It:
# MAGIC
# MAGIC 1. finds the repo root (widget `repo_root`; by default, the nearest folder above this notebook that holds `src/ltv`),
# MAGIC 2. puts `src/` on the Python path, so the notebooks run the code in this Git folder,
# MAGIC 3. checks that the libraries are present,
# MAGIC 4. loads the config chosen by the widgets and defines `cfg`, `pipeline` and `REPORTS`.
# MAGIC
# MAGIC **Widgets**
# MAGIC - `config`: `bakeoff` (real data) or `synth` (synthetic smoke test, separate schema).
# MAGIC - `profile`: `quick` (about 1 command-hour per model family, to shake out the pipeline) or `full`.
# MAGIC - `repo_root`: leave blank unless the default is wrong.
# MAGIC
# MAGIC **Libraries.** Pick the **ML** base environment in the notebook's Environment panel (it ships
# MAGIC LightGBM and PyTorch), or run the `%pip` cell at the top of the calling notebook. Do **not**
# MAGIC install PySpark on serverless: it breaks the session.

# COMMAND ----------

import os
import sys

dbutils.widgets.dropdown("config", "bakeoff", ["bakeoff", "synth"], "config")
dbutils.widgets.dropdown("profile", "quick", ["quick", "full"], "budget profile")
dbutils.widgets.text("repo_root", "", "repo root (blank = auto)")

REPO_ROOT = dbutils.widgets.get("repo_root").strip()
if not REPO_ROOT:
    # The working directory is not guaranteed on serverless, so derive the path from the notebook itself.
    nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    REPO_ROOT = "/Workspace" + os.path.dirname(os.path.dirname(nb))
    for _ in range(3):                      # notebooks in a subfolder (notebooks/parked) sit one level deeper
        if os.path.isdir(os.path.join(REPO_ROOT, "src", "ltv")):
            break
        REPO_ROOT = os.path.dirname(REPO_ROOT)
if not os.path.isdir(os.path.join(REPO_ROOT, "src", "ltv")):
    raise RuntimeError(f"{REPO_ROOT} does not look like the repo root (no src/ltv). Set the repo_root widget.")
if os.path.join(REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

missing = []
for lib in ("lightgbm", "torch", "sklearn", "scipy", "yaml", "tabulate"):
    try:
        __import__(lib)
    except ImportError:
        missing.append(lib)
if missing:
    raise RuntimeError(f"Missing libraries: {missing}. Use the ML base environment or run the %pip cell first.")

# COMMAND ----------

from ltv import pipeline
from ltv.config import load_config

_files = [os.path.join(REPO_ROOT, "conf", "bakeoff.yml")]
_over = {"budget": {"profile": dbutils.widgets.get("profile")}}
if dbutils.widgets.get("config") == "synth":
    _files.append(os.path.join(REPO_ROOT, "conf", "synth.yml"))
cfg = load_config(*_files, overrides=_over)
if dbutils.widgets.get("config") == "synth":
    # The smoke test writes its own source tables next to its outputs.
    _schema = ".".join(p for p in (cfg.get("catalog"), cfg["schema"]) if p)
    cfg["source"]["activity_table"] = f"{_schema}.synth_weekly_activity"
    cfg["source"]["calendar_table"] = f"{_schema}.synth_calendar"

REPORTS = os.path.join(REPO_ROOT, "reports")
print(f"repo root : {REPO_ROOT}")
print(f"config    : {dbutils.widgets.get('config')}  |  profile: {cfg['budget']['profile']}  |  "
      f"step limit: {cfg['budget']['step_minutes']} min")
print(f"outputs   : {'.'.join(p for p in (cfg.get('catalog'), cfg['schema']) if p)}   artifacts: {cfg['artifact_path']}")
print(f"sources   : {cfg['source']['activity_table']}, {cfg['source']['calendar_table']}")
