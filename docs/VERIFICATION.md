# Verification: what has been executed, and what has not

Written at the end of the build pass (2026-11-18); updated 2026-11-20 after the simulation
study. Update it whenever something here is run for the first time on Databricks, or stops
being true.

**Bottom line.** Every part of the pipeline has executed end to end on synthetic data, on
local Spark with Delta Lake. The models, training loops, calibration and evaluation have
also run on 2 million synthetic users without Spark (`reports/01_simulation_study.md`).
**Nothing has run on Databricks, on Spark Connect, on Unity Catalog, or on real data.** The
logic is tested; the platform is not.

## Where the build pass ran

| | |
| --- | --- |
| Machine | 4 CPU cores, 15 GB memory, no GPU |
| Python | 3.13 (Databricks serverless environment 6 has 3.12) |
| Spark | PySpark 4.0.1, classic local mode (not Spark Connect), ANSI SQL on, UTC |
| Tables | Delta Lake 4.0 (delta-spark), local catalog (not Unity Catalog) |
| Libraries | LightGBM 4.7, PyTorch 2.15 (CPU), pandas 3.0, NumPy 2.5, scikit-learn 1.9 |
| Data | Synthetic only (`docs/SYNTHETIC_DATA.md`) |

## What was executed

The test suite: **87 tests** (`pytest`, about 20 minutes; 85 fast tests plus two slow ones:
the end-to-end pipeline and the simulation driver).

| Area | Test file | What the tests establish |
| --- | --- | --- |
| Split schedule, future calendar | `test_splits.py` (7) | The schedule at 312 weeks; no training label window overlaps a scored window; too little history is an error; the projected calendar uses only what is known at the origin, and does not change when the future calendar changes |
| SQL data layer | `test_data_layer.py` (15) | Cleaning removes every planted defect; **SQL features equal the independent NumPy features on all 53 columns at three origins**; SQL population equals the NumPy population; labels are internally coherent; the sequence store reproduces the SQL labels table and the simulated truth; **deleting all data after an origin does not change that origin's features**; the dense panel only looks backward; **every audit query reports its planted defect** |
| R3 (BG/BB) | `test_bgbb.py` (4) | Closed-form expected transactions and P(no transaction) match a simulation of the model; the fit recovers forecasts; the fit survives a sample where one history pattern dominates |
| HMM | `test_hmm.py` (11) | B2's transition structure; **EM never decreases the likelihood**; a resumed fit equals an uninterrupted one; an expired time box changes nothing; shards give the same fit as one store; nothing after the origin is read; planted spend classes are recovered; forecasts are calibrated, coherent and track the oracle; **B2 separates serial returners from churned players and B1 cannot**; NumPy kernels equal the model, including the collapsed run of absent weeks |
| Metrics, calibration, comparison | `test_metrics.py` (16) | Each metric against hand-computed values (and scikit-learn where one exists); recalibration fixes level without reordering; **recalibration survives lumpy predictions that shift between fits, where plain isotonic regression fails**; the map outside the validation range; conformal intervals cover at their level, by segment; the paired bootstrap finds a real difference and not a fake one; **the leader is compared with the best model of another family**; coherence counting; **revisions of a true model average zero, and selecting on the outcome makes them negative** |
| Baselines and trees | `test_tabular_models.py` (11) | R1 keys on past returns; R2 repeats last year; trees beat persistence; A2 is coherent by construction; final fits use the frozen tree counts; **a time-boxed fit resumes model by model**; save and load round-trips; ZILN quantiles match sampling; the random search resumes without repeating trials; files are written through local disk |
| Sequence models | `test_seq.py` (17) | The block grid is fixed (earlier blocks never change); inputs cover the whole history and read nothing after the origin; episode tokens on a hand-built history; head quantiles and means; every encoder and head trains; **the history encoder's state can be carried forward**, and the NumPy GRU step matches; training lowers validation loss; predictions are coherent; **the model gives returners a higher return probability**; a resumed run follows the same path; final fits run the frozen number of steps; learning-rate drops are recorded by the tune fit and replayed by final fits |
| Whole pipeline | `test_end_to_end.py` (1, slow) | `pipeline.smoke_test` on 3,000 synthetic users: source tables, audit, build, baselines, trees, both HMMs, the GRU, the stack, evaluation, reports and the cost harness (including `mapInPandas` scoring) all run and produce complete, finite outputs; every step is recorded in the run log |

| Simulation driver, second world | `test_sim.py` (5; 1 slow) | The original generator's output is unchanged by the refactor; both worlds are written in chunks and resume; the rhythm world has personal lapse lengths and the Markov world does not; the whole driver runs every stage on 6,000 users, including same-fit calibration, and resumes |

**At scale, by hand (2026-11-20).** `python -m ltv.sim` on 2,000,000 synthetic users x 320
weeks in each of two worlds: features for 2.0M training rows, R1 to R3, the tree suite, both
HMMs, the GRU, three fits each, calibration and evaluation on 2.85M prediction rows. It uses
the pipeline's model, training and evaluation code but **not Spark, SQL or Delta**. The
report is `reports/01_simulation_study.md`; the defects it found are in `DECISIONS.md`,
section 5.1.

Also run by hand during the build, on 8,000 synthetic users: every sequence variant
including the Transformer (`C2`) and the mixture and ordinal heads; the learning-curve
function; the results and cost reports.

### Local throughput, for orientation only

4 cores. These say nothing about serverless; the smoke test measures the real numbers.

| Step | Local speed |
| --- | --- |
| Synthetic data, 50,000 users x 320 weeks | about 6 s |
| Full data build, 4,000 users | about 1.5 min (Spark start-up dominates) |
| Tree suite, 24 models, 140,000 rows, quick profile | about 35 s |
| HMM B2 (60 states), 14,500 users, to convergence | about 3.5 min, 19 EM iterations |
| GRU `C1`, batch 512 | about 13 steps per second |
| Transformer `C2`, batch 512 | about 3 steps per second |

## What was NOT executed

| Item | Status |
| --- | --- |
| Anything on Databricks | Never run |
| Spark Connect | Tests use classic Spark. The code avoids RDDs, `sparkContext`, caching and session tricks by design, but that is untested |
| Unity Catalog: three-part names, `CREATE VOLUME`, permissions | Never run. Tests use two-part names in a local catalog |
| Writing checkpoints and shards to a volume | Never run. Writes go through local disk and a sequential copy (`ltv.io`) because volumes reject random-access and append writes; the rename-then-fallback path is untested there |
| The notebooks themselves | They compile as Python and call tested functions, but `dbutils`, widgets, `%run`, `%pip` and the repo-root detection in `_setup.py` have never run |
| `mapInPandas` on serverless | Runs locally. On serverless it needs the project importable in Python workers, inside a 1 GiB sandbox |
| The Asset Bundle (`jobs/databricks.yml`) | A skeleton from documentation; never validated or deployed |
| The Genie Code skill | Written to the documented format; never loaded |
| Real data | None seen. Column types, null patterns, duplicates, scale and week definitions are all assumptions until Phase 0 |
| Scale | Models and evaluation: 2M users generated, 1M used for tree and GRU training, without Spark. **The Spark and SQL build**: largest run still 50,000 users generated, 8,000 built. Nothing near 150M |
| The `full` budget profile | Only `quick` has been run. The simulation used its own settings (300 tree rounds, 30 EM passes, 10,000 GRU steps) |
| Tree hyperparameter search at scale | Run only on a few thousand users. The simulation used default settings |
| Same-fit calibration in the Databricks pipeline | Not built (`DECISIONS.md`, open question 8). `evaluate` supports it and the simulation used it |
| MLflow logging | Every `run_*` step is logged by `ltv.tracking` to `run_log.jsonl` (tested) and to MLflow when it is importable. **The MLflow branch has never run**: MLflow was not installed in the build environment |
| DBU and cost figures | Cannot be measured here; the harness records time only |
| GRU carried-state weekly scoring job | Not built (the property it relies on is tested) |
| Revision check in a notebook | Functions and tests exist; not wired into a notebook |

## Most likely first failures on Databricks

In rough order of likelihood. Fix them in the smoke test (`config = synth`), not on real data.

1. **`_setup.py` cannot find the repo root.** It reads the notebook path through
   `dbutils.notebook.entry_point`, which may behave differently on serverless. *Fix*: set the
   `repo_root` widget to the Git folder's `/Workspace/...` path.
2. **Libraries missing.** *Fix*: use the ML base environment, or run the `%pip` cell. The
   plain `torch` wheel is large; if installs are slow, use a CPU-only wheel if the
   workspace's network policy allows its index.
3. **Schema or volume cannot be created** (permissions), or `artifact_path` is not writable.
   *Fix*: point `schema` and `artifact_path` at places you own.
4. **A Spark Connect difference**: an API that works on classic Spark but not on Connect.
   The likeliest spots are `spark.createDataFrame` from large pandas frames (uploads are
   chunked at 50,000 to 500,000 rows; lower the chunk if a message-size error appears),
   `toPandas` on wide frames, and `spark.catalog.tableExists`.
5. **File writes to the volume fail or are slow.** See `ltv.io.write_via_local`. If rename
   is unsupported the code falls back to a direct copy. If `np.load` or `torch.load` from a
   volume is slow, copy the file to local disk first.
6. **ANSI SQL errors on real data**: a cast or division that classic Spark would have turned
   into NULL. The audit queries cast `hours` and `spend` to DOUBLE; a non-numeric value will
   raise. *Fix the data mapping; do not turn ANSI off.*
7. **`ALTER TABLE ... CLUSTER BY` is rejected.** It is wrapped and only prints a warning.
   Set `cluster_tables: false` to silence it.
8. **Out of memory at 16 GB.** Tree training frames are capped at
   `training.max_tabular_rows` (5M rows, about 2.5 GB as float32, but conversion needs
   headroom); shards at `training.seq_shard_users`. Lower either.
9. **A step cannot finish inside the time box** at real sample sizes. The step is
   resumable; if even one unit of progress (one tuning trial, one EM pass over all shards)
   exceeds the box, shrink the sample or the shard count for that step.
10. **`mapInPandas` cannot import `ltv`** in the cost notebook. Use the driver-batch timings
    and record that the parallel path needs the project installed in the environment.

## Things seen on synthetic data that are worth knowing

None of these is a model comparison. They are behaviors to expect.

- **ZILN level bias.** After a short training run, the GRU's ZILN heads over-predicted
  synthetic hours by about 24% and spend by about 60% in aggregate, while ranking well.
  This is the head's known weakness. The shared recalibration layer corrects the level; the
  mixture and ordinal heads are there to test whether another head needs less correction.
- **Recalibration needs enough validation users.** With about 200 validation users the
  isotonic layer added noise and could make a model look worse. Real runs have about 10% of
  the sample (100,000 users at a 1M sample). Do not judge models through the calibration
  layer on tiny samples.
- **B1 cannot see return history.** It gave every long-lapsed user the same return
  probability regardless of past returns. That is the memoryless property, and why B2 exists.
- **B2 needs enough EM iterations.** It converged in about 20 passes; cut off at 15 it was
  noticeably worse. The `quick` profile allows 30, `full` allows `hmm.max_iters` (100).
- **A plain HMM with user types but no spend classes under-predicted spend by half.**
  Crossing types with spend classes fixed it (`DECISIONS.md`, section 5).
- **The Transformer is about 5 times slower per step than the GRU** on CPU.
- **Cost extrapolations from a few thousand users are dominated by fixed overhead** (Spark
  job start-up). Only trust them from the real sample.

## How to re-verify

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

In Databricks, `notebooks/05_synth_smoke_test` with `config = synth` runs the same
end-to-end function as the slow test. When it passes there, move the corresponding rows
from "not executed" to "executed" in this file, with the date and the environment.
