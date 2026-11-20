# Runbook: the run pass, step by step

For whoever runs this in Databricks (Genie Code or a person). Read `AGENTS.md` first for the
rules. This file says what to do, in order, what each step reads and writes, how long it
should take, what to check, and what to do when it fails.

Nothing here has been run on Databricks yet. `docs/VERIFICATION.md` says what was executed
locally and where the first failures are most likely.

## Before you start

1. **Add the repo as a Git folder** in the workspace.
2. **Compute**: serverless notebooks, environment version 6. In each notebook's Environment
   panel pick the **ML** base environment (it ships LightGBM and PyTorch). If it is not
   available, each notebook has a `%pip install` cell to run instead. Do not install PySpark.
3. **Decide where outputs go**: a catalog and schema you can create tables in, and a Unity
   Catalog volume for artifacts (checkpoints, shards, fitted models). Put them in
   `conf/bakeoff.yml` (`catalog`, `schema`, `artifact_path`) and `conf/synth.yml`
   (`schema`, `artifact_path` for the smoke test). The pipeline creates the schema and the
   volume if they do not exist and you have the rights.
4. **Widgets**. Every notebook has `config` (`bakeoff` for real data, `synth` for the smoke
   test) and `profile` (`quick` or `full`). Use `quick` for a first pass through everything,
   then `full` for the numbers that count.

### The one-hour rule and resumable steps

No command may run over an hour. Long steps stop themselves at `budget.step_minutes`
(default 50) and save a checkpoint. They return `complete: False`.

**`complete: False` is not an error. Run the same cell again.** It continues from the
checkpoint. Repeat until it prints `complete: True`.

To start a step over from scratch, delete its checkpoint under `artifact_path` (paths are
listed in `docs/DATA_CONTRACT.md`).

## Step 1. `05_synth_smoke_test` (run this first)

| | |
| --- | --- |
| Widgets | `config = synth`, `profile = quick` |
| Reads | nothing real; generates synthetic data |
| Writes | the synth schema: source tables, every pipeline table, predictions, eval tables; `reports/05_smoke_test_*.md` |
| Time | one long cell; if it nears an hour lower `synthetic.n_users` in `conf/synth.yml` |

It runs `pipeline.smoke_test`: generate data with planted defects, audit, build, all four
model families, evaluation, cost harness. The same function is the slow test in the test
suite.

**Check**
- The audit findings show the planted defects (`week alignment` FAILS on purpose).
- `predictions` has every model with rows for `tune`, `final_1`, `final_2`.
- The scorecard has finite numbers. **Do not read it as a ranking.**
- Record `environment_report` (cores, memory, LightGBM and GRU throughput): it sizes the real
  runs and goes in the Phase 0 report.

**If it fails**: this is the place to fix environment problems. See "Troubleshooting" below
and `docs/VERIFICATION.md`. Do not move to real data until it passes.

## Step 2. `00_audit` (Phase 0) and the gate

| | |
| --- | --- |
| Widgets | `config = bakeoff` |
| Before | set `source:` in `conf/bakeoff.yml`: the two tables and the column mapping |
| Reads | the raw activity and calendar tables |
| Writes | `calendar_weeks`, `activity_clean`, `dim_platform`, `dim_country`, `users`; `reports/00_data_audit.md` |
| Time | each audit query scans the raw activity table once or twice |

**Check every finding.** A `FAIL` stops the work:
- *week alignment*: activity `week_start` values are not calendar weeks. Fix the mapping or
  ask the table owners. Misaligned rows would be silently dropped.
- *weeks of history*: under 312. The schedule needs at least about 235 weeks; the pipeline
  raises a clear error below that.

`WARN` findings are handled by the cleaning rules; confirm each rule suits the real data.

**Ask the owner or table owners** what the data cannot tell you: time zone of `week_start`;
whether spend is net of refunds and in one currency; which accounts to exclude; how far
ahead season launches are known.

**Gate. Stop here.** Add a "Proposed settings" section to `reports/00_data_audit.md` with:
the user sample size (`sample.fraction`), the split schedule the notebook printed, the
known-calendar window (`calendar.known_window`), any cleaning changes, and the smoke test's
environment report. Wait for sign-off.

## Step 3. `10_build` (Phase 1)

| | |
| --- | --- |
| Writes | `origins`, `calendar_projected`, `user_origins`, `features_tabular`, `labels`, `user_sequences`, `panel_weekly`, shards under `artifact_path/shards` |
| Time | one statement per table; `features_tabular` is the heaviest |

**Check**
- Row counts: `features_tabular`, `labels` and `user_origins` match.
- Base rates by origin (last cell) are stable; a jump at one origin suggests a data problem
  near it.
- If you edited any SQL in `sql/build/`, rerun `tests/test_data_layer.py` (or at least
  compare a few users against `ltv.data.np_features.tabular_features`).

**If `features_tabular` is too slow**: lower `splits.max_train_origins_per_user` (fewer
training rows) or `sample.fraction`. **If shards are too big for memory later**: lower
`training.seq_shard_users`.

## Step 4. `15_baselines`

R1, R2, R3 for all three fits. Minutes. No tuning.

## Step 5. `20_trees` (candidate A) and the second gate

| Cell | Step | Notes |
| --- | --- | --- |
| 1 | `tune` | rerun until `complete: True` |
| 2 | `fit` | freezes tree counts; scores the validation origin; rerun until `complete: True` |
| 3, 4 | `final_1`, `final_2` | score the test origins; rerun each until `complete: True` |

The fit steps save after each of the 24 models (`partial_<step>.pkl` under the trees artifact
directory). In simulation one boosting round over the whole suite cost about 0.12 s at 2M
rows on 4 cores, and about half the models wanted more than 300 rounds, so a fit at 5M rows
and up to 2,000 rounds is likely to need more than one command.

Then run `50_eval` once. **Gate**: the first model has gone through the whole pipeline end
to end. Review the report's shape (not who wins) with the owner before spending the rest of
the budget.

**If memory runs out**: lower `training.max_tabular_rows`.

## Step 6. `40_seq` (candidate C)

Run the notebook once per variant (widget `variant`): `C1` first, then `C1e`. Run `C2` and
the ablations (`C1mix`, `C1ord`, `C1whale`, `C1notab`) only if the budget allows.

| Cell | Step | Notes |
| --- | --- | --- |
| 1 | `tune` | rerun until `complete: True` |
| 2 | `score` | validation origin |
| 3, 4 | `final_1`, `final_2` | rerun until complete |
| 5 | learning curve | `C1` only; rerun until all sizes are present |

**Record `steps_per_second`.** With the batch size it tells you how many user-origins the
budget can train on. If the learning curve is still falling at the largest size, **say so in
the final report**: CPU capacity, not the model, set the sequence result.

**Look at the validation history** in the `tune` output (`last_eval`, and `history` in
`tune_info.pkl`). Early stopping watches one number on about 5,000 to 20,000 validation
users, at a constant learning rate; in simulation that number moved up and down by about
0.01 between checks and training stopped after 3,000 steps. If the history looks like noise
around a slow fall, set `seq.lr_drops: 2` and rerun `tune` from a clean artifact directory:
the learning rate is then cut (x0.3) twice before training stops, and the final fits cut it
at the same steps. This is tuning on the validation origin, which is allowed. In simulation
it took the GRU from 4,000 to 9,400 steps and closed most of its gap to the trees
(`reports/01_simulation_study.md`, section 7).

## Step 7. `50_eval` (Phase 3, accuracy)

Runs the stack diagnostic, then the one evaluation for every model in `predictions`. Writes
`eval_metrics`, `eval_pairwise` and `reports/03_bakeoff_results.md`.

Read, in order: winners by target; the scorecard; the lapse-and-return table; aggregate bias
before recalibration; coherence violations; interval coverage.

**Like for like.** The evaluation starts by checking that every model scored exactly the
shared players at each origin (`res["same_users"]`). If one did not, it stops and names
it: rerun that model's scoring step. Do not switch the check off to get a table.

**Do not tune anything after this.** The test origins are for final numbers.

## Step 8. `55_examples` (player-level examples to plot)

Writes `example_weekly` and `example_users` (and CSV copies under
`<artifact_path>/examples`): about 200 `test` players in ten groups, each with their actual
weekly history, every model's week-by-week forecast from the test origin (`view = path`),
and every model's opinion at the end of each week of a rolling window (`view = as_of`).
The notebook shows one player and lists three plots to build in a plotting tool.

- It needs each model's `final_1` fit. Models without one are skipped; rerun the notebook
  after another model finishes and it adds that model.
- Add players by id with the `user_ids` widget. Ids outside the `test` split are allowed
  but may have been trained on; the notebook's table shows who was picked and why.
- Settings are under `examples:` in the config (players per group, window, models).
- Takes minutes: a few hundred players are scored at about 150 weeks each.

## Step 9. `60_cost` (Phase 3, speed and cost)

Times each scoring path on the sample and extrapolates linearly. Writes `cost_timings` and
`reports/02_cost.md`. Then, for the finalist only, do one full-population scoring run and
record its time. Join job or notebook IDs to `system.billing.usage` for DBUs.

The `mapInPandas` path needs the project importable in Spark's Python workers: add the repo
root as a dependency in the Environment panel. If that fails, record it and use the driver
timings.

## Step 10. Final report

Finish `reports/03_bakeoff_results.md` by hand with: the scorecard (models x accuracy,
usability, cost); the learning curve; a 1-to-5 usability score per model with a one-line
reason (coherence, uncertainty, journey states, explainability, segmentation, maintenance);
time and cost per weekly run; the stack diagnostic; and a recommendation with the runner-up,
or a plain statement that there is no clear winner. If trees and the GRU are within their
bootstrap intervals of each other, say so plainly: the owner leans toward the GRU in a
tie, because further weekly signals can be added to it as input columns. Include a few
player plots from `55_examples` that show where the two differ.

## Parked: `notebooks/parked/30_hmm` (candidate B)

The owner cut the HMM on 2026-11-20 (`docs/DECISIONS.md`, section 1.1). Do not run this as
part of the bake-off. The notes below are kept for anyone who brings it back
(`hmm.enabled: true` puts it back in the smoke test).


| Cell | Step | Notes |
| --- | --- | --- |
| 1 | `tune` | EM for every size in `hmm:`; rerun until `complete: True` |
| 2 | `score` | picks the best B1 and B2 by validation log-likelihood |
| 3 | state profiles | describe states by what they emit |
| 4, 5 | `final_1`, `final_2` | rerun until complete; writes `hmm_user_state` |

**Check** the state profiles of B2: dormant states should have return probabilities that
fall as dormancy deepens, and types should differ in how readily they return. If all types
look alike, try more types (`hmm.b2_types`) or check that EM converged (`iters` below
`hmm.max_iters`). Also compare the types' hours (`h_mean` of the first active state): if the
types differ mainly in hours and hardly in return probability, EM has spent them on heavy
and light players. `hmm.b2_user_hours_level: true` (the default) exists to prevent that; it
models hours relative to each user's own level.

**EM reads one shard by default** (`hmm.train_shards: 1`, about 80,000 training users).
One EM pass of B2 cost about 1 ms per user on 4 cores in simulation, so a pass over the
whole 1M-user sample would take about 13 minutes and a full fit most of a day; and B2 was no
more accurate with 60,000 training users than with 2,000. To confirm that on real data, fit
once with `train_shards: 2` and compare the validation log-likelihood in the `tune` output.
Scoring always covers every shard.

## Troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| `Missing libraries` from `_setup` | Standard base environment | Switch to the ML base environment, or run the `%pip` cell |
| `did not score the same users` from `run_eval` | A model's scoring step ran against an older build, or only partly | Rerun that model's `score` / `final_*` cells after `10_build`; check `pipeline.check_same_users` |
| `does not look like the repo root` | Notebook path could not be resolved | Set the `repo_root` widget to the Git folder's `/Workspace/...` path |
| Error mentioning RDD, `sparkContext`, or `CACHE TABLE` | Spark Connect does not support it | The pipeline should not use these; note the line and fix it with a DataFrame or SQL equivalent |
| A cast or arithmetic error in SQL | ANSI mode: bad data that classic Spark would turn into NULL | Find the rows with the audit queries; fix the mapping or cleaning rule. Do not turn ANSI off |
| `CREATE VOLUME` or `CREATE SCHEMA` denied | Permissions | Ask for the rights, or point `artifact_path` and `schema` at ones you own |
| Clustering warning on write | `ALTER TABLE ... CLUSTER BY` not allowed | Harmless; set `cluster_tables: false` |
| `createDataFrame` fails on a large pandas frame | Spark Connect message size | Lower `synthetic.n_users`; real-data paths do not upload large frames |
| Out of memory in a tree step | Too many rows in memory | Lower `training.max_tabular_rows`, or use the 32 GB setting if allowed |
| Out of memory in an HMM or sequence step | Shard too large | Lower `training.seq_shard_users` and rerun `export_sequences` |
| A step never completes | Budget too small for the sample | Raise `budget.step_minutes` (max about 55), or shrink the sample |
| `mapInPandas` fails with `No module named ltv` | Project not installed for Python workers | Add the repo root as an Environment dependency, or use the driver path |
| Predictions for a model look duplicated or missing | A step was interrupted mid-write | Rerun the step: writes replace that model's rows atomically |

When you change behavior to get past a problem, write what and why in `docs/DECISIONS.md`
and update `docs/VERIFICATION.md`.
