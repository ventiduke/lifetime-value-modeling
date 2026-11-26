# Instructions for agents working in this repo

This file is for any coding agent (Genie Code, Claude Code, others) and for people. Genie
Code loads it automatically when a notebook or file in this Git folder is open. Keep it
under 20,000 characters: Genie Code ignores anything past that.

## What this repo is

An LTV model bake-off. It compares two model families on identical data, splits and
metrics, to choose one for production:

- **A** gradient-boosted trees on tabular features,
- **C** small sequence models (GRU, Transformer),

against three reference baselines (**R1** segment lookup, **R2** persistence, **R3**
buy-till-you-die). A third family, **B** (hidden Markov models), was cut by the owner on
2026-11-20 after a simulation study; its code is parked (`notebooks/parked/`,
`hmm.enabled: false`) and is not part of the run. Five targets per user per scoring week, over the next 52 weeks:
P(re-engage), expected active weeks, expected hours, P(spend > 0), expected spend.

A main goal is **long-term behavior**: a player who lapses for six months, returns, and
lapses again must be recognised as a returner, not written off as churned. Every model sees
each user's whole history, and the evaluation has a segment for it.

The plan of record is the "Plan v2" tab of the planning document, mirrored in
`docs/DESIGN.md` and `docs/DECISIONS.md`.

## Two passes

| Pass | Who | Where | Data | State |
| --- | --- | --- | --- | --- |
| Build | Claude Code | outside Databricks | synthetic only | done: see `docs/VERIFICATION.md` |
| Run | Genie Code | Databricks serverless, this repo as a Git folder | real tables | **you are probably here** |

The code has been executed end to end on synthetic data with local Spark and Delta. It has
**never run on Databricks or on real data**. `docs/VERIFICATION.md` lists exactly what was
and was not executed, and where it is most likely to break first. Read it before debugging.

## Rules that are never broken

1. **Phase 0 comes first and ends with a stop.** Run the audit (`notebooks/00_audit`), write
   the report, propose settings, and wait for the owner (Alex) to sign off before modeling.
2. **Never tune on the test origins.** All tuning and early stopping use the validation
   origin. The two test origins are scored once, for final numbers. Do not go back and
   change a model after looking at `notebooks/50_eval`.
3. **Synthetic data is a test fixture only.** Never use synthetic results to compare, tune
   or choose models. A generated world rewards the model that resembles its generator.
   `reports/01_simulation_study.md` is a scale and speed check, not a result.
4. **No real names in SQL or Python.** Catalog, schema, table and column names live only in
   `conf/bakeoff.yml`. SQL files use `{{placeholders}}`.
5. **No future information.** Features read weeks at or before the origin. Calendar facts
   after the origin come only from the *projected* calendar (`ltv.data.splits.project_calendar`).
   Reading real future launch dates from the calendar table at a past origin is leakage.
6. **No command runs over 1 hour** (training may take 2 if it must). Long steps are
   time-boxed and resumable. A step that returns `complete: False` is not an error: **run
   the same cell again** and it continues from its checkpoint.
7. **Every candidate gets the same treatment**: the same users, origins, splits, time
   budget, projected calendar and post-hoc calibration layer. Each approach has its own
   notebook, run one at a time; all of them read the tables built by `10_build`. Every
   model notebook prints the shared inputs (`pipeline.describe_inputs`), and `50_eval`
   refuses to compare a model that scored different players. Do not work around that.
8. **Do not install PySpark** on serverless (it breaks the session), and do not add `dev`
   extras there.
9. **The SQL features and `src/ltv/data/np_features.py` are two implementations of the same
   thing.** If you change one, change the other and run `tests/test_data_layer.py`.

## Ask, don't guess

Stop and ask the owner when any of these is unclear:

- real table or column names, or which catalog and schema to write to;
- what a row means (is `hours > 0` the definition of active? can spend be negative?);
- spend currency, refunds, chargebacks;
- the week definition and time zone of `week_start`;
- which accounts are bots, test, banned or merged;
- how far ahead season launch dates are known (`calendar.known_window`);
- anything that would change a decision recorded in `docs/DECISIONS.md`.

Open questions with their current defaults are at the end of `docs/DECISIONS.md`.

## Compute constraints (from the owner)

Databricks **serverless**, environment version 6, **16 GB**, **no accelerator** (a serverless
GPU option exists but is not assumed). That means Spark Connect only: no RDDs, no
`CACHE TABLE`, ANSI SQL on, UTC session time zone, PySpark UDFs limited to 1 GiB each.
SQL is the priority for building input data; training and scoring use Python.

## Order of work (run pass)

| # | Notebook | Purpose | Gate |
| --- | --- | --- | --- |
| 1 | `05_synth_smoke_test` | Prove the workspace can run everything, on synthetic data; measure CPU throughput | fix environment problems here, not on real data |
| 2 | `00_audit` | Phase 0 checks on the real tables; `reports/00_data_audit.md` | **STOP for sign-off** |
| 3 | `10_build` | Splits, features, labels, sequences | check row counts and base rates |
| 4 | `15_baselines` | R1, R2, R3 | |
| 5 | `20_trees` | Candidate A: tune, fit, two final fits | **STOP after A1 is scored end to end** (run `50_eval` once) |
| 6 | `40_seq` | Candidate C, one variant at a time; learning curve | |
| 7 | `50_eval` | One evaluation for all models; `reports/03_bakeoff_results.md` | final numbers only |
| 8 | `55_examples` | Example players for plotting: weekly history against each model's weekly outputs | |
| 9 | `60_cost` | Timings per scoring path; `reports/02_cost.md` | |

Each notebook starts with `%run ./_setup` and has widgets: `config` (`bakeoff` or `synth`)
and `profile` (`quick` to shake out the pipeline, `full` for numbers that count).
`docs/RUNBOOK.md` has the detail for every step, including what to do when one fails.

## Where things are

```
conf/bakeoff.yml      every real name and setting; synth.yml overrides for the smoke test
sql/audit/            Phase 0 checks, one SELECT per file
sql/build/            the modeling tables, one SELECT per file, numbered in build order
src/ltv/pipeline.py   the functions the notebooks call (build_*, run_*, smoke_test)
src/ltv/data/         SQL runner, split schedule, future calendar, sequence store, NumPy features
src/ltv/models/       baselines, bgbb, trees, hmm, seq, heads, kernels_np (NumPy scoring kernels)
src/ltv/train/        time-boxed, resumable search and training loops
src/ltv/eval/         metrics, segments, calibration, evaluate, revisions, examples (players to plot)
src/ltv/score/        scoring paths and the timing harness
src/ltv/tracking.py   run log (run_log.jsonl under the artifact path, and MLflow when available)
src/ltv/io.py         file writes that are safe on Unity Catalog volumes
src/ltv/synth/        synthetic data generators with known processes (two worlds)
src/ltv/sim.py        Spark-free bake-off on synthetic shards (the simulation study)
src/ltv/playground/   local web page: draw or simulate one player, see each model's forecast (docs/PLAYGROUND.md)
notebooks/            thin Databricks notebooks (.py source format); parked/ holds the cut HMM
docs/                 RUNBOOK, DATA_CONTRACT, DESIGN, DECISIONS, VERIFICATION, SYNTHETIC_DATA, PLAYGROUND
reports/              written by the run pass; also the literature review and simulation study
tests/                pytest suite; runs on local Spark with Delta
```

Notebooks are thin on purpose: they call `ltv.pipeline`, which is what the tests exercise.
Put logic in the package, not in notebook cells.

## Conventions

- **SQL files** are a single `SELECT` with a header comment (purpose, grain, inputs, output,
  notes). Writing the result is done by `ltv.data.sqlrun`. Keep it that way: it is why the
  same SQL runs in tests and in Databricks.
- **Weeks are integers.** `week_idx` 0 is the first week of data. An *origin* `t` is a
  scoring week: features use weeks `<= t`, labels use `t+1 .. t+52`.
- **"Active" means `hours > 0`.** Spend-only weeks count toward spend, not activity.
- **Splits.** Users are hashed 80/10/10 into `train` / `val` / `test`. Each candidate has
  three fits: `tune` (scored at the validation origin), `final_1` and `final_2` (frozen
  settings, scored at the two test origins).
- **Predictions** all go to one table with one schema (`docs/DATA_CONTRACT.md`). A new
  model only has to produce a `ltv.models.base.Preds`.
- **Reports** are Markdown in `reports/`. End each phase with a short results note there.

## Running the tests

Locally, with Java 17 or newer:

```bash
pip install -e ".[dev]"
```

```bash
pytest -m "not slow"
```

```bash
pytest -m slow
```

The slow test runs the whole pipeline (`pipeline.smoke_test`) on 3,000 synthetic users. In
Databricks, `notebooks/05_synth_smoke_test` runs the same function.

## Definition of done, per phase

- **Phase 0**: `reports/00_data_audit.md` exists, every FAIL is resolved or escalated,
  proposed settings are written down, owner has signed off.
- **Phase 1**: all tables in `docs/DATA_CONTRACT.md` exist; row counts and base rates look
  sane across origins; the leakage test passes if any SQL was changed.
- **Phase 2**: every candidate has rows in `predictions` for `tune`, `final_1` and
  `final_2`; each step reported `complete: True`; `same_users` is true for every model;
  the sequence learning curve is recorded.
- **Phase 3**: `reports/03_bakeoff_results.md` and `reports/02_cost.md` exist; the tables
  `example_weekly` and `example_users` exist; the recommendation names a winner or says
  plainly that there is none, with the runner-up. If trees and the GRU are close, say so:
  the owner leans toward the GRU in a tie, because new weekly signals are easier to add.

## If something breaks on Databricks

1. Read `docs/VERIFICATION.md`, "Most likely first failures".
2. Do not work around a failure by weakening a rule above. Record what failed and why in
   `docs/DECISIONS.md` (decision log) and tell the owner.
3. If you change behavior, add or update a test, and update `docs/VERIFICATION.md`.
