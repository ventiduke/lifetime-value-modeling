---
name: ltv-bakeoff
description: Run or continue the LTV model bake-off in this repo (trees vs sequence models on weekly activity data). Use when asked to run a phase, a notebook or a model step of the bake-off, to interpret its audit, evaluation or cost reports, or to debug a step that failed or returned complete False.
---

# LTV bake-off: how to work in this repo

The repo's `AGENTS.md` holds the rules and is loaded automatically. This skill is the short
procedure for doing the work. Detail lives in `docs/RUNBOOK.md`.

## First, find out where things stand

1. Read `AGENTS.md`, then `docs/RUNBOOK.md`.
2. Look at `reports/`: which of `00_data_audit.md`, `03_bakeoff_results.md`, `02_cost.md` exist?
3. Check which models have predictions:

   ```sql
   SELECT model_id, fit, COUNT(*) FROM <catalog>.<schema>.predictions GROUP BY model_id, fit ORDER BY 1, 2
   ```

   Table names come from `conf/bakeoff.yml` (`catalog`, `schema`). Never guess them.
4. Check `docs/DECISIONS.md` for the open questions and anything decided since.

## Running a step

- Open the notebook for the phase (`notebooks/NN_name`). Set widgets `config` and `profile`.
- Run cells in order. Each cell is one step and stays under the time limit.
- **If a cell prints `complete: False`, run the same cell again.** It resumes from its
  checkpoint. This is normal, not a failure.
- After a phase, write a short results note in `reports/`.

## Stops you must respect

- After `00_audit`: stop and wait for the owner's sign-off on sample size, split schedule
  and known-calendar window.
- After the first model (A1) is evaluated end to end: stop for review.
- Never tune or change a model after looking at test-origin results in `50_eval`.
- Run one model notebook at a time. Each prints the shared inputs; `50_eval` stops if a
  model scored different players (`same_users` false). Rerun that model's scoring step.
- The HMM (`notebooks/parked/30_hmm`) is cut. Do not run it unless the owner asks.
- Never use results from `config = synth` to compare or choose models.

## When something fails

1. Read the error. Then read `docs/VERIFICATION.md`, section "Most likely first failures".
   The code has not run on Databricks before; platform differences are expected.
2. Check the troubleshooting table at the end of `docs/RUNBOOK.md`.
3. Fix the cause in the package (`src/ltv/`) or SQL (`sql/`), not in a notebook cell.
4. If you edited SQL features, also edit `src/ltv/data/np_features.py`: they are two
   implementations of the same features and must agree.
5. Record what you changed and why in `docs/DECISIONS.md`, and update `docs/VERIFICATION.md`.

## Ask the owner, don't guess

Real table or column names; what a row means; spend currency and refunds; the week
definition and time zone; which accounts to exclude; how far ahead season launches are
known. See `AGENTS.md`, "Ask, don't guess".

## Reading the results

- `reports/03_bakeoff_results.md`: "Winners by target" first. A model wins a target only if
  it leads at both test origins with a significant paired difference.
- The **lapse-and-return table** shows whether a model recognises players who habitually
  come back: compare each model's column with `actual` across the past-return groups.
- "Aggregate bias before recalibration" shows each model's own level error; headline numbers
  are after the shared recalibration layer.
- `55_examples` writes `example_weekly`: example players with their weekly history and each
  model's weekly outputs, for plotting. Use it to show where two models differ.
- `70_register` registers a saved fit in MLflow without retraining and checks it against `predictions`
  (`matches` must be true). It sets the alias `candidate`. Never set `champion` yourself: the owner does.
- `reports/02_cost.md`: weekly cost per family. Extrapolations are linear and need a full
  run to confirm.
