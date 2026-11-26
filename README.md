# lifetime-value-modeling

An LTV model bake-off: two model families compared on identical data, splits and metrics,
to choose one for production.

| | Family | What it is |
| --- | --- | --- |
| **A** | Gradient-boosted trees | LightGBM on tabular features built in SQL |
| **C** | Sequence models | Small GRU and Transformer over each user's whole history, for CPU |
| **R** | Reference baselines | Segment lookup, persistence, buy-till-you-die (BG/BB) |

A third family, hidden Markov models (**B**), was cut on 2026-11-20 after a simulation
study and is parked: the code and tests remain, the run does not use them.

Each approach has its own notebook and is run on its own. All of them read the same
players, weeks and outcomes, and the evaluation checks that before comparing them. The run
ends with **example players ready to plot**: a player's weekly hours overlaid with each
model's weekly outputs (`notebooks/55_examples`).

Five targets per user per scoring week, over the next 52 weeks: P(re-engage), expected
active weeks, expected hours, P(spend > 0), expected spend. Data is weekly and sparse: a
user has a row only in weeks they played or spent.

A main goal is **long-term behavior**: recognising a player who lapses for months, returns,
and lapses again. Every model reads each user's whole history, and the evaluation reports a
segment for it.

## Status

**Built and tested on synthetic data. Not yet run on Databricks or on real data.**

`docs/VERIFICATION.md` says exactly what was executed, what was not, and where the first
failures are most likely. Read it before trusting anything here.

## Two passes

| Pass | Who | Where | Data |
| --- | --- | --- | --- |
| Build (done) | Claude Code | outside Databricks | synthetic only |
| Run (next) | Genie Code | Databricks serverless, this repo as a Git folder | real tables |

## Start here

| If you are | Read |
| --- | --- |
| An agent or person about to run it in Databricks | `AGENTS.md`, then `docs/RUNBOOK.md` |
| Trying to understand how and why it works | `docs/DESIGN.md` |
| Looking for a table or column | `docs/DATA_CONTRACT.md` |
| Asking why something was decided, or what is still open | `docs/DECISIONS.md` |
| Asking what has actually been run | `docs/VERIFICATION.md` |
| Curious about the synthetic data | `docs/SYNTHETIC_DATA.md` |
| After the literature review behind the candidate list | `reports/Modern LTV modeling approaches.md` |
| After what a 2M-user simulation said about speed, scale and likely outcomes | `reports/01_simulation_study.md` |
| Building plots of individual players before the real run exists | `docs/sample_examples/` (synthetic sample of the `55_examples` export) |
| Wanting to draw a player's history and watch each model respond | `docs/PLAYGROUND.md` (`python -m ltv.playground`, a local page) |

## Run pass in Databricks (summary)

1. Add this repo as a Git folder. Use serverless notebooks with the ML base environment.
2. Set the PLACEHOLDER values in `conf/bakeoff.yml` and `conf/synth.yml`.
3. Run `notebooks/05_synth_smoke_test` (widget `config = synth`). It proves the workspace
   can run everything and measures CPU throughput.
4. Run `notebooks/00_audit` on the real tables. **Stop for review.**
5. Then `10_build`, `15_baselines`, `20_trees`, `30_hmm`, `40_seq`, `50_eval`, `60_cost`.

No command may run over an hour, so long steps are time-boxed and resumable. A cell that
prints `complete: False` should simply be run again.

## Run the tests locally

Needs Python 3.10+ and Java 17+. Tests use local Spark with Delta Lake.

```bash
pip install -e ".[dev]"
```

```bash
pytest -m "not slow"
```

```bash
pytest -m slow
```

The slow test runs the whole pipeline end to end on 3,000 synthetic users, through the same
function (`ltv.pipeline.smoke_test`) that the Databricks smoke-test notebook calls.

## Layout

```
AGENTS.md, README.md, pyproject.toml
conf/            bakeoff.yml (every real name and setting), synth.yml (smoke-test overrides)
sql/audit/       Phase 0 checks, one SELECT per file
sql/build/       modeling tables, one SELECT per file, numbered in build order
src/ltv/         the package: data, synth, models, train, eval, score, pipeline
notebooks/       thin Databricks notebooks (.py source), one per step; parked/ holds the cut HMM
jobs/            Asset Bundle skeleton (unverified)
docs/            RUNBOOK, DATA_CONTRACT, DESIGN, DECISIONS, VERIFICATION, SYNTHETIC_DATA
genie/skills/    optional Genie Code skill (register the folder as an extra skill folder)
reports/         written by the run pass; also the literature review and the simulation study
research_notes/  notes behind the literature review
tests/           pytest suite
```

## Design in one paragraph

Input data is built by SQL from the sparse activity table, with no real names in code.
Models are trained at *origins* (scoring weeks) and scored on the 52 weeks that follow; no
training label window overlaps a scored window; each candidate is tuned on a validation
origin and then refit and scored once at two test origins. Calendar facts after an origin
come only from a calendar projected from what was known then. Every candidate writes one
prediction schema, passes through one shared recalibration layer and one evaluation, and
wins a target only with a significant paired difference at both test origins. Scoring
keeps a small per-user state instead of rereading history.
