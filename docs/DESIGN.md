# Design: how the bake-off works and why

This is the reference for *how* each part works and *why* it was built that way. For the
steps to run it see `RUNBOOK.md`; for tables see `DATA_CONTRACT.md`; for the history of
decisions see `DECISIONS.md`; for what has actually been executed see `VERIFICATION.md`.

## 1. The question

Pick one production LTV model family by comparing candidates on identical data, splits and
metrics. The winner must be accurate, give coherent and usable outputs, and score every
user each week on serverless CPU.

**Five targets per user per scoring week**, over the next 52 weeks:

| Target | Column | Label |
| --- | --- | --- |
| P(re-engage): any active week | `p_reengage` | `any_active_52` |
| Expected active weeks | `exp_active_weeks` | `active_weeks_52` |
| Expected hours | `exp_hours` | `hours_52` |
| P(spend > 0) | `p_spend` | `any_spend_52` |
| Expected spend | `exp_spend` | `spend_52` |

**Long-term behavior is a main goal.** A player who lapses for about six months, returns,
and lapses again should be recognised as a returner. This shapes three things: every model
family reads each user's whole history (not a recent window); the features, the HMM and the
sequence inputs each represent lapse-and-return explicitly; and the evaluation reports a
segment for it.

## 2. Data layer

**SQL first.** Input data is built by SQL files in `sql/`, one `SELECT` per file. Python
(`ltv.data.sqlrun`) fills `{{placeholders}}` from the config and writes the result. Reasons:
the owner asked for SQL where possible; a plain `SELECT` runs unchanged on local Spark and on
Databricks serverless (Spark Connect); and analysts can read and adapt it.

**Weeks are integers.** `week_idx` 0 is the first calendar week with activity. All logic is
in week offsets, so there is no date arithmetic to get wrong.

**The source is sparse and stays sparse.** A user has rows only in active or spend weeks.
Features and labels are computed from those rows with conditional sums over week offsets.
The dense user x week panel is built for the sample only, as a convenience and a test
oracle; at full scale it would be tens of billions of rows that nothing needs.

**Two implementations of the features.** `sql/build/07_features_tabular.sql` (sparse) and
`src/ltv/data/np_features.py` (dense arrays). They exist for different users (the trees read
the table; the sequence models and scoring need features at arbitrary origins) and they
check each other: a test requires them to agree on every feature.

**One source for inputs and labels of sequence models.** `user_sequences` holds each user's
whole history as arrays. Loaders slice inputs (weeks `<= t`) and labels (weeks `t+1..t+52`)
from the same arrays, so they cannot drift apart. A test checks the sliced labels equal the
SQL `labels` table.

**Sampling.** The bake-off runs on a hashed user sample (`sample.fraction`), stable across
runs. Scoring at full scale is only needed for the cost test and the finalist.

## 3. Population, origins and splits

**Origin.** A scoring week `t`. Features use weeks `<= t`; labels use `t+1 .. t+52`.

**Population at an origin.** Users with at least one active or spend week in the 104 weeks
up to `t`, including users dormant now. By the owner's counts that is about 30M of 150M
users. The other 120M are the **long-lapsed tier**: scored by lookup (R1) unless a model
does better, and reported separately.

**Split schedule** (`ltv.data.splits.make_schedule`). With `last` = the last week index that
has data and `H` = 52:

| Step | Origin(s) | With 312 weeks (last = 311) |
| --- | --- | --- |
| `test_1` | `last - H` | 259 |
| `test_2` | `test_1 - 26` | 233 |
| `validation` | `test_2 - H` | 181 |
| `tune_train` | every 4 weeks, from week 52 to `validation - H` | 53 .. 129 |
| `final_train_1` | every 4 weeks, from week 52 to `test_1 - H` | .. 207 |
| `final_train_2` | every 4 weeks, from week 52 to `test_2 - H` | .. 181 |

Why this shape:

- **No training label window overlaps the window being scored.** The last training origin
  is always exactly 52 weeks before the origin it is evaluated on. Otherwise a model learns
  the test period's season launches from training labels.
- **Two test origins, 26 weeks apart**, so results do not rest on one season phase. A win
  must hold at both.
- **Three fits per candidate.** `tune` (scored at validation; all tuning and early stopping
  happen here), then `final_1` and `final_2`, which refit with frozen settings on the latest
  matured labels and score a test origin once. The final fits mirror production, where a
  model always trains on the newest labels available.
- **Users are split too**, by hash, 80/10/10. Training uses `train` users; early stopping
  and the calibration layer use `val` users at the validation origin; headline numbers are
  `test` users at the test origins. `train` users at the test origins are also reported:
  that is closer to production, where the same users are rescored.
- The schedule is computed from the data, so it moves forward as history grows.

**HMM exception, on purpose.** The HMM is generative and needs no labels, so its fit for an
origin `t` uses every week `<= t`, including the most recent year that supervised models
cannot use for labels. That is a real property of the model, not leakage: at scoring time
`t` all of that data exists.

## 4. Future-calendar rule

Seasons run about two months and vary, so launch dates a year ahead are probably not known.
A model may use only calendar facts known at the origin.
`ltv.data.splits.project_calendar` builds the calendar every model sees for weeks
`t+1 .. t+52`:

- inside the **known window** (`calendar.known_window`: `current_season`, or a number of
  weeks), the real calendar;
- after it, seasons repeat at the **median length of seasons completed by `t`**.

Reading real future launch dates from the calendar table at a past origin would leak the
future. A test checks that changing the calendar beyond the known window does not change
the projection. Phase 0 must find out how far ahead launches are really announced and set
`known_window`.

## 5. Features for long-term behavior

Besides recency, frequency, trend and seasonality, the tabular features describe lapses
directly (all from weeks `<= t`):

- how many past lapses of at least 4, 13 and 26 weeks ended in a return;
- the median and longest past lapse, and the **current lapse relative to the user's own
  median lapse** (a returner eight weeks into a usual twelve-week break looks different from
  a steady player eight weeks gone);
- number and typical length of play spells; weeks since the last return;
- the share of past returns that fell in the first two weeks of a season;
- activity in the 13 weeks after this date one and two years ago.

## 6. Candidates

Every candidate writes the same prediction schema (`ltv.models.base.Preds`).

### R. Reference baselines (no tuning)

- **R1 segment lookup**: training means by dormancy x tenure x prior spender x past returns,
  shrunk toward the dormancy-level mean in small cells. Also the default for the long-lapsed
  tier.
- **R2 persistence**: the next 52 weeks repeat the last 52.
- **R3 BG/BB**, discrete-time buy-till-you-die (Fader, Hardie, Shang 2010). Fitted to weekly
  activity and to weekly spend incidence; amounts from each user's own shrunk mean. Scored
  on its 52-week forecasts, never on P(alive). It is the classical benchmark, costs almost
  nothing to score, and is the HMM with two states, permanent churn and no calendar, so
  R3 vs B shows what returns and seasons add. Its closed forms are tested against simulation.

### A. Trees (`models/trees.py`)

LightGBM on `features_tabular`.

- **A1**: one model per target. Binary for the two probabilities; Poisson for active weeks;
  Tweedie for hours; spend as a hurdle (P(spend) x lognormal mean among spenders, with a
  smearing factor). **A1tw** replaces the hurdle with one Tweedie model for spend.
- **A2**: for each of five horizon buckets (weeks 1-4, 5-13, 14-26, 27-39, 40-52), models
  for active weeks, hours and spend. Totals are sums of buckets, so cumulative horizons
  never decrease. Quantiles come from a zero-inflated lognormal fitted in decomposed form.
- **Decomposed ZILN.** The ZILN loss is `BCE(p) + 1[y>0] * lognormal_nll(mu, sigma)`. With
  separate trees per output, `p` does not interact with `(mu, sigma)`, so `p` is exactly the
  binary model; `mu` is least squares on `log y` among positives; `log sigma^2` is fitted to
  the squared residuals. This is the stable form of the ZILN objective.
- **Tuning**: a time-boxed random search for one shared hyperparameter set, scored on three
  representative models. Each model's number of trees comes from its own early stopping in
  the `tune` fit and is frozen for the final fits.

### B. Hidden Markov models (`models/hmm.py`)

Each user is in one of K latent states each week. A state emits activity (Bernoulli), hours
given active (lognormal), spend incidence (Bernoulli) and spend amount (lognormal).
Transitions depend on the calendar bucket of the week being entered (launch week, season
week 2, weeks 3-4, later) and on nothing user-specific, so one set of matrices serves all
users.

- **B1** is a plain HMM.
- **B2 is lapse-aware**, built for the long-term-behavior goal:
  - *Dormancy is a chain of states* (lapsed recently, a while, long). A plain HMM is
    memoryless: it forgets how long a user has been away. The chain makes the chance of
    returning depend on the length of the lapse.
  - *States are split into user types that never mix* (block-diagonal transitions). The
    filter infers from the whole history which type a user is, so a player who has lapsed
    and returned before keeps a high return probability during the next lapse.
  - *Types are crossed with spend classes*, with tied parameters: transitions, activity and
    hours depend on type and role, not spend class; spend depends on spend class and role,
    not type. How a player engages and how much they spend are different axes. Without
    this, a first version under-predicted synthetic spend by half.
  - *Hours are modeled relative to each user's own level* (`hmm.b2_user_hours_level`). Each
    week's hours are divided by the user's level so far: the mean of their log-hours in
    earlier active weeks, shrunk toward the population's. Players differ persistently in how
    much they play, and explaining that is the largest likelihood gain on offer. Without
    the level, EM spent three of four types on heavy, medium and light players, the types'
    return behavior came out nearly identical, and B2 could not tell a serial returner from
    a churned player (`reports/01_simulation_study.md`). The level uses earlier weeks only,
    so nothing from the future is read, the likelihood stays comparable with a model that
    lacks it, and weekly scoring can carry it forward as two numbers per user.
  - B2 is still an HMM: B1 with some transitions forbidden. Fitting, forecasting and scoring
    are the same code.
- **Fitting is EM** (Baum-Welch), streamed over shards: one iteration is one pass that
  accumulates sufficient statistics, then closed-form updates. EM never decreases the
  likelihood, needs no learning rate, and is deterministic, so a fit stopped at the time box
  resumes exactly. (An earlier gradient-based fit was slower and stopped at weak solutions.)
- **Expected hours and spend use each state's arithmetic mean**, tracked beside the
  lognormal. A lognormal fitted on logs gets the mean wrong when a state mixes ordinary and
  very large spenders.
- **Forecasting is linear.** Filter to the state probabilities `alpha` at the origin, then
  push forward 52 weeks under the projected calendar. Every expectation is `alpha @ V`,
  where `V` depends only on the origin's calendar. P(re-engage) is 1 minus the probability
  of 52 inactive weeks in a row, also linear. So scoring is a matrix product over a per-user
  state table, and the outputs are coherent by construction.
- **Journey states.** Describe each state by what it does (`state_profile`): activity rate,
  mean hours, spend rate, and for dormant states the return probability in an ordinary week
  and in a launch week. Latent labels alone are weakly identified.

### C. Sequence models (`models/seq.py`, `data/seq_inputs.py`)

Small PyTorch models for CPU. Cost grows with time steps x hidden size squared, so the
designs keep the whole history but compress it.

- **C1, two-resolution GRU.** Everything up to about 26 weeks before the origin enters as
  4-week blocks on a *fixed* calendar grid; the remaining 26 to 29 weeks enter week by week.
  A 320-week history is about 100 steps. A six-month lapse is six or seven empty blocks:
  nothing is forgotten. A GRU reads the blocks; its final state starts a second GRU over the
  recent weeks. The linear layer on each flattened block is a temporal convolution with
  kernel and stride 4.
- **C1e, episodes.** History is cut into play spells (runs of active weeks with no gap of 4+
  weeks). Each spell is one token: length, activity, hours, spend, the lapse before it, where
  in the season it started, how long ago it ended. This states lapse-and-return directly,
  and cost scales with activity, not calendar time.
- **C2, small Transformer** over blocks plus recent weeks. About five times slower per step
  than C1 on CPU; the literature shows no consistent gain over GRUs here. Optional.
- **Static inputs**: the trees' features (default, because Bauer and Jannach found such
  features the largest contributor to their sequence model) and platform and country
  embeddings.
- **Heads** (multi-task): 52 weekly activity sigmoids, each seeing its own projected calendar
  position (their sum is expected active weeks); a separate any-active head, because weeks
  are not independent; an any-spend head; and for hours and spend a distribution head for
  the total given it is positive, plus a 5-way share head that splits the total across the
  horizon buckets. Expected value = P(positive) x head mean; cumulative horizons = total x
  cumulative shares, so they cannot decrease.
- **Distribution heads** (`models/heads.py`): **ZILN** (default), a 3-component lognormal
  **mixture**, and **ordinal** buckets. The alternatives are ablations (`C1mix`, `C1ord`),
  as is a high-spender auxiliary task (`C1whale`) and dropping the trees' features
  (`C1notab`). Keep one only if it lowers aggregate or top-1% bias on validation without
  losing Gini. *Known behavior*: ZILN's mean is `exp(mu + sigma^2/2)`, very sensitive to
  sigma; expect it to rank well but over- or under-shoot the level. The shared calibration
  layer corrects the level; the ablations test whether a different head needs less correction.
- **Training** (`train/seq_fit.py`) counts steps, not epochs. The batch for step `s` depends
  only on `(seed, s)`, so a resumed run follows the same path. The `tune` fit early-stops on
  validation; final fits run exactly the number of steps it chose.
- **Self-supervised pretraining was dropped**: it does not fit a CPU budget.

### Stack (diagnostic, not a candidate)

`pipeline.run_stack`: a second-level LightGBM on the features plus the predictions of one
tree model and one sequence model, fitted on `val` users at the validation origin (where
both sets of predictions are out of sample) and scored at the test origins as `STACK`.
Bauer and Jannach's stack cut RMSE by about 4% against their best single model. A large gain
means each family misses signal the other has. It is not a candidate: a production stack
means maintaining two models, and its calibrated numbers are slightly optimistic because the
calibration layer is fitted on the users it trained on.

## 7. Time budget and resumability

No command may run over an hour, so every long step is time-boxed (`budget.step_minutes`)
and resumable from a checkpoint under `artifact_path`:

| Step | Unit of progress | Resume is |
| --- | --- | --- |
| Tree search | one trial, logged to JSON lines | exact: trial `i` always draws the same configuration |
| HMM fit | one EM iteration | exact: EM is deterministic |
| Sequence fit | one optimiser step; checkpoint at each validation check | the same path (batches and dropout follow the step number) |

A step that runs out of time returns `complete: False`. Rerun the cell.

**Run tracking** (`ltv/tracking.py`). Every `run_*` step is logged with its parameters,
numeric results and wall-clock time: to `run_log.jsonl` under the artifact path, and to
MLflow when it is importable. DBUs cannot be read inside a run; join job and run IDs to
`system.billing.usage` afterwards.

**Files on volumes** (`ltv/io.py`). Unity Catalog volumes reject random-access and append
writes, which zip-based formats (`np.savez`, `torch.save`) and append-mode logs need. Every
checkpoint, shard and log is written to local disk first and copied over in one pass. Two budget profiles
(`budget.profile`): `quick` to shake out the pipeline, `full` for numbers that count. Every
family gets the same budget.

## 8. Evaluation

**One evaluation for all models** (`eval/evaluate.py`), reading only the predictions table.

**Metrics** (`eval/metrics.py`). Primary metric per target:

| Target | Primary | Also reported |
| --- | --- | --- |
| `p_reengage`, `p_spend` | log loss | AUC, expected calibration error |
| `exp_active_weeks` | Poisson deviance | aggregate bias, MAE |
| `exp_hours` | Tweedie deviance (power 1.5) | normalized Gini, top-1% and top-5% capture, bias |
| `exp_spend` | normalized Gini | Tweedie deviance, top-k capture, bias, Gini among positives |

Level metrics (deviance, bias) and rank metrics (AUC, Gini, capture) are both always
reported: a model can be good at one and poor at the other. **Tail checks**: bias in the top
1% and 0.1% by prediction, and bias by band of the actual value. MAE is never used to rank:
on a mostly-zero target it rewards predicting zero.

**Segments** (`eval/segments.py`): current dormancy, tenure, prior spender, platform, top
countries, and **lapse-and-return history** (0, 1, 2+ past returns from a 26-week lapse).
The returner x dormancy cut is a headline table: it shows whether a model recognises players
who habitually come back.

**Shared post-hoc layer** (`eval/calibration.py`), identical for every candidate:

- *Recalibration*: a monotone map from prediction to outcome, per target, fitted on `val`
  users at the validation origin (from the `tune` fit) and applied to the test predictions.
  It fixes levels without reordering users. The map is isotonic regression on up to 50
  equal-count bins of validation users, not on single users: probabilities are smoothed so
  the map never returns exactly 0 or 1, amounts above the validation range are scaled in
  proportion, and amounts below it take the bottom bin's value. Plain isotonic regression
  failed at 2M simulated users (`reports/01_simulation_study.md`, section 5): its end steps
  rest on a handful of users, and the map is applied to a different fit from the one it was
  fitted on, so a model with lumpy predictions put thousands of users on those steps.
- *Intervals*: split conformal by dormancy segment on the ratio outcome / prediction.
  Coverage is reported overall and by segment, because the calibration cohort is a year
  older than the scored cohort and zero-heavy segments are hard.

Both raw and calibrated metrics are stored; headline numbers are calibrated.

**Coherence checks**, counted per model: expected active weeks in [0, 52];
`P(re-engage) <= E[active weeks] <= 52 P(re-engage)`; P(re-engage) at least the largest
weekly P(active) and at most their sum; cumulative horizons never decrease and end at the
total. P(spend) may exceed P(re-engage), because spend-only weeks exist.

**Deciding a win.** For each target, models are ranked on the primary metric among `test`
users at each test origin. The leader wins only if the 95% bootstrap interval of its *paired*
difference from the runner-up excludes zero at **both** test origins. The runner-up is the
best model of **another family**: variants of one candidate can share a head outright (A1,
A1tw and A2 use the same P(re-engage) model), and a tie between siblings says nothing about
which family wins. The bootstrap resamples users and uses the same resampled users for both
models; comparing two separate intervals would call real differences ties.

**Forecast revisions** (`eval/revisions.py`), low priority, finalist only. A forecast of a
fixed quantity should be revised by zero on average. The check holds the label window fixed
and compares `actual so far + new forecast of the rest - old forecast`. Segments must be
defined at `t`, never by what the user did afterwards: a quiet week is bad news, so "users
with an ordinary week" have a negative mean revision even under a perfect model.

## 9. Scoring and cost

Requirement: weekly scoring of all users on serverless CPU, no command over an hour, batching
allowed. Total time and cost per weekly run is the main tiebreaker after accuracy. **Measure,
do not estimate** (`pipeline.run_cost`, `notebooks/60_cost`).

No model should reread a user's whole history each week:

| Family | Weekly state | Weekly work |
| --- | --- | --- |
| Trees | none | rebuild features in SQL from the sparse table, then score |
| HMM | K-vector per user, plus two numbers for the hours level | one matrix-vector step per user (`kernels_np.hmm_level_update`, `hmm_step`), then `alpha @ V` and the level applied to hours. A run of weeks with no rows collapses to one cached matrix per (start week, run length) |
| GRU (C1) | block-level hidden state | advance the state when a 4-week block completes; rerun only the recent weeks |

`models/kernels_np.py` implements the HMM and GRU steps in NumPy, so they can run inside a
Spark pandas UDF (1 GiB sandbox on serverless) without PyTorch. Tests check them against the
models. The HMM state table (`hmm_user_state`) is written by the final fits; the weekly
update is `score.batch.hmm_weekly_update`. The GRU's carried-state weekly job is designed and
its key property is tested, but the job itself is not built.

The **long-lapsed tier** (about 120M users) is scored by the R1 lookup unless the evaluation
shows a model does better for users lapsed over N weeks.

## 10. What the literature review changed

No new candidate family was added (`reports/Modern LTV modeling approaches.md`). It added
baseline R3, cumulative horizons for hours and spend, the head ablations, the shared
calibration layer, the tail checks, and two test origins. Survival and hazard models reduce
to the trees on person-week rows or to the HMM; neural point processes, time-series and
tabular foundation models and per-user trajectory simulation either fold into an existing
candidate or cannot meet the scoring budget.
