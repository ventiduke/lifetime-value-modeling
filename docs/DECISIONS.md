# Decision log

Why the bake-off is built the way it is. Newest sections last. When you change a decision,
add an entry: do not rewrite history. Open questions are at the end.

The plan of record is the "Plan v2" tab of the planning document (2026-11-18). This file
and `DESIGN.md` are its counterpart in the repo.

## 1. Decisions from the owner (Alex)

| Topic | Decision |
| --- | --- |
| Who is scored | Every user, every week. About 150M users all-time; about 30M active in the last two years. Long-lapsed users may move to a cheaper tier if accuracy loss is small. |
| Spend | Total spend only; no breakdown by type. |
| Compute | Databricks serverless, environment version 6, 16 GB, no accelerator. A serverless GPU option exists; assume CPU for now. |
| Run time | No command over 1 hour. Training may take 2 hours if it must (not ideal for testing). Scoring can be batched. |
| History | Complete weekly histories for all users, over 6 years (312+ weeks) and growing. Data begins at product launch. |
| Seasons | About 2 months long, with variation. |
| Product stability | Very consistent. Train on all history. |
| Incumbent | No tree model is in production; A1 is a fresh build. |
| Names | Catalog, schema, table and column names are placeholders for now. |
| Languages | SQL is the priority for building input data. Model training and scoring can use Python, pandas, Arrow. |
| Lift measurement | Low priority for now; there is no control group. |
| **Long-term behavior** | **A main thing to capture: for example a player who commonly lapses for 6 months, returns, then lapses again.** |

### 1.1 Decisions from the owner after the simulation study (2026-11-20)

| Topic | Decision |
| --- | --- |
| **Candidates** | **The HMM (candidate B) is cut. The bake-off continues with trees (A) and sequence models (C).** It was the least predictable family in simulation. The code, tests and notebook are parked (`notebooks/parked/30_hmm.py`, `hmm.enabled: false`), not deleted. |
| If trees and the GRU are close | Lean toward the GRU: it reads weekly sequences directly, so further weekly signals (social levels, playlist hours and the like) can be added as input columns without hand-building features for each. |
| How the run is organised | One approach per notebook, run one at a time. Every model reads the same raw source (the same players' weekly histories), the same splits, origins and outcomes. The features each builds from them may differ. |
| Example players | The results must include player-level examples that are easy to plot: a player's weekly hours history overlaid with each model's weekly outputs, by user id, to compare models by eye. |
| A third simulated world built on a decaying "engagement" state | Not for now. |

## 2. Decisions from the plan review (2026-11-18)

| Decision | Why | Instead of |
| --- | --- | --- |
| Split schedule computed from history length; two test origins; final refits | With the last training origin 52 weeks before the scored origin, no training label window overlaps it. Two origins in different season phases guard against one launch deciding the result. Final refits mirror production | One test origin; training origins "across the available history", whose label windows overlapped the test window |
| Fixed-window revision check, segments defined at `t` | "Users with an ordinary week" selects on the outcome; a quiet week is bad news, so a correct model would fail the original check | Mean change in the rolling 52-week forecast among users with no event |
| Paired bootstrap on the difference | Both models are scored on the same users; comparing two separate intervals calls real differences ties | "Wins if its CI separates from the runner-up's" |
| HMM dormant-run shortcut is a product of calendar-week matrices | "Matrix to the power k" assumes transitions never change, but they depend on the season week | Cached matrix powers |
| GRU scoring carries a hidden state | Hidden states differ per user, so no lookup table can replace them | A "precomputed decay table" |
| Build pass and run pass | The build has no data access; the run is done by an agent with no other context | One executor inside Databricks |
| Dense panel for the sample only | 150M users x 300+ weeks is tens of billions of rows that nothing needs; scoring reads the sparse table | A full-scale dense panel |
| One `user_sequences` table feeds the HMM and both sequence models | One source for inputs and labels; the HMM had no defined input | Per-origin sequence tensors |
| Reference baselines R1 and R2 | Without a floor nobody can say what a model buys | None |
| Synthetic data as a test fixture | Without data, nothing could be executed before handoff; the first real run would be the debugger | Handing over unexecuted code |

## 3. Decisions from the literature review

Full report: `reports/Modern LTV modeling approaches.md`.

| Decision | Evidence |
| --- | --- |
| **No new candidate family** | Every other family examined reduces to A, B or C, or cannot meet the scoring budget |
| Add R3, discrete-time BTYD (BG/BB) | The classical benchmark; other BTYD variants lose clearly in app and game data; cheap to score; nests inside the HMM |
| Hours and spend at cumulative horizons 4/13/26/39/52 | One industrial paper on accuracy; also gives shorter horizons to evaluate and revise |
| Head ablations against ZILN (mixture, ordinal, high-spender task) | Several papers: ZILN ranks well but is biased in level and tail. None wins consistently, so ZILN stays the default |
| One shared recalibration and interval layer | Recalibration evidence moderate; conformal on LTV untested in the literature |
| Tail calibration (top 1%, 0.1%), bias by value band, Gini on positives | Strong evidence of top-end under-prediction in game LTV papers |
| The trees' features as static inputs to the sequence models, by default | Bauer and Jannach 2021 (read in full): their largest ablation effect |
| Stack of trees and sequence model as a diagnostic | Bauer and Jannach: about 4% lower RMSE, 8% lower MAE than the best single model |
| Not adopted: survival models, point processes, foundation models, trajectory simulation, graph and LLM methods | They fold into A or B, target new users, or fail the budget |

## 4. Decisions that follow from the owner's answers

| Answer | Decision |
| --- | --- |
| Serverless, CPU, 16 GB | Sequence models are small and read compressed full histories; pretraining dropped; Transformer small and optional; tabular training rows capped; sequences read shard by shard |
| No command over 1 hour | Every fit and search is time-boxed and resumable; each family gets the same budget; two profiles, `quick` and `full` |
| Scoring can be batched | The original 30-minute knockout is treated as replaced by "no command over 1 hour"; total time stays the main tiebreaker (open question 1) |
| 150M users, 30M recently active | The 120M long-lapsed users are an explicit tier, scored by lookup unless a model does better |
| Seasons about 2 months, varying | The future-calendar rule: only launches known at the origin; a projected calendar after that |
| Data begins at launch | No left-censoring flag; the audit only confirms the start |
| Product consistent | No test of older against newer training origins |
| Lift low priority | Lift protocol dropped; the revision check is built and run for the finalist only |
| Long-term behavior matters | Full-history inputs for every family; lapse-and-return features; the lapse-aware HMM (B2); episode tokens (C1e); a returner segment in the evaluation |

## 5. Decisions made during the build

These were not in the plan. Each was forced or suggested by something found while building.

| Decision | Why | What the plan said |
| --- | --- | --- |
| **HMM is fitted by EM**, not by Adam on the likelihood | The gradient fit took 25 s per pass on 14,500 users and stopped at solutions that barely separated returners. EM converged in about 20 passes, never decreases the likelihood, needs no learning rate and resumes exactly | "maximizing the marginal likelihood with Adam" |
| **B2 crosses user types with spend classes**, parameters tied | With types only, the type blocks were spent on return behavior and synthetic spend was under-predicted by half. Engagement and spend are different axes. Tying keeps the parameter count small | User types only |
| HMM expectations use each state's **arithmetic mean** | A lognormal fitted on logs gets the mean wrong when a state mixes ordinary and very large spenders | Lognormal means |
| HMM tenure covariate dropped | The filter runs from each user's first week, where tenure is zero for everyone, so an initial-state covariate on tenure carries no information | Tenure in the initial-state distribution |
| **ZILN for the trees is fitted in decomposed form** | With separate trees per output the loss separates; a joint custom objective is fiddly and unstable for sigma | Custom ZILN objective, with this as the fallback |
| A2 uses five buckets matching the cumulative horizons | So every model reports the same horizons (4, 13, 26, 39, 52) | Four quarterly buckets |
| **Tree tuning is a seeded random search with a JSON-lines log**, not Optuna | It resumes across commands with no database (SQLite on a volume can misbehave) and no extra dependency | Optuna, about 50 trials |
| One shared hyperparameter set for all tree models | A search per model (24 of them) does not fit the budget; tree counts are still per model | Tune each model |
| Sequence heads: total from the distribution head, **horizon split from a share head** | Guarantees cumulative horizons never decrease and end at the total | Non-negative increments |
| Sequence blocks sit on a **fixed calendar grid** | A block's features never change once complete, so the encoder state can be carried forward | Blocks relative to the origin |
| Label arrays stored as plain columns in training frames | pandas operations misalign or reject array attributes | - |
| Local tests run on Delta Lake | Delta installs locally, so table writes, clustering and partition-scoped overwrites are exercised | Plain Spark tables |
| Predictions are replaced per model with Delta `replaceWhere` | Dynamic partition overwrite did not replace per partition through `insertInto` | Partition overwrite |
| `sql/` installs as a data package (`ltv_sql`) | So SQL is found whether the project is pip-installed or run from a checkout | - |
| R3 fitted on one row per user (latest origin) | Repeated origins of a user are not independent | - |
| Stack diagnostic is fitted on validation users | Both models' predictions are out of sample there | - |

### 5.1 Decisions from the simulation study (2026-11-20)

`reports/01_simulation_study.md` ran the bake-off on 2 million synthetic users in each of
two worlds. These changes came out of it. None was made to help a model win on synthetic
data; each fixes something that failed, was wrong, or could not finish at scale.

| Decision | Why | Before |
| --- | --- | --- |
| **The calibration layer is a binned monotone map** (`eval/calibration.MonotoneMap`): isotonic regression on at most 50 equal-count bins, smoothed probabilities, proportional extrapolation of amounts above the validation range | Isotonic regression on single users ends in steps fitted on a few users, and clips everything outside the validation range onto them. Applied to another fit, it set P(spend) to exactly 0 for 12,700 users of B1 and raised B2's hours by half | scikit-learn isotonic regression per user, clipped |
| **The win rule compares the leader with the best model of another family** (`evaluate.family`) | A1, A1tw and A2 share one P(re-engage) model and tie exactly, so comparing with the runner-up could never name a winner | Leader against the runner-up |
| **HMM EM trains on `hmm.train_shards` shards** (default 1); scoring and the validation likelihood still use all shards | One EM pass of B2 over 800,000 users is about 13 minutes on 4 cores; 100 passes do not fit any budget. The learning curve was flat from 2,000 to 59,000 users | EM over every shard |
| **Tree `fit` and `final_*` save after each of the 24 models and resume** (`partial_<step>.pkl`) | At 5M rows the step can exceed one command, and it saved nothing until the end | One uninterruptible step |
| **Optional learning-rate drops for the sequence models** (`seq.lr_drops`, default 0). The tune fit records the steps; final fits replay them | Early stopping at a constant rate stopped the GRU after 3,000 steps. Off by default: turning it on is a tuning choice for the real run, made on the validation origin | Constant rate after warm-up |
| Evaluation loads only `evaluation.splits` (`val`, `test`); predictions are written one model at a time | Memory at 16 GB with a 1M sample | Whole step in memory; every split loaded |
| R3 samples users before counting history patterns | Weighted sampling of patterns crashed when one pattern dominated | Sample patterns by weight |
| **B2 models hours relative to each user's own level** (`hmm.b2_user_hours_level`, default on). Each week's hours are divided by the user's level from earlier weeks; `hmm_user_state` carries `level_sum` and `level_n` | At 2M users EM spent three of B2's four types on heavy, medium and light players. The types' return behavior was nearly identical, and B2 could not tell a serial returner from a churned player, even in the world generated by a process of its own shape. This is the same failure that led to the spend classes, on another axis. **The owner should know**: it changes the candidate; set the flag to false to get the earlier B2 | Hours emitted by type and state only |
| Reference models take no part in the win rule (`evaluate(references=...)`) | The simulation's ORACLE knows the hidden state and was named the winner | Every scored model ranked |
| `evaluate(calibration="same_fit")` exists and is used by `ltv.sim` | To measure open question 8. The Databricks pipeline still uses the plan of record (`tune_fit`) | - |
| A Spark-free driver (`ltv.sim`) and a second, non-Markov synthetic world (`ltv.synth.worlds`) | Local Spark cannot build 2M users in reasonable time, and the first generator has B2's structure | One generator, pipeline only |

Not built, and why:

| Item | Status |
| --- | --- |
| Weekly carried-state scoring job for the GRU | The property it relies on is tested (`test_block_state_can_be_carried_forward`) and the kernel exists; the job is not built. Scoring recomputes from history |
| HMM quantiles | Not produced; the shared conformal layer gives intervals |
| Transformer on episode tokens | C2 runs on blocks plus recent weeks only |
| Revision check wired into a notebook | The functions and tests exist (`eval/revisions.py`); not wired, as lift is low priority |
| Asset Bundle jobs | A skeleton only (`jobs/databricks.yml`), unverified |
| Usability scores (1 to 5) | A human judgement for the final report; the inputs (coherence, coverage, state profiles) are produced |

### 5.2 Built after the owner's decisions of 2026-11-20

| Decision | Why |
| --- | --- |
| The HMM is **parked, not deleted**: `hmm.enabled: false`, its notebook moved to `notebooks/parked/`, and the smoke test, job bundle and run order no longer include it. `ltv.models.hmm`, its tests and `run_hmm` remain, and the end-to-end test still runs them once | Cutting a candidate should not cost the tested code if it is wanted again. `ltv.sim` and the simulation report still use it |
| **Like-for-like is checked, not assumed.** `pipeline.describe_inputs` prints the shared players per origin and split with a fingerprint of their ids at the top of every model notebook; `pipeline.check_same_users` compares each model's prediction rows with it; `run_eval` refuses to compare a model that scored different players (`evaluation.require_same_users`) | The owner wants each approach run separately on the same inputs and test data. Separate notebooks make it possible to drift apart without noticing |
| **Example players** (`ltv.eval.examples`, `pipeline.build_examples`, `notebooks/55_examples`): tables `example_users` and `example_weekly`, also as CSV. Three views per player: actual weeks; each model's forecast path from the test origin; and each model's opinion as of the end of every week in a rolling window | The owner's request. The rolling view needs each model scored at weeks that are not in the feature tables, so trees are scored on features rebuilt from the history by the NumPy implementation (tested equal to the SQL) |
| Example outputs are shown after the shared calibration layer (`examples.calibrated`) | A model's raw level bias would otherwise dominate the picture; the evaluation compares calibrated outputs too |
| **A playground** (`docs/PLAYGROUND.md`): a page where one player's weekly history is drawn or simulated and every saved model is applied to it. Two forms: local (`python -m ltv.playground`, Python does the scoring, any bundle) and a static site in `docs/playground` for GitHub Pages (JavaScript does the scoring, works on a phone) | The owner asked for both. The local one runs the pipeline's own code and is the only one that may load real fits; a run's artifact folder is a bundle as it stands. The static one needed the features and model inference rewritten in JavaScript (`engine.js`), checked against the Python by a test. **It carries synthetic demo models only: a Pages site is public, and the export refuses a bundle not marked synthetic** |
| **A registered MLflow model per fit, without retraining** (`ltv.serve`, `pipeline.register_model`, `notebooks/70_register`). One contract for every family: rows of `user_sequences` in, five outputs out. The model carries its calibration layer, the calendar and the package code. Registration ends by checking the model against the `predictions` table | The owner uses MLflow on Databricks and does not want to retrain the winner. A single contract means the caller does not change with the winner. The check makes "this is the evaluated model" a tested fact |
| The registered model ships the **evaluation's** calibration layer by default (`calibration="tune_fit"`); `same_fit` is an option | Served numbers then equal evaluated numbers, which is the plan of record. Open question 8 still stands: `same_fit` is what production would normally do |
| The champion alias is never set by code | Choosing the winner is the owner's decision. The pipeline sets `candidate` only |
| A1's weekly path is a flat line | A1 predicts 52-week totals only. A2 and the sequence models give a shape over time |

## 6. Open questions

Defaults are what the code does today. Change the config or code when answered, and record
the answer here.

| # | Question | Why it matters | Default in the code |
| --- | --- | --- | --- |
| 1 | Does "no command over 1 hour, batching allowed" replace the 30-minute limit for the whole weekly scoring run? | Decides whether scoring time can knock a model out or only break ties | Yes: total time is reported and breaks ties |
| 2 | How far ahead are season launch dates known when scoring runs? | Using dates not yet announced at a past origin is leakage | `calendar.known_window: current_season`; later launches projected at the median season length |
| 3 | May notebooks use the 32 GB high-memory setting? | Doubles the tabular training rows held in memory | No: sized for 16 GB |
| 4 | If CPU proves too slow for a fair test of the sequence models, can a serverless GPU be used for that candidate? | Without one, sequence models train on less data than the literature uses | **Answered 2026-11-18: assume CPU for now.** The learning curve shows whether capacity set the result |
| 5 | How many CPU cores does a serverless notebook expose? | Sets sequence and tree throughput; undocumented | Measured by the smoke test (`environment_report`) |
| 6 | Is the planted definition "active = hours > 0" right for the real product? | Every target depends on it | Yes |
| 7 | Is a lapse of 26+ weeks the right threshold for the returner segment? | The owner's example was "about 6 months" | 26 weeks, with 4 and 13 also as features |
| 8 | **Should each final fit be calibrated on itself?** The plan fits the calibration layer on the `tune` fit and applies it to `final_1` and `final_2`. In simulation the GRU's bias differed between fits, so it was reported 6 to 9% low on hours; the HMMs were reported 5 to 15% high. Calibrating each final fit on `val` users at its own last training origin removed that and changed no ranking (`reports/01_simulation_study.md`, section 8.1) | It changes reported levels for every family but the trees, and it is how production would calibrate | **No** (plan of record). **Recommended: yes.** Needs each final fit to score `val` users at one more origin; `evaluate` already accepts `calibration="same_fit"` |
