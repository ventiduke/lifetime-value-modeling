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

Not built, and why:

| Item | Status |
| --- | --- |
| Weekly carried-state scoring job for the GRU | The property it relies on is tested (`test_block_state_can_be_carried_forward`) and the kernel exists; the job is not built. Scoring recomputes from history |
| HMM quantiles | Not produced; the shared conformal layer gives intervals |
| Transformer on episode tokens | C2 runs on blocks plus recent weeks only |
| Revision check wired into a notebook | The functions and tests exist (`eval/revisions.py`); not wired, as lift is low priority |
| Asset Bundle jobs | A skeleton only (`jobs/databricks.yml`), unverified |
| Usability scores (1 to 5) | A human judgement for the final report; the inputs (coherence, coverage, state profiles) are produced |

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
