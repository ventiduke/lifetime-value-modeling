# Data contract

What the pipeline expects from the source tables, and the grain, keys and columns of every
table it writes. All output tables live in `{catalog}.{schema}` from `conf/bakeoff.yml`.
Code gets a table's full name from `cfg.table("<name>")`; never write a name by hand.

Conventions used everywhere:

- **`week_idx`** (INT): integer week. 0 is the first calendar week that has activity.
- **`origin_week`** (INT): a scoring week `t`. Features use weeks `<= t`; labels use weeks
  `t+1 .. t+52`.
- **Active** means `hours > 0`. A week with spend but no hours is a *spend-only* week.
- **`split`**: `train`, `val` or `test`, a hash of `user_id`, fixed for the user.
- **`fit`**: `tune`, `final_1` or `final_2` (see `docs/DESIGN.md`, "Splits").

## Source tables (read only)

Real names and column names are mapped in `conf/bakeoff.yml` under `source:`.

### Activity table (`source.activity_table`)

Sparse: a user has a row only in weeks they were active or spent.

| Logical column | Type | Expectation |
| --- | --- | --- |
| `user_id` | any, cast to STRING | Stable over time for a person |
| `week_start` | DATE or castable | The first day of the week; must equal a `week_start` in the calendar |
| `hours` | numeric | Hours played that week, `>= 0` |
| `spend` | numeric | Total spend that week, one currency, net of refunds, `>= 0` |
| `platform` | string | One per user-week |
| `country` | string | One per user-week |

What the pipeline does when an expectation fails (rules in `sql/build/02_activity_clean.sql`,
settings under `cleaning:`; audit `a01` to `a08` report how often each happens):

| Problem | Handling |
| --- | --- |
| `week_start` is not a calendar week | Row dropped. Audit `a05` FAILS: fix the mapping, do not continue |
| Several rows for one user-week | Hours and spend summed; platform and country from the row with most hours |
| Negative spend | Clipped to 0 (`negative_spend: clip`) or row dropped (`drop`) |
| Null hours or spend | Treated as 0 |
| Hours and spend both 0 | Row dropped |
| A user with any week over `max_weekly_hours` | User dropped as a suspect account |
| User listed in `source.exclude_users_table` | User dropped |

### Calendar table (`source.calendar_table`)

One row per week, covering every week of activity; it may extend into the future.

| Logical column | Type | Expectation |
| --- | --- | --- |
| `week_start` | DATE | One row per week, no gaps |
| `season` | any orderable | Season identifier |
| `season_week` | INT | 1 = the launch week of the season, counting up |

## Tables the pipeline writes

Built by `ltv.pipeline.build_all` from the numbered files in `sql/build/`.

### `calendar_weeks` (01)
Grain: one row per calendar week from the first data week on. Key: `week_idx`.

| Column | Type | Meaning |
| --- | --- | --- |
| `week_idx` | INT | Integer week, 0 = first week with activity |
| `week_start` | DATE | |
| `season` | STRING | |
| `season_idx` | INT | Seasons numbered in time order from 0 |
| `season_week` | INT | 1 = launch week |
| `is_launch` | BOOLEAN | `season_week = 1` |
| `has_data` | BOOLEAN | False for calendar weeks after the last week of activity |

### `activity_clean` (02)
Grain: one row per user x week with activity or spend, **for the user sample**
(`sample.fraction`, by hash). Key: (`user_id`, `week_idx`).

| Column | Type | Meaning |
| --- | --- | --- |
| `user_id` | STRING | |
| `week_idx` | INT | |
| `hours` | DOUBLE | `>= 0` |
| `spend` | DOUBLE | `>= 0` |
| `platform`, `country` | STRING | May be null |
| `n_source_rows` | INT | Source rows merged into this one |
| `had_negative_spend` | BOOLEAN | A source row had negative spend |

### `dim_platform` (03), `dim_country` (04)
Grain: one row per value. `platform_id` / `country_id` (INT) number the values from 1, most
common first. Id 0 means unknown.

### `users` (05)
Grain: one row per sampled user. Key: `user_id`.

| Column | Type | Meaning |
| --- | --- | --- |
| `first_week` | INT | First week with any row; tenure starts here |
| `last_week` | INT | Last week with any row |
| `first_active_week` | INT | First week with `hours > 0` (null if spend-only) |
| `n_active_weeks`, `n_spend_weeks` | INT | Lifetime counts |
| `split` | STRING | `train` / `val` / `test` |

### `origins` (Python, `ltv.data.splits`)
Grain: one row per origin x role. An origin can have several roles.

| Column | Type | Meaning |
| --- | --- | --- |
| `origin_week` | INT | |
| `role` | STRING | `tune_train`, `validation`, `final_train_1`, `final_train_2`, `test_1`, `test_2` |

### `calendar_projected` (Python, `ltv.data.splits`)
Grain: one row per origin x forecast week. The calendar a model is allowed to see for the
future (the future-calendar rule).

| Column | Type | Meaning |
| --- | --- | --- |
| `origin_week` | INT | |
| `h` | INT | 1 .. 52, the week `origin_week + h` |
| `season_week_proj` | INT | Projected season week |
| `is_launch_proj` | BOOLEAN | Projected launch week |
| `known` | BOOLEAN | True if taken from the real calendar (inside the known window) |

### `user_origins` (06)
Grain: one row per user x origin, for users **in the population** at that origin (any active
or spend week in the 104 weeks up to it). Key: (`user_id`, `origin_week`).
Evaluation origins keep every such user; training-only origins keep a hashed subset, so each
user contributes about `splits.max_train_origins_per_user` origins.

| Column | Type | Meaning |
| --- | --- | --- |
| `split` | STRING | The user's split |
| `is_eval_origin` | BOOLEAN | Validation or test origin |
| `tenure_weeks` | INT | `origin_week - first_week` |

### `features_tabular` (07)
Grain and key as `user_origins`. Columns: `user_id`, `origin_week`, `split`, then the
features below, all DOUBLE; NULL means "not defined" (for example no past lapse).
`src/ltv/data/np_features.py` lists them in order as `FEATURE_NAMES`.

| Group | Columns |
| --- | --- |
| Recency | `tenure_weeks`, `wk_since_active`, `wk_since_spend` |
| Frequency and level, windows of 4/13/26/52/104 weeks | `aw_*` active weeks, `hrs_*` hours, `spd_*` spend, `sw_*` spend weeks |
| Lifetime | `aw_life`, `hrs_life`, `spd_life`, `sw_life`, `max_hrs_52` |
| Trend | `ratio_aw_4_13` (last 4 weeks vs the 13 before), `hrs_slope_26` |
| Seasonality | `share_aw_launch` (share of active weeks in season weeks 1-2), `n_seasons_active`, `aw_next13_1y`, `aw_next13_2y` (activity in the 13 weeks after this date one and two years ago), `origin_season_week` |
| Lapse and return | `n_returns_4`, `n_returns_13`, `n_returns_26` (past lapses of at least that many weeks that ended in a return), `median_lapse`, `max_lapse`, `cur_lapse_rel` (current lapse / median past lapse), `n_spells`, `mean_spell_len`, `wk_since_return`, `share_returns_launch` |
| Context | `platform_id`, `country_id` (latest known), `n_platforms` |
| Horizon context, projected calendar only | `wk_to_next_launch`, `launches_q1` .. `launches_q4` |

Definitions: a *gap* is the number of inactive weeks before an active week; a *lapse* is a
gap of 4 or more; a *spell* is a run of active weeks with no lapse inside it.

### `labels` (08)
Grain and key as `user_origins`, for origins with a full 52-week label window.

| Column | Type | Meaning |
| --- | --- | --- |
| `any_active_52` | INT | 1 if any active week in the window: the P(re-engage) label |
| `active_weeks_52` | INT | 0 .. 52 |
| `hours_52` | DOUBLE | |
| `any_spend_52` | INT | The P(spend) label |
| `spend_52` | DOUBLE | |
| `spend_weeks_52` | INT | |
| `cum_hours`, `cum_spend` | ARRAY<DOUBLE> (5) | Cumulative totals at weeks 4, 13, 26, 39, 52 |
| `weekly_active` | ARRAY<TINYINT> (52) | Element `h` (1-based) is 1 if active in week `origin_week + h` |

### `user_sequences` (09)
Grain: one row per sampled user. The input of the HMM and the sequence models.

| Column | Type | Meaning |
| --- | --- | --- |
| `user_id`, `split` | STRING | |
| `first_week`, `last_data_week` | INT | Arrays cover weeks `first_week .. last_data_week` |
| `hours`, `spend` | ARRAY<FLOAT> | 0 in weeks with no row |
| `platform_id`, `country_id` | ARRAY<SMALLINT> | 0 in weeks with no row (not carried forward) |

`ltv.pipeline.export_sequences` also writes these as NumPy shards under
`{artifact_path}/shards/` (about `training.seq_shard_users` users each).

### `panel_weekly` (10)
Grain: one row per sampled user x week, from the user's first week. A convenience for
analysis and a test oracle; **full-scale scoring never builds it**.
Columns: `user_id`, `week_idx`, `hours`, `spend`, `active`, `platform`, `country` (carried
forward), `platform_carried`, `tenure_weeks`, `weeks_since_last_active`,
`weeks_since_last_spend`, `active_streak`, `season_idx`, `season_week`, `is_launch`.

### `predictions`
Grain: one row per model x origin x user. Every candidate writes this schema
(`ltv.models.base.PRED_SCHEMA`). Rerunning one model replaces only its own rows.

| Column | Type | Meaning |
| --- | --- | --- |
| `model_id` | STRING | `R1`, `R2`, `R3`, `A1`, `A1tw`, `A2`, `B1`, `B2`, `C1`, `C1e`, `C2`, ablations |
| `fit` | STRING | `tune`, `final_1`, `final_2` |
| `origin_week` | INT | The validation origin for `tune`; a test origin otherwise |
| `user_id` | STRING | Every user in the population at the origin, all splits (`STACK`: only the splits in `evaluation.splits`) |
| `p_reengage`, `exp_active_weeks`, `exp_hours`, `p_spend`, `exp_spend` | DOUBLE | The five targets |
| `cum_hours`, `cum_spend` | ARRAY<DOUBLE> (5) | Cumulative expectations at weeks 4, 13, 26, 39, 52; null if the model has none (A1) |
| `weekly_p_active` | ARRAY<FLOAT> (52) | P(active) per forecast week; null if none |
| `q_hours`, `q_spend` | ARRAY<DOUBLE> (3) | Quantiles 0.1, 0.5, 0.9 of the 52-week total; null if none |
| `state` | ARRAY<FLOAT> | HMM state probabilities, or a sequence-model embedding; null otherwise |

### `hmm_user_state`
Grain: one row per user x test origin. The B2 state vector (`alpha`, ARRAY<DOUBLE>) that
weekly scoring would carry forward.

### `eval_metrics`
Long format, written by `ltv.pipeline.run_eval`. One row per model x origin x split x
calibrated x target x segment x metric.
Columns: `model_id`, `origin_week`, `split`, `calibrated` (after the shared recalibration
layer), `target`, `segment_dim`, `segment`, `n`, `metric`, `value`.
Headline rows: `split = 'test'`, `calibrated = true`, `segment_dim = 'all'`.

### `eval_pairwise`
One row per test origin x target x model: the model against the leader (or the leader
against the best model of another family), on the target's primary metric.
Columns: `origin_week`, `target`, `rank`, `model_a`, `model_b`, `value_a`, `value_b`, `n`,
`metric`, `diff`, `ci_low`, `ci_high`, `a_better` (+1 significant, 0 not, -1 worse).

### `score_user_origins`, `score_features`, `score_predictions`, `cost_timings`
Scratch tables of the timing harness (`ltv.pipeline.run_cost`). `cost_timings` holds one
row per timed step: `step`, `model`, `users`, `seconds`, `users_per_second`, and linear
extrapolations to 30M and 150M users.

## Files under `artifact_path`

| Path | Contents |
| --- | --- |
| `shards/shard_*.npz` | User histories as NumPy arrays |
| `trees/tuning.jsonl`, `trees/suite_*.pkl` | Search log; fitted tree suites per fit |
| `hmm/tune_*.pkl`, `hmm/selection.pkl`, `hmm/model_*.pkl` | EM checkpoints; candidate scores; chosen models |
| `seq/<variant>/config.pkl`, `*.pt`, `model_*.pkl`, `learning_curve.pkl` | Sequence-model checkpoints and results |
| `run_log.jsonl` | One line per pipeline step: name, parameters, numeric results, wall-clock seconds, context |

Deleting a checkpoint makes the step start again from scratch; nothing else depends on it.
