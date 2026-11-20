# Sample of the example-players export (SYNTHETIC)

A small sample of what `notebooks/55_examples` writes, so a plotting workbook can be built
before the real run exists. **The data is simulated.** It says nothing about real players
or about which model is better.

| File | What it is |
| --- | --- |
| `example_weekly_sample.csv` | The long table: 11 players, one row per player, series, view and week. Same columns as the `example_weekly` table (`docs/DATA_CONTRACT.md`) |
| `example_users_sample.csv` | One row per player: why they were picked, their state at the origin, what happened next |
| `returns_after_a_long_lapse.png`, `hours_falling.png` | The notebook's preview plot (`ltv.eval.examples.plot_player`) for two of the players |

How it was made: the "rhythm" world of the simulation study (`reports/01_simulation_study.md`),
one shard of 100,000 users, models A1, A2 and C1. It shows the `tune` fits at the
validation origin (week 189), because the simulation saves no final tree fits; the real
export shows the `final_1` fits at the first test origin. Dates are invented: week 0 is
2020-01-06.

Three plots to build (also listed in the notebook):

| Question | Filter | x | y | Color |
| --- | --- | --- | --- | --- |
| How did the forecast compare with what happened? | one `user_id`; `view` in (`actual`, `path`) | `week_start` | `hours` | `series` |
| What did each model think during a lapse? | one `user_id`; `view` = `as_of` | `week_start` | `p_reengage_52` | `series` |
| Did the totals add up over the year? | one `user_id`; `view` in (`actual`, `path`); `weeks_from_origin` > 0 | `weeks_from_origin` | `cum_hours` | `series` |
