# Simulation study: the bake-off on 2 million synthetic users

Written 2026-11-20, at the end of the build pass, before anything ran on Databricks.

**STATUS: DRAFT.** Cells marked TBD wait for the last reruns.

## Read this first

**These results do not choose the model.** `AGENTS.md` rule 3 stands: synthetic data is a
test fixture. A generated world rewards whichever model resembles the generator. The study
answers three narrower questions:

1. **Does the code survive scale?** Until now the largest run was 8,000 modeled users.
2. **How fast is each family on CPU**, and do the steps fit the one-hour rule?
3. **Which conclusions hold in two different synthetic worlds, and which flip?** A
   conclusion that flips cannot be settled on synthetic data at all. One that holds in both
   is a prior worth carrying into the real run, no more.

## 1. Bottom line

TBD

## 2. What was run

`python -m ltv.sim` (`src/ltv/sim.py`) drives the bake-off straight from sequence shards,
without Spark. It uses the same code as the Databricks pipeline for everything that matters
to accuracy and speed: the NumPy features (tested equal to the SQL features on all 53
columns), every model class, the EM and training loops, the shared calibration layer and
`evaluate`. It replaces only the tables.

| | |
| --- | --- |
| Machine | 4 CPU cores, 15 GB memory, no GPU. One job at a time |
| Data | 2,000,000 users x 320 weeks per world, 20 shards of 100,000 users |
| Schedule | Validation origin week 189; test origins 241 and 267; three fits per model, as in the plan |
| Tabular training | 2.0M user-origin rows, sampled from the `train` users of 10 shards (1M users) |
| HMM training | `train` users of one shard (59,000 to 73,000 users in scope, depending on the origin) |
| GRU training | `train` users of 10 shards (about 595,000 in scope for the tune fit) |
| Evaluation | `val` and `test` users of 12 shards: 61,000 to 82,000 users per origin, depending on the world |

Models: R1, R2, R3; A1, A1tw, A2 at default tree settings with at most 300 rounds; B1 with
6 states; B2 with 4 types x 3 spend classes x (2 active + 3 dormant) = 60 states, 30 EM
passes for the tune fit and 10 warm-started passes for each final fit; C1 (GRU) with early
stopping. In the Markov world an ORACLE that knows each user's hidden state gives the
floor no model can reach.

B2 appears twice. **B2** is the model as it now stands, with the per-user hours level that
this study led to (section 6). **B2raw** is the model as it was when the study began.

**Not exercised here**: Spark and the SQL build (section 10), the tree hyperparameter
search, the other sequence variants (C1e, C2, ablations), the stack, the cost notebook.

### The two worlds

Both are in `src/ltv/synth/` and described in `docs/SYNTHETIC_DATA.md`.

| | `markov` | `rhythm` |
| --- | --- | --- |
| Process | A hidden Markov chain per user: 4 user types, engaged / casual / three dormancy depths | Spells of play and lapses with their own lengths (semi-Markov) |
| Lapse length | Geometric within a state | Gamma around a **personal** typical length, which differs widely between users |
| User differences | 4 discrete types, a spender class, a personal hours multiplier | Continuous traits that drift over a user's life |
| Favours | The HMM: it is the same kind of process. An exact oracle exists | Nothing in particular; no model here matches it |
| Successive lapse lengths of one user correlate at | about 0.05 | about 0.2 |
| Scored users who re-engage within 52 weeks | 74% | 57% |
| Scored users who spend within 52 weeks | 18% | 12% |
| Scored users dormant for more than 26 weeks | 31% | 37% |

## 3. Speed

Measured in the rhythm world with nothing else running. Serverless will differ; the smoke
test notebook measures the real numbers. What carries over is the **relative** cost and the
shape of each step.

| Step | Size | Time | Rate |
| --- | --- | --- | --- |
| Generate the world | 2M users x 320 weeks | 129 s | |
| NumPy features, training frame | 2.0M user-origin rows | 40 to 50 s | 40,000 to 50,000 rows/s |
| R1, R2 | fit on 2.0M rows, score 61,000 | 3 to 5 s, under 1 s | |
| R3 (BG/BB) | the same | 3 to 4 min | fit is the fixed cost |
| Trees, tune fit (24 models, early stopping) | 2.0M rows x 53 features, 4,479 rounds in all | 547 s | 0.12 s per boosting round |
| Trees, each final fit (frozen rounds) | the same | 490 s | |
| Trees, scoring A1 + A1tw + A2 | 61,000 to 66,000 users | 8 s | 8,000 users/s |
| HMM B1 (6 states), EM | 59,472 users, 30 passes | 236 s | 0.13 ms per user per pass |
| HMM B2 (60 states), EM, tune fit | 59,472 users, histories to week 189, 30 passes | 1,129 s | 0.63 ms per user per pass |
| HMM B2, EM, final fits | 69,000 to 73,000 users, histories to week 241 / 267, 10 passes | 614 s / 727 s | 0.9 to 1.0 ms per user per pass |
| HMM scoring, B1 / B2 | 61,000 to 66,000 users | 13 to 16 s / 39 to 54 s | 0.2 / 0.6 to 0.8 ms per user |
| GRU C1, training | batch 512 | 11 steps/s | 5,600 user-origins/s |
| GRU C1, tune fit | stopped after 4,000 steps (best at 3,000) | 366 s | |
| GRU C1, scoring | 61,000 to 66,000 users | 15 to 17 s | 4,000 users/s |
| Evaluation | 2.85M prediction rows, 9 models, 200 bootstrap draws | 157 s | |

The Markov world cost the same per unit of work. Its trees needed more rounds (5,254 in
all; 690 s). The per-user hours level adds nothing measurable to B2's cost.

### What this implies for the real run

Assumptions: a 1M-user sample (800,000 `train` users), about 300 weeks of history, the same
4 cores. Scale the times by the core count the smoke test reports.

| Step | Estimate | Fits one command (50 min)? | What was done about it |
| --- | --- | --- | --- |
| Trees `fit`, `final_*` at 5M rows | 0.3 s per round. At an average of 600 rounds per model: about 70 min | **No**, at the `full` profile's 2,000-round cap | The fit steps now save after each model and resume (`run_trees`) |
| Trees `tune` | A trial fits 3 models: 10 to 30 min at 5M rows, so 30 trials take 5 to 15 hours | Yes, 1 to 4 trials per command | Tune on 1M rows: accuracy was flat from 1M to 2M rows (section 7) |
| HMM B2 EM over all 800,000 users | 13 min per pass; 100 passes is 22 hours, for each of 3 sizes | **No** | `hmm.train_shards: 1`. One shard is about 80 s per pass, 40 min for 30 passes |
| HMM B1 EM, one shard | about 15 s per pass | Yes | |
| GRU C1 at the `full` cap of 20,000 steps | 30 min | Yes | |
| Scoring 30M users, one 4-core node | Trees about 1 hour; GRU about 2 hours; B1 about 2 hours; B2 6 to 7 hours | Batches | B2 must be spread over workers (`mapInPandas`) or it is the slowest path by 3 to 6 times |

**Run one training job at a time on a compute.** Twice during this study two multi-threaded
jobs shared the 4 cores; a tree fit that takes 8 minutes alone had not finished after an
hour. LightGBM and PyTorch each start a thread per core and stall when they compete.

## 4. Accuracy

Headline numbers: `test` users, after the shared calibration layer (plan of record), mean of
the two test origins. Lower is better except Gini. Level is predicted total / actual total
at origins 267 / 241 (1.00 is unbiased).

### Rhythm world

| Model | P(re-engage) log loss | Active weeks, Poisson deviance | Hours, Tweedie deviance | P(spend) log loss | Spend, normalized Gini | Hours level | Spend level |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A1 trees | **0.3712** | 5.632 | **8.659** | **0.1934** | **0.9320** | 1.02 / 1.01 | 0.99 / 0.97 |
| A2 trees, coherent | 0.3712 | **5.629** | 8.682 | 0.1934 | 0.9308 | 1.02 / 1.01 | 0.98 / 0.96 |
| C1 GRU | 0.3760 | 5.778 | 8.791 | 0.1964 | 0.9318 | 0.91 / 0.94 | 0.92 / 1.13 |
| B2 lapse-aware HMM | 0.4224 | 6.483 | 9.607 | 0.2115 | 0.9163 | 1.06 / 1.03 | 1.00 / 0.96 |
| B2raw (before section 6) | 0.4253 | 6.910 | 9.940 | 0.2117 | 0.9146 | 1.10 / 1.05 | 1.01 / 0.96 |
| B1 plain HMM | 0.4288 | 7.446 | 13.560 | 0.3290 | 0.5195 | 1.15 / 1.08 | 1.00 / 0.95 |
| R1 segment lookup | 0.4177 | 7.376 | 14.877 | 0.2149 | 0.8560 | 1.08 / 1.00 | 0.98 / 0.92 |
| R3 BG/BB | 0.4391 | 7.262 | 12.900 | 0.2177 | 0.9017 | 1.07 / 1.02 | 1.04 / 0.99 |
| R2 persistence | 0.5485 | 7.322 | 10.847 | 0.2510 | 0.8492 | 1.02 / 1.02 | 0.97 / 0.96 |

### Markov world

| Model | P(re-engage) log loss | Active weeks, Poisson deviance | Hours, Tweedie deviance | P(spend) log loss | Spend, normalized Gini | Hours level | Spend level |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ORACLE (knows the hidden state) | 0.2955 | 6.463 | 11.397 | 0.2102 | 0.9426 | 1.01 / 1.01 | 1.00 / 0.98 |
| A1 trees | **0.3675** | 8.333 | **14.845** | 0.2602 | 0.9016 | 1.01 / 1.01 | 1.02 / 0.99 |
| A2 trees, coherent | 0.3675 | **8.330** | 14.850 | 0.2602 | 0.9008 | 1.01 / 1.01 | 1.01 / 0.99 |
| C1 GRU | 0.3701 | 8.561 | 15.401 | 0.2639 | 0.9003 | 1.02 / 0.92 | 0.73 / 0.87 |
| B2 lapse-aware HMM | 0.3733 | 8.731 | 15.248 | **0.2598** | **0.9038** | 1.02 / 1.02 | 1.02 / 1.00 |
| B2raw (before section 6) | 0.4301 | 9.317 | 16.112 | 0.2668 | 0.8981 | 1.03 / 1.03 | 1.03 / 1.01 |
| B1 plain HMM | 0.5478 | 11.708 | 19.258 | 0.4344 | 0.4648 | 1.13 / 1.12 | 1.06 / 1.03 |
| R1 segment lookup | 0.4204 | 10.444 | 18.403 | 0.2958 | 0.7856 | 1.02 / 1.01 | 1.03 / 1.01 |
| R3 BG/BB | 0.4545 | 10.638 | 18.203 | 0.3019 | 0.8431 | 1.01 / 1.01 | 1.04 / 1.02 |
| R2 persistence | 0.5158 | 11.187 | 18.765 | 0.3507 | 0.7318 | 1.00 / 1.01 | 1.02 / 0.99 |

The oracle is given each user's true type and state, which no model can see, so the gap to
it is not all error. It marks how much of the outcome is noise: even perfect knowledge
leaves a log loss of 0.30.

### Winners

By the win rule: the same model leads at both test origins, and its paired 95% interval
against the best model of another family excludes zero at both.

| Target | Rhythm: winner (runner-up, margin) | Markov: winner (runner-up, margin) |
| --- | --- | --- |
| P(re-engage) | A1 (C1; log loss lower by 0.005, 1.3%) | A1 (C1; log loss lower by 0.002 to 0.004, about 0.7%) |
| Active weeks | A2 (C1; deviance lower by 0.11 to 0.18, 2 to 3%) | A2 (C1; deviance lower by 0.20 to 0.26, about 3%) |
| Hours | A1 (C1; deviance lower by 0.13, 1.5%) | Trees lead (B2 next, 2.7% behind); A1 and A2 swap first place between origins, so the rule names no single model |
| P(spend) | A1 (C1; log loss lower by 0.003, 1.5%) | No clear winner: B2 and A1 tie |
| Spend ranking | No clear winner: A1 and C1 tie (Gini within 0.001) | **B2** (A1; Gini higher by 0.002) |

What stands out:

- **Trees lead or tie on every target in both worlds**, including the world built in the
  HMM's image. The margins over the next family are small: 0.4 to 3%.
- **The GRU is the closest challenger in both worlds**: 0.7 to 1.3% behind on P(re-engage),
  level on spend ranking.
- **The HMM's standing flips between worlds.** In the Markov world B2 is within 1.6% of the
  trees on P(re-engage) and best on spend. In the rhythm world it is 14% behind, and behind
  the segment lookup on that target. Which of these real players resemble cannot be learned
  here.
- **B1 cannot rank spenders** (Gini about 0.5): it has no spend classes. Expected, and the
  reason B2 has them.
- **Whales**: the top 1% of users by predicted spend hold 38% of actual spend for the trees
  and the GRU in the rhythm world (35% for B2, 11% for the segment lookup), and 30% in the
  Markov world, where the oracle reaches 32%.
- **Intervals over-cover for every model**: the shared 80% band covers 91 to 93% of hours
  outcomes and 94 to 95% of spend outcomes. The bands are wider than they need to be;
  zero-heavy outcomes do that. Nothing separates the models here.

## 5. Lapse and return: the headline segment

The owner's test: a player who lapses for six months, returns and lapses again must be
recognised as a returner. Mean predicted P(re-engage) against the realised rate, `test`
users, mean of the two origins.

### Rhythm world

| Segment | Users per origin | Actual | A1 | C1 | B2 | B2raw | B1 | R1 | R3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No past return, dormant 27 to 104 weeks | 16,624 | 0.114 | 0.119 | 0.114 | 0.165 | 0.168 | 0.137 | 0.121 | 0.185 |
| 1 past return, dormant 27 to 104 weeks | 4,991 | 0.229 | 0.234 | 0.236 | 0.191 | 0.188 | 0.158 | 0.232 | 0.383 |
| **2+ past returns, dormant 27 to 104 weeks** | 2,682 | **0.496** | 0.459 | 0.460 | **0.214** | 0.216 | 0.190 | 0.509 | 0.479 |
| No past return, dormant 5 to 26 weeks | 11,848 | 0.639 | 0.646 | 0.654 | 0.700 | 0.703 | 0.709 | 0.635 | 0.540 |
| 2+ past returns, dormant 5 to 26 weeks | 3,750 | 0.774 | 0.793 | 0.819 | 0.663 | 0.678 | 0.682 | 0.789 | 0.819 |

### Markov world

| Segment | Users per origin | Actual | A1 | C1 | B2 | B2raw | B1 | R1 | R3 | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No past return, dormant 27 to 104 weeks | 13,428 | 0.326 | 0.326 | 0.327 | 0.358 | 0.480 | 0.292 | 0.341 | 0.466 | 0.325 |
| 1 past return, dormant 27 to 104 weeks | 6,278 | 0.575 | 0.582 | 0.609 | 0.586 | 0.527 | 0.292 | 0.590 | 0.551 | 0.576 |
| **2+ past returns, dormant 27 to 104 weeks** | 4,952 | **0.761** | 0.763 | 0.779 | 0.762 | **0.503** | 0.292 | 0.776 | 0.595 | 0.765 |
| No past return, dormant 5 to 26 weeks | 8,715 | 0.643 | 0.629 | 0.651 | 0.636 | 0.726 | 0.436 | 0.635 | 0.573 | 0.636 |
| 2+ past returns, dormant 5 to 26 weeks | 8,194 | 0.820 | 0.825 | 0.832 | 0.813 | 0.709 | 0.448 | 0.836 | 0.821 | 0.824 |

- **Trees and the GRU recognise serial returners in both worlds.** Among long-dormant
  players, those with two or more past returns come back two to four times as often as those
  with none (76% against 33% in the Markov world, 50% against 11% in the rhythm world), and
  both models say so. In the rhythm world they are a little low for the serial returners
  (46% against 50%).
- **The lapse-aware HMM recognises them only in the Markov world**, and only since the
  change in section 6: 76% predicted against 76% actual, where B2raw said 50%. In the rhythm
  world B2 gives serial returners 21% and first-time lapsers 17%: almost no separation.
  There a player's chance of returning *rises* as the lapse approaches their personal
  typical length. Four types and three dormancy depths cannot express that.
- **The plain HMM (B1) gives every long-dormant player the same answer**, as designed.
- The segment lookup (R1) is nearly exact in this table because the segment *is* its key.
  Read the table together with the overall metrics, where R1 is far behind.

### Falling hours before a lapse

The owner's hypothesis: engagement drives both hours and retention, so hours fall before a
player lapses. Both worlds plant this. The check: players active at the origin with at
least 13 active weeks in the last year, split by their hours per active week in the last 4
weeks against their own 52-week norm. Raw predictions, both test origins. Each model cell
is predicted / actual active weeks.

| World | Recent hours against own norm | Users per origin | Actual active weeks, next 52 | A1 | C1 | B2 | R1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Markov | under 0.5 | 2,424 | 24.1 | 1.03 | 1.07 | 0.96 | 1.03 |
| Markov | 0.5 to 0.8 | 2,716 | 28.4 | 1.00 | 1.02 | 0.93 | 0.88 |
| Markov | 0.8 to 1.25 | 5,817 | 30.0 | 0.99 | 1.01 | 0.92 | 0.83 |
| Markov | over 1.25 | 4,519 | 29.7 | 0.98 | 1.01 | 0.91 | 0.83 |
| Rhythm | under 0.5 | 469 | 21.3 | 1.03 | 0.93 | 0.88 | 0.94 |
| Rhythm | 0.5 to 0.8 | 2,140 | 22.9 | 1.02 | 0.95 | 0.87 | 0.87 |
| Rhythm | 0.8 to 1.25 | 3,036 | 24.0 | 0.99 | 0.94 | 0.86 | 0.85 |
| Rhythm | over 1.25 | 1,170 | 23.6 | 0.98 | 0.94 | 0.89 | 0.85 |

Players whose hours have halved go on to play 11 to 20% fewer weeks. The trees, the GRU and
the HMM follow most of that: each over-predicts the decliners by about 5 points relative to
the rest (the HMM's ratios sit low throughout because these are raw predictions). The
segment lookup does not follow it: 10 to 20 points. This check is not in `evaluate`; it was run once, by
hand, from the saved predictions.

## 6. The HMM spent its user types on hours

When the study began, B2 lost in the Markov world too: P(re-engage) log loss 0.430 against
0.368 for the trees, worse than the segment lookup, in a world generated by a process of
its own shape. Its fitted parameters showed why.

| B2raw, Markov world | Type 1 | Type 2 | Type 3 | Type 4 |
| --- | ---: | ---: | ---: | ---: |
| Share of users | 14% | 19% | 16% | 51% |
| Mean hours in the engaged state | 19.4 | 12.3 | 7.9 | 12.1 |
| Weekly return probability, first dormant state, ordinary week | 6.6% | 7.7% | 6.0% | 1.4% |

Three of the four types are the same player at three hours levels. Every user in that world
has a personal hours multiplier. Hours are observed every active week, and returns are
rare, so explaining who plays a lot is worth far more likelihood than explaining who comes
back. EM took the larger prize. Real players certainly differ in hours, so this would have
happened on real data. The build tests, on 6,000 users, did not show it.

**The change.** B2 now models hours relative to each user's own level: each week's hours
are divided by the mean of the user's earlier active weeks (in logs, shrunk toward the
population). It uses earlier weeks only, costs no extra states, and can be carried forward
in weekly scoring as two numbers per user. `hmm.b2_user_hours_level`, on by default.

| | B2raw | B2 | Trees (A1) |
| --- | ---: | ---: | ---: |
| Markov: P(re-engage) log loss | 0.4301 | 0.3733 | 0.3675 |
| Markov: serial returners, long dormant (actual 0.761) | 0.503 | 0.762 | 0.763 |
| Markov: hours deviance | 16.11 | 15.25 | 14.85 |
| Rhythm: P(re-engage) log loss | 0.4253 | 0.4224 | 0.3712 |
| Rhythm: serial returners, long dormant (actual 0.496) | 0.216 | 0.214 | 0.459 |
| Rhythm: active weeks deviance | 6.91 | 6.48 | 5.63 |

The change repaired the HMM where the world is HMM-shaped. In the rhythm world it improved
amounts and left the returner problem as it was, because that one is structural (section 5).

**For the real run**: after `30_hmm` tunes, read the state profiles. If the types differ in
hours and not in return probability, the model has not learned what it is for
(`docs/RUNBOOK.md`, step 6).

TBD-B2BIG

## 7. Learning curves

Validation origin, raw predictions, rhythm world.

| Family | Training size | P(re-engage) log loss | Hours, Tweedie deviance | Spend Gini |
| --- | ---: | ---: | ---: | ---: |
| A1 trees | 30,000 rows | 0.3801 | 9.83 | 0.918 |
| | 100,000 rows | 0.3774 | 9.52 | 0.920 |
| | 300,000 rows | 0.3748 | 9.32 | 0.922 |
| | 1,000,000 rows | 0.3734 | 9.20 | 0.923 |
| | 2,000,000 rows | 0.3728 | 9.17 | 0.923 |
| B2 HMM | 2,000 users | 0.4299 | 10.07 | 0.906 |
| | 10,000 users | 0.4203 | 9.85 | 0.909 |
| | 59,472 users | 0.4276 | 10.07 | 0.909 |
| C1 GRU | 5,000 users | 0.3904 | 9.87 | 0.911 |
| | 30,000 users | 0.3852 | 9.78 | 0.918 |
| | 595,055 users | 0.3763 | 9.36 | 0.922 |

- **Trees are nearly saturated by 1M rows.** Going to 2M buys 0.0006 in log loss. The 5M-row
  cap is more than this problem needed; a smaller frame costs little.
- **The HMM does not improve with data** beyond a few thousand users, and its three points
  are not even in order. It has few parameters; its error is in its structure and in where
  EM happens to settle, not in its estimates. This is what justifies training it on one
  shard. (B2raw was just as flat: 0.435, 0.436, 0.434.)
- **The GRU is still improving** at the largest size, and it is the family whose result is
  most likely to be set by the compute budget. This matches the plan's expectation and is
  why the real run records a learning curve for it.

## 8. Calibrating each final fit on itself

The plan fits the calibration layer on the `tune` fit (validation origin) and applies it to
`final_1` and `final_2`. That assumes a model's bias is the same from one fit to the next.
It is for the trees. It is not for the GRU: in the rhythm world its `tune` fit
over-predicted hours by 20%, its final fits by 8 and 12%, so the layer over-corrected and
the GRU is **reported** 6 to 9% low. In the Markov world the same thing put its reported
spend 13 to 27% low. That bias belongs to the evaluation design, not to the model, and
production would not have it, because production calibrates the fit it serves.

The simulation therefore also scored each final fit on `val` users at its last training
origin (labels complete before the test origin, users never trained on) and fitted the
layer there (`calibration="same_fit"`). Level at origins 267 / 241:

| World | Model | Target | Level, plan of record | Level, same fit |
| --- | --- | --- | --- | --- |
| Rhythm | A1 | Hours | 1.02 / 1.01 | 1.00 / 1.01 |
| Rhythm | C1 | Hours | 0.91 / 0.94 | 0.99 / 1.00 |
| Rhythm | B1 | Hours | 1.15 / 1.08 | 1.00 / 1.01 |
| Markov | A1 | Spend | 1.02 / 0.99 | 1.02 / 1.00 |
| Markov | C1 | Spend | 0.73 / 0.87 | 1.03 / 1.00 |
| Markov | C1 | Hours | 1.02 / 0.92 | 1.01 / 1.01 |

Rankings did not change in either world. Levels did, and the GRU's accuracy on amounts
improved a little (Markov active weeks deviance 8.56 to 8.44).

**Recommendation**: adopt same-fit calibration in the Databricks pipeline before the real
run. It needs each final fit to score `val` users at one more origin, and the evaluation
frame to carry those rows. It is not built there yet: it changes a decision of record, so
it is open question 8 in `docs/DECISIONS.md`.

## 9. What the study found wrong, and what was changed

Each of these would have surfaced on Databricks, most of them as a failure or a wrong
number on real data.

| # | Finding | Consequence if left | Change |
| --- | --- | --- | --- |
| 1 | R3's fit crashed: weighted sampling of history patterns fails when one pattern dominates | `15_baselines` fails on any large sample | Sample users before counting patterns (`models/bgbb.py`) |
| 2 | **The shared calibration layer broke the HMM.** Isotonic regression ends in steps fitted on a handful of validation users and clips everything outside the validation range onto them. The map is fitted on the `tune` fit and applied to the final fits, whose predictions sit slightly differently. B1 had 12,700 users below the validation range: their P(spend) became exactly 0. B2's predicted hours rose by 42 to 54% | The HMM's reported numbers would have been an artefact: hours deviance 24.0 at one origin instead of 10.1, P(spend) log loss 0.55 instead of 0.33 | The map is fitted on up to 50 bins of validation users, smooths probabilities, and extrapolates amounts in proportion (`eval/calibration.py`). No model lost accuracy |
| 3 | **The win rule could not name a winner.** The leader was compared with the runner-up, which was its own sibling: A1, A1tw and A2 share one P(re-engage) model and tie exactly | "No clear winner" on every target, whatever the data | The leader is compared with the best model of another family (`eval/evaluate.py`) |
| 4 | **B2 spent its user types on hours** (section 6) | The lapse-aware HMM could not recognise a returner, which is what it exists for | Hours are modeled relative to each user's own level (`hmm.b2_user_hours_level`) |
| 5 | HMM EM read every shard | About 22 hours per B2 candidate at the 1M sample | `hmm.train_shards: 1` |
| 6 | Tree `fit` and `final_*` saved nothing until all 24 models were done | A step over the time limit with no checkpoint | The steps save after each model and resume |
| 7 | Predictions for a whole step were built in memory before writing; evaluation loaded every split | Memory pressure at 16 GB with a 1M sample | Predictions are written one model at a time; evaluation loads `val` and `test` only (`evaluation.splits`) |
| 8 | The GRU's early stopping watches a noisy number at a constant learning rate and stopped after 3,000 steps | The sequence candidate may be under-trained | Optional `seq.lr_drops` (off by default); TBD |
| 9 | The calibration map is fitted on one fit and applied to another | Reported levels are wrong for models whose bias moves between fits (section 8) | Measured here; **a decision for the owner** |

## 10. Spark build at scale

TBD

## 11. What to expect on real data

TBD

## 12. What this study cannot tell you

- **Which model wins on real data.** Real players are neither world.
- **Serverless speed.** Core count, memory behaviour and Spark Connect overhead are unknown
  until the smoke test runs there.
- **Anything about the SQL build at real scale**, beyond section 10.
- **Tuned performance.** Trees ran at default settings; one HMM size per family; one GRU
  variant. Tuning can move each family by more than the gap between the top two.
- **Calendar effects.** Both worlds use the same simple season calendar.

## 13. How to reproduce

```bash
python -m ltv.sim --world rhythm --users 2000000 --out /path/to/rhythm --eval_shards 12 --seq_steps 10000
```

```bash
python -m ltv.sim --world markov --users 2000000 --out /path/to/markov --eval_shards 12 --seq_steps 10000 --curves false
```

Each world needs about 400 MB of disk for shards and takes about 2.5 hours on 4 cores.
Every stage is cached: an interrupted run continues when called again. Results are written
to `results/` (plan-of-record calibration) and `results_same_fit/` under the output
directory; `timings.jsonl` holds one line per stage. `tests/test_sim.py` runs the whole
driver on 6,000 users. B2raw is not produced by these commands; set
`hmm.b2_user_hours_level: false` to get it.
