# Synthetic data

`src/ltv/synth/generator.py` simulates weekly activity and spend from a known process.

## What it is for, and what it is not for

**For**: executing the pipeline before it meets real data. It lets the tests check logic
against known answers, shows that each Phase 0 audit query catches the defect it is meant
to, and gives Databricks a smoke test (`notebooks/05_synth_smoke_test`) that needs no real
table.

**Not for**: comparing, tuning or choosing models. **Never.** The generator is a hidden
Markov process with user types and a dormancy chain, which is the structure of candidate
B2. It favours the HMM by construction. A synthetic scorecard says the pipeline runs; it
says nothing about which model is better on real players.

Guardrails in the code: synthetic tables go to their own schema (`conf/synth.yml`), and the
results report generated from a schema with `synth` in its name is titled "SYNTHETIC DATA:
pipeline check only, not a model comparison".

## What is simulated

| Element | How |
| --- | --- |
| User types (fixed for life) | steady 25%, serial returner 20%, launch returner 20%, one-time 35% |
| Weekly states | two active levels (`hi`, `lo`) and a dormancy chain `d1 -> d2 -> d3` |
| Transitions | per type and calendar bucket (launch week, season week 2, other) |
| Activity | P(active) 0.93 in `hi`, 0.55 in `lo`, 0 when dormant |
| Hours | lognormal by state, times a per-user effect, capped at 120 |
| Spender classes | none 65%, normal 30%, whale 5%; they scale the weekly spend probability, and whales spend about 4.5 times as much |
| Spend-only weeks | a small spend probability in dormant states |
| Seasons | lengths drawn from 7 to 11 weeks; launches multiply return probabilities |
| Arrivals | 15% at week 0, 10% in weeks 1 to 8, the rest spread out and favouring launches |
| Platform, country | a main platform with occasional switches; countries Zipf-distributed |

**Serial returners** are the planted long-term behavior: they lapse readily, sit in dormancy
for months, and nearly always come back, repeatedly. **Launch returners** come back mostly in
the first two weeks of a season. **One-time players** rarely return. Among synthetic users
dormant at a late origin, about 91% of serial returners, 78% of launch returners, 67% of
steady players and 5% of one-time players re-engage within 52 weeks (20,000 users, seed 5).
Tests use this: a model that sees the whole history should give returners a higher return
probability than churned players, and a memoryless model cannot.

Defaults (`conf/synth.yml`): 50,000 users, 320 weeks, seed 7. Generation takes a few seconds.

## Planted defects (`plant_defects`)

One per Phase 0 check. `tests/test_data_layer.py` checks that each audit query reports its
defect and that the clean tables contain none of them.

| Defect | What is planted | Audit that catches it | What cleaning does |
| --- | --- | --- | --- |
| Pipeline outage | one week loses 85% of its rows | `a01` flags the week | nothing; a real outage must be handled by people |
| Zero rows | 300 rows with zero hours and zero spend | `a02` | dropped |
| Negative spend | 150 refund rows at -9.99 | `a02` | clipped to 0 |
| Duplicate user-weeks | 200 extra rows on a platform `handheld` | `a04` | summed; platform from the row with most hours |
| Test accounts | 20 users at 400 hours a week | `a07`, `a02` | users dropped |
| Misaligned weeks | 100 rows shifted two days | `a05` FAILS | rows dropped |
| Missing dimensions | 100 rows with null platform and country | `a02` | kept, id 0 |

## Ground truth and the oracle

`generate()` returns the simulated truth: each user's type, spender class, per-user effect,
weekly state, hours and spend. `oracle_predict` uses it to compute the **exact** 52-week
expectations for every user at an origin, given their true state. The oracle is the best any
model could do, and is calibrated by construction (its mean predictions match realised means
to within sampling noise). Tests use it to check that:

- metrics read near zero error on true probabilities;
- the HMM's forecasts correlate with the oracle's and are calibrated in aggregate;
- B2 recovers the planted spend classes.

`to_store` turns the truth into a `SequenceStore` directly, so model tests need no Spark.

## What synthetic data cannot tell you

Scale, cost, the one-hour limit, Spark Connect behavior, and every quirk of the real tables.
Phase 0 on real data remains the first step of the run pass.
