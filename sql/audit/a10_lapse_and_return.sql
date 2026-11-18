-- Purpose : How common lapse-and-return behavior is: users by number of past returns from a lapse of
--           26 weeks or more, and how long lapses that ended in a return lasted.
-- Grain   : one row per returns bucket (0, 1, 2, 3+).
-- Inputs  : activity_clean  (run after the clean tables are built)
-- Output  : returns_from_26w_lapse, n_users, share_users, median_lapse_weeks, p90_lapse_weeks
-- Read as : capturing these users is a stated goal of the bake-off. If the 1 and 2+ buckets are tiny,
--           the returner segments in the evaluation will be noisy; consider a larger user sample.
WITH act AS (
  SELECT user_id, week_idx,
         week_idx - LAG(week_idx) OVER (PARTITION BY user_id ORDER BY week_idx) - 1 AS gap
  FROM {{t_activity_clean}}
  WHERE hours > 0
),
u AS (
  SELECT user_id,
         COUNT_IF(gap >= 26)                                   AS n_returns_26,
         PERCENTILE(CASE WHEN gap >= 4 THEN gap END, 0.5)      AS median_lapse
  FROM act GROUP BY user_id
)
SELECT CASE WHEN n_returns_26 >= 3 THEN '3+' ELSE CAST(n_returns_26 AS STRING) END AS returns_from_26w_lapse,
       COUNT(*)                                  AS n_users,
       COUNT(*) / SUM(COUNT(*)) OVER ()          AS share_users,
       PERCENTILE_APPROX(median_lapse, 0.5)      AS median_lapse_weeks,
       PERCENTILE_APPROX(median_lapse, 0.9)      AS p90_lapse_weeks
FROM u
GROUP BY CASE WHEN n_returns_26 >= 3 THEN '3+' ELSE CAST(n_returns_26 AS STRING) END
ORDER BY returns_from_26w_lapse
