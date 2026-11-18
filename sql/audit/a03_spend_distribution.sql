-- Purpose : How rare and how heavy-tailed spend is. Sets expectations for the spend metrics.
-- Grain   : one row.
-- Inputs  : {{activity_table}}
-- Output  : n_users, share_users_ever_spend, weekly spend quantiles (positive weeks only),
--           lifetime spend quantiles (spenders only), share of spend from the top 1% of spenders
WITH u AS (
  SELECT {{act_user_id}} AS user_id, SUM(GREATEST(CAST({{act_spend}} AS DOUBLE), 0)) AS life_spend
  FROM {{activity_table}} GROUP BY {{act_user_id}}
),
uq AS (
  SELECT COUNT(*) AS n_users,
         COUNT_IF(life_spend > 0) / COUNT(*)                                AS share_users_ever_spend,
         PERCENTILE_APPROX(CASE WHEN life_spend > 0 THEN life_spend END, 0.5)   AS life_p50,
         PERCENTILE_APPROX(CASE WHEN life_spend > 0 THEN life_spend END, 0.9)   AS life_p90,
         PERCENTILE_APPROX(CASE WHEN life_spend > 0 THEN life_spend END, 0.99)  AS life_p99,
         PERCENTILE_APPROX(CASE WHEN life_spend > 0 THEN life_spend END, 0.999) AS life_p999,
         SUM(life_spend)                                                    AS total_spend
  FROM u
),
top AS (
  SELECT SUM(u.life_spend) AS top1pct_spend
  FROM u CROSS JOIN uq
  WHERE u.life_spend >= uq.life_p99
),
wq AS (
  SELECT PERCENTILE_APPROX(CAST({{act_spend}} AS DOUBLE), 0.5)   AS week_p50,
         PERCENTILE_APPROX(CAST({{act_spend}} AS DOUBLE), 0.9)   AS week_p90,
         PERCENTILE_APPROX(CAST({{act_spend}} AS DOUBLE), 0.99)  AS week_p99,
         PERCENTILE_APPROX(CAST({{act_spend}} AS DOUBLE), 0.999) AS week_p999
  FROM {{activity_table}} WHERE {{act_spend}} > 0
)
SELECT uq.n_users, uq.share_users_ever_spend,
       wq.week_p50, wq.week_p90, wq.week_p99, wq.week_p999,
       uq.life_p50, uq.life_p90, uq.life_p99, uq.life_p999,
       top.top1pct_spend / uq.total_spend AS share_spend_top1pct_spenders
FROM uq CROSS JOIN wq CROSS JOIN top
