-- Purpose : Base rates at a recent origin: how many users are in each dormancy bucket, and how often
--           they re-engage or spend in the next 52 weeks. Sets the sample size needed for rare outcomes.
-- Grain   : one row per dormancy bucket.
-- Inputs  : activity_clean, users, calendar_weeks  (run after the clean tables are built)
-- Output  : dormancy_bucket, n_users, share_users, p_reengage_52, p_spend_52, mean_active_weeks_52
-- Notes   : origin = last data week - {{horizon}}. "Dormancy" = weeks since the last active week at the origin.
--           Users with no active week by the origin (spend-only) are in bucket 'never active'.
WITH lim AS (
  SELECT MAX(week_idx) - {{horizon}} AS origin FROM {{t_calendar_weeks}} WHERE has_data
),
past AS (
  SELECT a.user_id, MAX(CASE WHEN a.hours > 0 THEN a.week_idx END) AS last_active
  FROM {{t_activity_clean}} a CROSS JOIN lim
  WHERE a.week_idx <= lim.origin
  GROUP BY a.user_id
),
fut AS (
  SELECT a.user_id, COUNT_IF(a.hours > 0) AS aw, COUNT_IF(a.spend > 0) AS sw
  FROM {{t_activity_clean}} a CROSS JOIN lim
  WHERE a.week_idx > lim.origin AND a.week_idx <= lim.origin + {{horizon}}
  GROUP BY a.user_id
),
j AS (
  SELECT CASE WHEN p.last_active IS NULL THEN '6: never active'
              WHEN lim.origin - p.last_active = 0 THEN '1: active this week'
              WHEN lim.origin - p.last_active <= 4 THEN '2: dormant 1-4'
              WHEN lim.origin - p.last_active <= 26 THEN '3: dormant 5-26'
              WHEN lim.origin - p.last_active <= 104 THEN '4: dormant 27-104'
              ELSE '5: dormant over 104' END AS dormancy_bucket,
         COALESCE(f.aw, 0) AS aw, COALESCE(f.sw, 0) AS sw
  FROM past p CROSS JOIN lim LEFT JOIN fut f ON f.user_id = p.user_id
)
SELECT dormancy_bucket,
       COUNT(*)                                 AS n_users,
       COUNT(*) / SUM(COUNT(*)) OVER ()         AS share_users,
       AVG(CAST(aw > 0 AS DOUBLE))              AS p_reengage_52,
       AVG(CAST(sw > 0 AS DOUBLE))              AS p_spend_52,
       AVG(CAST(aw AS DOUBLE))                  AS mean_active_weeks_52
FROM j
GROUP BY dormancy_bucket
ORDER BY dormancy_bucket
