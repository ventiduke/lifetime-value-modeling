-- Purpose : The five targets and their sub-horizon breakdowns for each user x origin.
-- Grain   : one row per user x origin (same keys as user_origins, origins with a full label window only).
-- Inputs  : user_origins, activity_clean, calendar_weeks
-- Output  : labels(user_id, origin_week, any_active_52, active_weeks_52, hours_52, any_spend_52,
--                  spend_52, spend_weeks_52, cum_hours, cum_spend, weekly_active)
-- Notes   : * Label window is weeks origin+1 .. origin+{{horizon}}. Nothing at or before the origin is read.
--           * cum_hours / cum_spend are cumulative totals at weeks 4, 13, 26, 39 and 52 (array of 5).
--           * weekly_active is an array of {{horizon}} 0/1 flags; element h (1-based) is week origin+h.
--           * "Active" means hours > 0. Spend-only weeks count toward spend, not activity.
WITH lim AS (
  SELECT MAX(week_idx) AS last_data_week FROM {{t_calendar_weeks}} WHERE has_data
),
fut AS (
  SELECT uo.user_id,
         uo.origin_week,
         COUNT_IF(a.hours > 0)                                                            AS active_weeks_52,
         COUNT_IF(a.spend > 0)                                                            AS spend_weeks_52,
         COALESCE(SUM(a.hours), 0.0)                                                      AS hours_52,
         COALESCE(SUM(a.spend), 0.0)                                                      AS spend_52,
         COALESCE(SUM(CASE WHEN a.week_idx <= uo.origin_week + 4  THEN a.hours END), 0.0) AS hours_4,
         COALESCE(SUM(CASE WHEN a.week_idx <= uo.origin_week + 13 THEN a.hours END), 0.0) AS hours_13,
         COALESCE(SUM(CASE WHEN a.week_idx <= uo.origin_week + 26 THEN a.hours END), 0.0) AS hours_26,
         COALESCE(SUM(CASE WHEN a.week_idx <= uo.origin_week + 39 THEN a.hours END), 0.0) AS hours_39,
         COALESCE(SUM(CASE WHEN a.week_idx <= uo.origin_week + 4  THEN a.spend END), 0.0) AS spend_4,
         COALESCE(SUM(CASE WHEN a.week_idx <= uo.origin_week + 13 THEN a.spend END), 0.0) AS spend_13,
         COALESCE(SUM(CASE WHEN a.week_idx <= uo.origin_week + 26 THEN a.spend END), 0.0) AS spend_26,
         COALESCE(SUM(CASE WHEN a.week_idx <= uo.origin_week + 39 THEN a.spend END), 0.0) AS spend_39,
         COLLECT_LIST(CASE WHEN a.hours > 0 THEN a.week_idx - uo.origin_week END)         AS active_offsets
  FROM {{t_user_origins}} uo
  CROSS JOIN lim
  LEFT JOIN {{t_activity_clean}} a
    ON a.user_id = uo.user_id
   AND a.week_idx > uo.origin_week
   AND a.week_idx <= uo.origin_week + {{horizon}}
  WHERE uo.origin_week + {{horizon}} <= lim.last_data_week
  GROUP BY uo.user_id, uo.origin_week
)
SELECT user_id,
       origin_week,
       CAST(active_weeks_52 > 0 AS INT)                              AS any_active_52,
       CAST(active_weeks_52 AS INT)                                  AS active_weeks_52,
       hours_52,
       CAST(spend_weeks_52 > 0 AS INT)                               AS any_spend_52,
       spend_52,
       CAST(spend_weeks_52 AS INT)                                   AS spend_weeks_52,
       ARRAY(hours_4, hours_13, hours_26, hours_39, hours_52)        AS cum_hours,
       ARRAY(spend_4, spend_13, spend_26, spend_39, spend_52)        AS cum_spend,
       TRANSFORM(SEQUENCE(1, {{horizon}}),
                 h -> CAST(ARRAY_CONTAINS(active_offsets, h) AS TINYINT)) AS weekly_active
FROM fut
