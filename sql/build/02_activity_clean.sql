-- Purpose : Clean, sampled copy of the sparse activity table, keyed by integer week.
-- Grain   : one row per user x week in which the user was active (hours > 0) or spent (spend > 0).
-- Inputs  : {{activity_table}}, calendar_weeks, optional exclude table.
-- Output  : activity_clean(user_id, week_idx, hours, spend, platform, country, n_source_rows, had_negative_spend)
-- Rules   : * Sample users by hash (sample.fraction), so the same users are kept on every run.
--           * Drop any user with a week above cleaning.max_weekly_hours (suspect account).
--           * Rows whose week_start is not a calendar week are dropped (see audit a05).
--           * Several source rows for one user-week are summed; platform and country come
--             from the row with the most hours.
--           * Negative spend is clipped to 0, or the row dropped (cleaning.negative_spend).
--           * Rows left with hours = 0 and spend = 0 are dropped.
WITH raw AS (
  SELECT CAST({{act_user_id}} AS STRING)                                   AS user_id,
         CAST({{act_week_start}} AS DATE)                                  AS week_start,
         GREATEST(COALESCE(CAST({{act_hours}} AS DOUBLE), 0.0), 0.0)       AS hours,
         COALESCE(CAST({{act_spend}} AS DOUBLE), 0.0)                      AS spend_raw,
         CAST({{act_platform}} AS STRING)                                  AS platform,
         CAST({{act_country}} AS STRING)                                   AS country
  FROM {{activity_table}}
  WHERE PMOD(XXHASH64(CAST({{act_user_id}} AS STRING)), {{hash_buckets}}) < {{sample_threshold}}
),
suspects AS (
  SELECT user_id FROM raw GROUP BY user_id HAVING MAX(hours) > {{max_weekly_hours}}
),
kept AS (
  SELECT r.*
  FROM raw r
  LEFT ANTI JOIN suspects s ON r.user_id = s.user_id
  {{sql_exclude_join}}
  WHERE {{sql_negative_spend_filter}}
),
agg AS (
  SELECT k.user_id,
         c.week_idx,
         SUM(k.hours)                         AS hours,
         SUM(GREATEST(k.spend_raw, 0.0))      AS spend,
         MAX_BY(k.platform, k.hours)          AS platform,
         MAX_BY(k.country, k.hours)           AS country,
         CAST(COUNT(*) AS INT)                AS n_source_rows,
         BOOL_OR(k.spend_raw < 0)             AS had_negative_spend
  FROM kept k
  JOIN {{t_calendar_weeks}} c ON c.week_start = k.week_start AND c.has_data
  GROUP BY k.user_id, c.week_idx
)
SELECT user_id, week_idx, hours, spend, platform, country, n_source_rows, had_negative_spend
FROM agg
WHERE hours > 0 OR spend > 0
