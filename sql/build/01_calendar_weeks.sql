-- Purpose : Number the calendar's weeks. week_idx 0 is the first week that has activity.
-- Grain   : one row per calendar week (weeks after the last data week are kept, has_data = false).
-- Inputs  : {{calendar_table}}, {{activity_table}} (only to find the first and last data week).
-- Output  : calendar_weeks(week_idx, week_start, season, season_idx, season_week, is_launch, has_data)
-- Notes   : A calendar week that is not a whole number of weeks after the first week is dropped.
--           Activity rows whose week_start is not a calendar week are dropped in 02_activity_clean.
WITH cal AS (
  SELECT CAST({{cal_week_start}} AS DATE)   AS week_start,
         CAST({{cal_season}} AS STRING)     AS season,
         CAST({{cal_season_week}} AS INT)   AS season_week
  FROM {{calendar_table}}
),
act AS (
  SELECT DISTINCT CAST({{act_week_start}} AS DATE) AS week_start FROM {{activity_table}}
),
bounds AS (
  SELECT MIN(c.week_start) AS first_week, MAX(c.week_start) AS last_week
  FROM cal c JOIN act a ON a.week_start = c.week_start
),
seasons AS (
  SELECT season, MIN(week_start) AS season_start FROM cal GROUP BY season
)
SELECT
  CAST(DATEDIFF(c.week_start, b.first_week) DIV 7 AS INT)        AS week_idx,
  c.week_start,
  c.season,
  CAST(DENSE_RANK() OVER (ORDER BY s.season_start) - 1 AS INT)   AS season_idx,
  c.season_week,
  c.season_week = 1                                              AS is_launch,
  c.week_start <= b.last_week                                    AS has_data
FROM cal c
CROSS JOIN bounds b
JOIN seasons s ON s.season = c.season
WHERE c.week_start >= b.first_week
  AND PMOD(DATEDIFF(c.week_start, b.first_week), 7) = 0
