-- Purpose : Season lengths, and how far the calendar runs past the data. Sets the future-calendar rule.
-- Grain   : one row per season.
-- Inputs  : {{calendar_table}}, {{activity_table}}
-- Output  : season, season_start, n_weeks, max_season_week, starts_after_last_data_week, weeks_known_ahead
-- Read as : n_weeks should equal max_season_week (no gaps inside a season). weeks_known_ahead is how
--           many calendar weeks exist beyond the last data week TODAY. It does not tell you how far
--           ahead launches were known at PAST origins; ask the calendar's owners, then set
--           calendar.known_window (docs/DESIGN.md, "Future-calendar rule").
WITH b AS (
  SELECT MAX(CAST({{act_week_start}} AS DATE)) AS last_data_week FROM {{activity_table}}
),
c AS (
  SELECT CAST({{cal_week_start}} AS DATE) AS week_start, CAST({{cal_season}} AS STRING) AS season,
         CAST({{cal_season_week}} AS INT) AS season_week
  FROM {{calendar_table}}
)
SELECT c.season,
       MIN(c.week_start)                                              AS season_start,
       COUNT(*)                                                       AS n_weeks,
       MAX(c.season_week)                                             AS max_season_week,
       MIN(c.week_start) > MAX(b.last_data_week)                      AS starts_after_last_data_week,
       CAST(DATEDIFF(MAX(MAX(c.week_start)) OVER (), MAX(b.last_data_week)) DIV 7 AS INT) AS weeks_known_ahead
FROM c CROSS JOIN b
GROUP BY c.season
ORDER BY season_start
