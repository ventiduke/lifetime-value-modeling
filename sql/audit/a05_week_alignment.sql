-- Purpose : The week definition (start day) must match between the activity table and the calendar.
-- Grain   : one row per weekday on which activity week_start values fall.
-- Inputs  : {{activity_table}}, {{calendar_table}}
-- Output  : weekday (1 = Sunday .. 7 = Saturday), n_rows, n_rows_not_in_calendar
-- Read as : there should be exactly one weekday, with n_rows_not_in_calendar = 0. Rows not in the
--           calendar are DROPPED by 02_activity_clean. Time zone cannot be checked from the data:
--           confirm it with the table owners (serverless sessions default to UTC).
WITH a AS (
  SELECT CAST({{act_week_start}} AS DATE) AS week_start, COUNT(*) AS n_rows
  FROM {{activity_table}} GROUP BY CAST({{act_week_start}} AS DATE)
),
c AS (
  SELECT DISTINCT CAST({{cal_week_start}} AS DATE) AS week_start FROM {{calendar_table}}
)
SELECT DAYOFWEEK(a.week_start)                                        AS weekday,
       SUM(a.n_rows)                                                  AS n_rows,
       SUM(CASE WHEN c.week_start IS NULL THEN a.n_rows ELSE 0 END)   AS n_rows_not_in_calendar
FROM a LEFT JOIN c ON c.week_start = a.week_start
GROUP BY DAYOFWEEK(a.week_start)
ORDER BY n_rows DESC
