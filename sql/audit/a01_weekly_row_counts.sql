-- Purpose : Row and user counts per week, to spot gaps or pipeline outages that would look like mass churn.
-- Grain   : one row per calendar week between the first and last week of activity (weeks with no rows included).
-- Inputs  : {{activity_table}}, {{calendar_table}}
-- Output  : week_start, n_rows, n_users, hours, spend, ratio_to_trailing_median, suspect_outage
-- Read as : suspect_outage = the week has under half the rows of the median of the 8 weeks before it.
--           Rows whose week_start is not a calendar week are ignored here; audit a05 reports them.
WITH a AS (
  SELECT CAST({{act_week_start}} AS DATE) AS week_start,
         COUNT(*)                                   AS n_rows,
         COUNT(DISTINCT {{act_user_id}})            AS n_users,
         SUM(CAST({{act_hours}} AS DOUBLE))         AS hours,
         SUM(CAST({{act_spend}} AS DOUBLE))         AS spend
  FROM {{activity_table}}
  GROUP BY CAST({{act_week_start}} AS DATE)
),
c AS (
  SELECT DISTINCT CAST({{cal_week_start}} AS DATE) AS week_start FROM {{calendar_table}}
),
b AS (
  SELECT MIN(a.week_start) AS first_week, MAX(a.week_start) AS last_week
  FROM a JOIN c ON c.week_start = a.week_start
),
w AS (
  SELECT c.week_start,
         COALESCE(a.n_rows, 0) AS n_rows, COALESCE(a.n_users, 0) AS n_users,
         COALESCE(a.hours, 0.0) AS hours, COALESCE(a.spend, 0.0) AS spend
  FROM c CROSS JOIN b
  LEFT JOIN a ON a.week_start = c.week_start
  WHERE c.week_start BETWEEN b.first_week AND b.last_week
),
m AS (
  SELECT w.*,
         PERCENTILE(n_rows, 0.5) OVER (ORDER BY week_start ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING) AS trailing_median
  FROM w
)
SELECT week_start, n_rows, n_users, hours, spend,
       n_rows / trailing_median                                      AS ratio_to_trailing_median,
       COALESCE(n_rows / trailing_median < 0.5, FALSE)               AS suspect_outage
FROM m
ORDER BY week_start
