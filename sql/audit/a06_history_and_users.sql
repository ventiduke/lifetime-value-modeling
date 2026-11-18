-- Purpose : How much history there is and how many users. Sets the split schedule and the sample size.
-- Grain   : one row.
-- Inputs  : {{activity_table}}, {{calendar_table}}
-- Output  : first_week, last_week, weeks_of_history, distinct_weeks, n_users_all_time,
--           n_users_active_last_104w, share_users_first_seen_in_first_week
-- Read as : weeks_of_history should exceed 312. distinct_weeks < weeks_of_history means missing weeks.
--           A large share of users first seen in the first data week suggests the data does NOT
--           start at product launch (those users would be left-censored).
--           Only rows on a calendar week are counted; audit a05 reports the rest.
WITH a AS (
  SELECT x.{{act_user_id}} AS user_id, CAST(x.{{act_week_start}} AS DATE) AS week_start
  FROM {{activity_table}} x
  LEFT SEMI JOIN {{calendar_table}} c ON CAST(c.{{cal_week_start}} AS DATE) = CAST(x.{{act_week_start}} AS DATE)
),
u AS (
  SELECT user_id, MIN(week_start) AS first_seen, MAX(week_start) AS last_seen FROM a GROUP BY user_id
),
b AS (
  SELECT MIN(week_start) AS first_week, MAX(week_start) AS last_week,
         COUNT(DISTINCT week_start) AS distinct_weeks
  FROM a
)
SELECT b.first_week, b.last_week,
       CAST(DATEDIFF(b.last_week, b.first_week) DIV 7 + 1 AS INT)            AS weeks_of_history,
       b.distinct_weeks,
       COUNT(*)                                                              AS n_users_all_time,
       COUNT_IF(u.last_seen > DATE_SUB(b.last_week, 7 * {{lookback}}))       AS n_users_active_last_104w,
       COUNT_IF(u.first_seen = b.first_week) / COUNT(*)                      AS share_users_first_seen_in_first_week
FROM u CROSS JOIN b
GROUP BY b.first_week, b.last_week, b.distinct_weeks
