-- Purpose : Find accounts that cannot be real players (bots, test accounts): impossible weekly hours.
-- Grain   : one row per suspect user (top 100 by maximum weekly hours).
-- Inputs  : {{activity_table}}
-- Output  : user_id, max_weekly_hours, n_weeks, total_hours, total_spend
-- Read as : every user listed is DROPPED by 02_activity_clean. If the list is long or contains real
--           players, change cleaning.max_weekly_hours. Banned and merged accounts cannot be seen
--           here: supply them through source.exclude_users_table.
SELECT {{act_user_id}}                              AS user_id,
       MAX(CAST({{act_hours}} AS DOUBLE))           AS max_weekly_hours,
       COUNT(*)                                     AS n_weeks,
       SUM(CAST({{act_hours}} AS DOUBLE))           AS total_hours,
       SUM(CAST({{act_spend}} AS DOUBLE))           AS total_spend
FROM {{activity_table}}
GROUP BY {{act_user_id}}
HAVING MAX(CAST({{act_hours}} AS DOUBLE)) > {{max_weekly_hours}}
ORDER BY max_weekly_hours DESC
LIMIT 100
