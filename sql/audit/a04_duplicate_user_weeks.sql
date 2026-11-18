-- Purpose : A user should have at most one row per week, with one platform and one country.
-- Grain   : one row (totals).
-- Inputs  : {{activity_table}}
-- Output  : n_user_weeks, duplicate_user_weeks, multi_platform_user_weeks, multi_country_user_weeks
-- Read as : if duplicates exist, 02_activity_clean sums them and takes platform and country from
--           the row with the most hours. Confirm that rule is right for the real data.
WITH g AS (
  SELECT {{act_user_id}} AS user_id, CAST({{act_week_start}} AS DATE) AS week_start,
         COUNT(*) AS n, COUNT(DISTINCT {{act_platform}}) AS n_platform, COUNT(DISTINCT {{act_country}}) AS n_country
  FROM {{activity_table}}
  GROUP BY {{act_user_id}}, CAST({{act_week_start}} AS DATE)
)
SELECT COUNT(*)                  AS n_user_weeks,
       COUNT_IF(n > 1)           AS duplicate_user_weeks,
       COUNT_IF(n_platform > 1)  AS multi_platform_user_weeks,
       COUNT_IF(n_country > 1)   AS multi_country_user_weeks
FROM g
