-- Purpose : Check what a row means. A row should have hours > 0 OR spend > 0, non-negative spend,
--           and a platform and country.
-- Grain   : one row (totals over the RAW activity table).
-- Inputs  : {{activity_table}}
-- Output  : n_rows, zero_zero_rows, spend_only_rows, negative_spend_rows, null_hours_rows,
--           null_spend_rows, null_platform_rows, null_country_rows, over_max_hours_rows, spend_only_share
SELECT COUNT(*)                                                                       AS n_rows,
       COUNT_IF(COALESCE({{act_hours}}, 0) <= 0 AND COALESCE({{act_spend}}, 0) = 0)   AS zero_zero_rows,
       COUNT_IF(COALESCE({{act_hours}}, 0) <= 0 AND {{act_spend}} > 0)                AS spend_only_rows,
       COUNT_IF({{act_spend}} < 0)                                                    AS negative_spend_rows,
       COUNT_IF({{act_hours}} IS NULL)                                                AS null_hours_rows,
       COUNT_IF({{act_spend}} IS NULL)                                                AS null_spend_rows,
       COUNT_IF({{act_platform}} IS NULL)                                             AS null_platform_rows,
       COUNT_IF({{act_country}} IS NULL)                                              AS null_country_rows,
       COUNT_IF({{act_hours}} > {{max_weekly_hours}})                                 AS over_max_hours_rows,
       COUNT_IF(COALESCE({{act_hours}}, 0) <= 0 AND {{act_spend}} > 0) / COUNT(*)     AS spend_only_share
FROM {{activity_table}}
