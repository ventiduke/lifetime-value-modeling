-- Purpose : Integer ids for platforms, most common first. Id 0 is reserved for "unknown".
-- Grain   : one row per platform value.
-- Inputs  : activity_clean
-- Output  : dim_platform(platform, platform_id, n_rows)
SELECT platform,
       CAST(ROW_NUMBER() OVER (ORDER BY n_rows DESC, platform) AS INT) AS platform_id,
       n_rows
FROM (
  SELECT platform, COUNT(*) AS n_rows
  FROM {{t_activity_clean}}
  WHERE platform IS NOT NULL
  GROUP BY platform
) t
