-- Purpose : Integer ids for countries, most common first. Id 0 is reserved for "unknown".
-- Grain   : one row per country value.
-- Inputs  : activity_clean
-- Output  : dim_country(country, country_id, n_rows)
SELECT country,
       CAST(ROW_NUMBER() OVER (ORDER BY n_rows DESC, country) AS INT) AS country_id,
       n_rows
FROM (
  SELECT country, COUNT(*) AS n_rows
  FROM {{t_activity_clean}}
  WHERE country IS NOT NULL
  GROUP BY country
) t
