-- Purpose : Each user's whole weekly history as dense arrays. Input to the HMM and the sequence models.
-- Grain   : one row per user.
-- Inputs  : users, activity_clean, dim_platform, dim_country, calendar_weeks
-- Output  : user_sequences(user_id, split, first_week, last_data_week, hours, spend, platform_id, country_id)
-- Notes   : * Arrays run from the user's first_week to the last week of data; element i is week first_week + i.
--           * Weeks with no row are 0 hours, 0 spend, platform_id 0 and country_id 0 (unknown).
--             Loaders carry platform and country forward; the raw arrays do not.
--           * Calendar channels (season week, launch flag) are NOT stored per user: they are
--             the same for everyone and come from calendar_weeks.
--           * Loaders slice these arrays at an origin, so features and labels come from one source.
WITH lim AS (
  SELECT MAX(week_idx) AS last_data_week FROM {{t_calendar_weeks}} WHERE has_data
),
m AS (
  SELECT a.user_id,
         MAP_FROM_ENTRIES(COLLECT_LIST(STRUCT(a.week_idx, CAST(a.hours AS FLOAT))))                          AS h,
         MAP_FROM_ENTRIES(COLLECT_LIST(STRUCT(a.week_idx, CAST(a.spend AS FLOAT))))                          AS s,
         MAP_FROM_ENTRIES(COLLECT_LIST(STRUCT(a.week_idx, CAST(COALESCE(p.platform_id, 0) AS SMALLINT))))    AS p,
         MAP_FROM_ENTRIES(COLLECT_LIST(STRUCT(a.week_idx, CAST(COALESCE(c.country_id, 0) AS SMALLINT))))     AS c
  FROM {{t_activity_clean}} a
  LEFT JOIN {{t_dim_platform}} p ON p.platform = a.platform
  LEFT JOIN {{t_dim_country}}  c ON c.country  = a.country
  GROUP BY a.user_id
)
SELECT u.user_id,
       u.split,
       u.first_week,
       lim.last_data_week,
       TRANSFORM(SEQUENCE(u.first_week, lim.last_data_week),
                 w -> COALESCE(TRY_ELEMENT_AT(m.h, w), CAST(0 AS FLOAT)))    AS hours,
       TRANSFORM(SEQUENCE(u.first_week, lim.last_data_week),
                 w -> COALESCE(TRY_ELEMENT_AT(m.s, w), CAST(0 AS FLOAT)))    AS spend,
       TRANSFORM(SEQUENCE(u.first_week, lim.last_data_week),
                 w -> COALESCE(TRY_ELEMENT_AT(m.p, w), CAST(0 AS SMALLINT))) AS platform_id,
       TRANSFORM(SEQUENCE(u.first_week, lim.last_data_week),
                 w -> COALESCE(TRY_ELEMENT_AT(m.c, w), CAST(0 AS SMALLINT))) AS country_id
FROM {{t_users}} u
JOIN m ON m.user_id = u.user_id
CROSS JOIN lim
