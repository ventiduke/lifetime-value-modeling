-- Purpose : Dense user x week panel for the user SAMPLE. A convenience for analysis and the
--           test oracle for the sparse feature path. Full-scale scoring never builds this.
-- Grain   : one row per user x week, from the user's first week to the last week of data.
-- Inputs  : users, activity_clean, calendar_weeks
-- Output  : panel_weekly(user_id, week_idx, hours, spend, active, platform, country, platform_carried,
--                        tenure_weeks, weeks_since_last_active, weeks_since_last_spend, active_streak,
--                        season_idx, season_week, is_launch)
-- Notes   : * Missing weeks are 0 hours and 0 spend. active = hours > 0.
--           * platform and country are carried forward from the last week that had them;
--             platform_carried marks carried values. Only past weeks are read (no leakage).
--           * weeks_since_last_* is 0 in a week with the event and NULL before the first one.
WITH lim AS (
  SELECT MAX(week_idx) AS last_data_week FROM {{t_calendar_weeks}} WHERE has_data
),
grid AS (
  SELECT u.user_id, u.first_week, EXPLODE(SEQUENCE(u.first_week, lim.last_data_week)) AS week_idx
  FROM {{t_users}} u CROSS JOIN lim
),
j AS (
  SELECT g.user_id, g.week_idx, g.first_week,
         COALESCE(a.hours, 0.0)      AS hours,
         COALESCE(a.spend, 0.0)      AS spend,
         COALESCE(a.hours, 0.0) > 0  AS active,
         a.platform                  AS platform_raw,
         a.country                   AS country_raw
  FROM grid g
  LEFT JOIN {{t_activity_clean}} a ON a.user_id = g.user_id AND a.week_idx = g.week_idx
),
w AS (
  SELECT j.*,
         LAST(CASE WHEN active THEN week_idx END, TRUE)      OVER win AS last_active_week,
         LAST(CASE WHEN spend > 0 THEN week_idx END, TRUE)   OVER win AS last_spend_week,
         LAST(CASE WHEN NOT active THEN week_idx END, TRUE)  OVER win AS last_inactive_week,
         LAST(platform_raw, TRUE)                            OVER win AS platform,
         LAST(country_raw, TRUE)                             OVER win AS country
  FROM j
  WINDOW win AS (PARTITION BY user_id ORDER BY week_idx
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
)
SELECT w.user_id,
       w.week_idx,
       w.hours,
       w.spend,
       w.active,
       w.platform,
       w.country,
       w.platform_raw IS NULL                                AS platform_carried,
       CAST(w.week_idx - w.first_week AS INT)                AS tenure_weeks,
       CAST(w.week_idx - w.last_active_week AS INT)          AS weeks_since_last_active,
       CAST(w.week_idx - w.last_spend_week AS INT)           AS weeks_since_last_spend,
       CAST(CASE WHEN w.active
                 THEN w.week_idx - COALESCE(w.last_inactive_week, w.first_week - 1)
                 ELSE 0 END AS INT)                          AS active_streak,
       c.season_idx,
       c.season_week,
       c.is_launch
FROM w
JOIN {{t_calendar_weeks}} c ON c.week_idx = w.week_idx
