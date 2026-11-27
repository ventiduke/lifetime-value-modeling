-- Purpose : Tabular features for the trees, built straight from the SPARSE activity table.
-- Grain   : one row per user x origin (same keys as user_origins).
-- Inputs  : user_origins, activity_clean, calendar_weeks, calendar_projected, dim_platform, dim_country
-- Output  : features_tabular(user_id, origin_week, split, <features>)
-- Leakage : every activity row joined here has week_idx <= origin_week. The only future
--           information is the PROJECTED calendar (calendar_projected), which is built from
--           what was known at the origin (docs/DESIGN.md, "Future-calendar rule").
-- Oracle  : src/ltv/data/np_features.py computes the same features from dense arrays.
--           tests/test_features.py checks the two agree. Change both together.
-- Fixed   : windows 4/13/26/52/104 and lapse thresholds 4/13/26 are written out below; they
--           must match feature_windows and lapse_thresholds in the config (validated in Python).
--
-- Definitions
--   active week : hours > 0. Spend-only weeks are not active.
--   gap         : for an active week, the number of inactive weeks since the previous active week.
--   lapse       : a gap of at least 4 weeks. A lapse that is followed by an active week has "ended in a return".
--   spell       : a run of active weeks with no gap of 4 or more inside it.
WITH act AS (
  SELECT user_id, week_idx,
         LAG(week_idx) OVER (PARTITION BY user_id ORDER BY week_idx) AS prev_active
  FROM {{t_activity_clean}}
  WHERE hours > 0
),
r AS (
  SELECT a.user_id, a.week_idx, a.hours, a.spend,
         p.platform_id, k.country_id,
         c.season_week, c.season_idx,
         g.prev_active,
         CASE WHEN a.hours > 0 THEN a.week_idx - g.prev_active - 1 END                              AS gap,
         (a.hours > 0 AND (g.prev_active IS NULL OR a.week_idx - g.prev_active - 1 >= 4))           AS spell_start
  FROM {{t_activity_clean}} a
  JOIN {{t_calendar_weeks}} c ON c.week_idx = a.week_idx
  LEFT JOIN act g ON g.user_id = a.user_id AND g.week_idx = a.week_idx
  LEFT JOIN {{t_dim_platform}} p ON p.platform = a.platform
  LEFT JOIN {{t_dim_country}}  k ON k.country  = a.country
),
agg AS (
  SELECT
    uo.user_id, uo.origin_week, uo.split, uo.tenure_weeks,
    MAX(CASE WHEN r.hours > 0 THEN r.week_idx END)                                  AS last_active,
    MAX(CASE WHEN r.spend > 0 THEN r.week_idx END)                                  AS last_spend,
    -- frequency and level over trailing windows (origin - w, origin]
    COUNT_IF(r.hours > 0 AND r.week_idx > uo.origin_week - 4)                       AS aw_4,
    COUNT_IF(r.hours > 0 AND r.week_idx > uo.origin_week - 13)                      AS aw_13,
    COUNT_IF(r.hours > 0 AND r.week_idx > uo.origin_week - 26)                      AS aw_26,
    COUNT_IF(r.hours > 0 AND r.week_idx > uo.origin_week - 52)                      AS aw_52,
    COUNT_IF(r.hours > 0 AND r.week_idx > uo.origin_week - 104)                     AS aw_104,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 4   THEN r.hours END), 0.0) AS hrs_4,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 13  THEN r.hours END), 0.0) AS hrs_13,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 26  THEN r.hours END), 0.0) AS hrs_26,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 52  THEN r.hours END), 0.0) AS hrs_52,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 104 THEN r.hours END), 0.0) AS hrs_104,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 4   THEN r.spend END), 0.0) AS spd_4,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 13  THEN r.spend END), 0.0) AS spd_13,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 26  THEN r.spend END), 0.0) AS spd_26,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 52  THEN r.spend END), 0.0) AS spd_52,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 104 THEN r.spend END), 0.0) AS spd_104,
    COUNT_IF(r.spend > 0 AND r.week_idx > uo.origin_week - 4)                       AS sw_4,
    COUNT_IF(r.spend > 0 AND r.week_idx > uo.origin_week - 13)                      AS sw_13,
    COUNT_IF(r.spend > 0 AND r.week_idx > uo.origin_week - 26)                      AS sw_26,
    COUNT_IF(r.spend > 0 AND r.week_idx > uo.origin_week - 52)                      AS sw_52,
    COUNT_IF(r.spend > 0 AND r.week_idx > uo.origin_week - 104)                     AS sw_104,
    -- lifetime
    COUNT_IF(r.hours > 0)                                                           AS aw_life,
    COALESCE(SUM(r.hours), 0.0)                                                     AS hrs_life,
    COALESCE(SUM(r.spend), 0.0)                                                     AS spd_life,
    COUNT_IF(r.spend > 0)                                                           AS sw_life,
    COALESCE(MAX(CASE WHEN r.week_idx > uo.origin_week - 52 THEN r.hours END), 0.0) AS max_hrs_52,
    -- trend
    COUNT_IF(r.hours > 0 AND r.week_idx > uo.origin_week - 17
                         AND r.week_idx <= uo.origin_week - 4)                      AS aw_prior13,
    COALESCE(SUM(CASE WHEN r.week_idx > uo.origin_week - 26
                      THEN (r.week_idx - (uo.origin_week - 25) - 12.5) * r.hours END), 0.0) / 1462.5
                                                                                    AS hrs_slope_26,
    -- seasonality
    COUNT_IF(r.hours > 0 AND r.season_week <= 2)                                    AS aw_launch,
    COUNT(DISTINCT CASE WHEN r.hours > 0 THEN r.season_idx END)                     AS n_seasons_active,
    COUNT_IF(r.hours > 0 AND r.week_idx > uo.origin_week - 52
                         AND r.week_idx <= uo.origin_week - 39)                     AS aw_next13_1y,
    COUNT_IF(r.hours > 0 AND r.week_idx > uo.origin_week - 104
                         AND r.week_idx <= uo.origin_week - 91)                     AS aw_next13_2y,
    -- lapse and return
    COUNT_IF(r.gap >= 4)                                                            AS n_returns_4,
    COUNT_IF(r.gap >= 13)                                                           AS n_returns_13,
    COUNT_IF(r.gap >= 26)                                                           AS n_returns_26,
    PERCENTILE(CASE WHEN r.gap >= 4 THEN r.gap END, 0.5)                            AS median_lapse,
    COALESCE(MAX(r.gap), 0)                                                         AS max_lapse,
    COUNT_IF(r.spell_start)                                                         AS n_spells,
    COALESCE(SUM(CASE WHEN r.hours > 0 AND NOT r.spell_start
                      THEN r.week_idx - r.prev_active END), 0)                      AS spell_span_extra,
    MAX(CASE WHEN r.spell_start THEN r.week_idx END)                                AS last_return_week,
    COUNT_IF(r.gap >= 4 AND r.season_week <= 2)                                     AS n_returns_launch,
    -- context
    MAX_BY(r.platform_id, CASE WHEN r.platform_id IS NOT NULL THEN r.week_idx END)  AS platform_id,
    MAX_BY(r.country_id,  CASE WHEN r.country_id  IS NOT NULL THEN r.week_idx END)  AS country_id,
    COUNT(DISTINCT r.platform_id)                                                   AS n_platforms
  FROM {{t_user_origins}} uo
  JOIN r ON r.user_id = uo.user_id AND r.week_idx <= uo.origin_week
  GROUP BY uo.user_id, uo.origin_week, uo.split, uo.tenure_weeks
),
pc AS (
  SELECT origin_week,
         MIN(CASE WHEN is_launch_proj THEN h END)        AS wk_to_next_launch,
         COUNT_IF(is_launch_proj AND h <= 13)            AS launches_q1,
         COUNT_IF(is_launch_proj AND h > 13 AND h <= 26) AS launches_q2,
         COUNT_IF(is_launch_proj AND h > 26 AND h <= 39) AS launches_q3,
         COUNT_IF(is_launch_proj AND h > 39)             AS launches_q4
  FROM {{t_calendar_projected}}
  GROUP BY origin_week
)
SELECT
  a.user_id,
  a.origin_week,
  a.split,
  -- recency
  CAST(a.tenure_weeks AS DOUBLE)                                              AS tenure_weeks,
  CAST(a.origin_week - a.last_active AS DOUBLE)                               AS wk_since_active,
  CAST(a.origin_week - a.last_spend AS DOUBLE)                                AS wk_since_spend,
  -- frequency and level
  CAST(a.aw_4 AS DOUBLE) AS aw_4, CAST(a.aw_13 AS DOUBLE) AS aw_13, CAST(a.aw_26 AS DOUBLE) AS aw_26,
  CAST(a.aw_52 AS DOUBLE) AS aw_52, CAST(a.aw_104 AS DOUBLE) AS aw_104,
  a.hrs_4, a.hrs_13, a.hrs_26, a.hrs_52, a.hrs_104,
  a.spd_4, a.spd_13, a.spd_26, a.spd_52, a.spd_104,
  CAST(a.sw_4 AS DOUBLE) AS sw_4, CAST(a.sw_13 AS DOUBLE) AS sw_13, CAST(a.sw_26 AS DOUBLE) AS sw_26,
  CAST(a.sw_52 AS DOUBLE) AS sw_52, CAST(a.sw_104 AS DOUBLE) AS sw_104,
  CAST(a.aw_life AS DOUBLE) AS aw_life, a.hrs_life, a.spd_life, CAST(a.sw_life AS DOUBLE) AS sw_life,
  a.max_hrs_52,
  -- trend
  (CAST(a.aw_4 AS DOUBLE) / 4.0) / ((CAST(a.aw_prior13 AS DOUBLE) + 0.5) / 13.0)  AS ratio_aw_4_13,
  a.hrs_slope_26,
  -- seasonality
  CASE WHEN a.aw_life > 0 THEN a.aw_launch / CAST(a.aw_life AS DOUBLE) END    AS share_aw_launch,
  CAST(a.n_seasons_active AS DOUBLE)                                          AS n_seasons_active,
  CAST(a.aw_next13_1y AS DOUBLE)                                              AS aw_next13_1y,
  CAST(a.aw_next13_2y AS DOUBLE)                                              AS aw_next13_2y,
  CAST(c.season_week AS DOUBLE)                                               AS origin_season_week,
  -- lapse and return
  CAST(a.n_returns_4 AS DOUBLE)                                               AS n_returns_4,
  CAST(a.n_returns_13 AS DOUBLE)                                              AS n_returns_13,
  CAST(a.n_returns_26 AS DOUBLE)                                              AS n_returns_26,
  CAST(a.median_lapse AS DOUBLE)                                              AS median_lapse,
  CAST(a.max_lapse AS DOUBLE)                                                 AS max_lapse,
  CASE WHEN a.median_lapse > 0
       THEN (a.origin_week - a.last_active) / CAST(a.median_lapse AS DOUBLE) END AS cur_lapse_rel,
  CAST(a.n_spells AS DOUBLE)                                                  AS n_spells,
  CASE WHEN a.n_spells > 0
       THEN (a.spell_span_extra + a.n_spells) / CAST(a.n_spells AS DOUBLE) END AS mean_spell_len,
  CAST(a.origin_week - a.last_return_week AS DOUBLE)                          AS wk_since_return,
  CASE WHEN a.n_returns_4 > 0
       THEN a.n_returns_launch / CAST(a.n_returns_4 AS DOUBLE) END            AS share_returns_launch,
  -- context
  CAST(COALESCE(a.platform_id, 0) AS DOUBLE)                                  AS platform_id,
  CAST(COALESCE(a.country_id, 0) AS DOUBLE)                                   AS country_id,
  CAST(a.n_platforms AS DOUBLE)                                               AS n_platforms,
  -- horizon context, from the projected calendar only
  CAST(pc.wk_to_next_launch AS DOUBLE)                                        AS wk_to_next_launch,
  CAST(pc.launches_q1 AS DOUBLE) AS launches_q1, CAST(pc.launches_q2 AS DOUBLE) AS launches_q2,
  CAST(pc.launches_q3 AS DOUBLE) AS launches_q3, CAST(pc.launches_q4 AS DOUBLE) AS launches_q4
FROM agg a
JOIN {{t_calendar_weeks}} c ON c.week_idx = a.origin_week
LEFT JOIN pc ON pc.origin_week = a.origin_week
