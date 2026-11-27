-- Purpose : One row per sampled user, with the week tenure starts and the user's hash split.
-- Grain   : one row per user.
-- Inputs  : activity_clean
-- Output  : users(user_id, first_week, last_week, first_active_week, n_active_weeks, n_spend_weeks, split)
-- Notes   : split is 'train' / 'val' / 'test' by hash of user_id (splits.user_split_pct), so a
--           user is in exactly one split at every origin. first_week is the first week with
--           any row (active or spend-only).
SELECT user_id,
       CAST(MIN(week_idx) AS INT)                              AS first_week,
       CAST(MAX(week_idx) AS INT)                              AS last_week,
       CAST(MIN(CASE WHEN hours > 0 THEN week_idx END) AS INT) AS first_active_week,
       CAST(COUNT_IF(hours > 0) AS INT)                        AS n_active_weeks,
       CAST(COUNT_IF(spend > 0) AS INT)                        AS n_spend_weeks,
       CASE WHEN PMOD(XXHASH64(user_id, 'split'), 100) < {{split_train_pct}} THEN 'train'
            WHEN PMOD(XXHASH64(user_id, 'split'), 100) < {{split_val_pct}}   THEN 'val'
            ELSE 'test' END                                    AS split
FROM {{t_activity_clean}}
GROUP BY user_id
