-- Purpose : The user x origin pairs that features and labels are built for.
-- Grain   : one row per user x origin, for users in the population at that origin.
-- Inputs  : users, origins, activity_clean
-- Output  : user_origins(user_id, origin_week, split, is_eval_origin, tenure_weeks)
-- Rules   : * Population at origin t: at least one active or spend week in (t - lookback, t].
--           * Evaluation origins (validation, test_1, test_2) keep every such user.
--           * Training-only origins keep a hashed subset of users, so each user contributes
--             about splits.max_train_origins_per_user origins ({{train_origin_keep_permille}} per mille kept).
WITH o AS (
  SELECT origin_week,
         BOOL_OR(role IN ('validation', 'test_1', 'test_2')) AS is_eval_origin
  FROM {{t_origins}}
  GROUP BY origin_week
),
uw AS (
  SELECT user_id, SORT_ARRAY(COLLECT_LIST(week_idx)) AS weeks
  FROM {{t_activity_clean}}
  GROUP BY user_id
)
SELECT u.user_id,
       o.origin_week,
       u.split,
       o.is_eval_origin,
       CAST(o.origin_week - u.first_week AS INT) AS tenure_weeks
FROM {{t_users}} u
JOIN uw ON uw.user_id = u.user_id
CROSS JOIN o
WHERE u.first_week <= o.origin_week
  AND EXISTS(uw.weeks, w -> w <= o.origin_week AND w > o.origin_week - {{lookback}})
  AND (o.is_eval_origin
       OR PMOD(XXHASH64(u.user_id, o.origin_week), 1000) < {{train_origin_keep_permille}})
