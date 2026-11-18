"""Inputs for the sequence models, sliced from a SequenceStore at an origin.

The goal is to show the model a user's WHOLE history cheaply, because long-term behavior
(a player who lapses for six months, returns, and lapses again) is the point, and CPU cost
grows with the number of time steps.

Two-resolution input (candidate C1)
    blocks  : everything up to about 26 weeks before the origin, as 4-week blocks on a FIXED
              calendar grid (block b = weeks 4b .. 4b+3). A six-month lapse is six or seven
              empty blocks; nothing is forgotten. Because the grid does not move with the
              origin, a block's features never change once the block is complete, so the
              encoder's state after a block can be cached and carried forward.
    weekly  : the weeks after the last complete block, one step per week (26 to 29 weeks).

Episode input (candidate C1-episodes)
    tokens  : one token per play spell (a run of active weeks with no gap of 4+ weeks): its
              length, activity, hours, spend, the lapse before it, where in the season it
              started, and how long ago it ended. States lapse-and-return directly.
    weekly  : the same recent weeks as above.

Both also get static features (the trees' features, see np_features) and the projected
calendar for the 52 forecast weeks. Only weeks <= origin are read.
"""
from __future__ import annotations

import numpy as np

from ltv.data.np_features import CATEGORICAL, FEATURE_NAMES, tabular_features
from ltv.data.sequences import SequenceStore
from ltv.data.splits import project_calendar

BLOCK = 4
RECENT = 26                    # the weekly part covers at least this many weeks
MAX_WEEKLY = RECENT + BLOCK - 1
MAX_BLOCKS = 130               # ten years of history; older blocks are dropped
MAX_TOKENS = 64
SPELL_GAP = 4

F_BLOCK = 4 * BLOCK + BLOCK + 1
F_WEEK = 6
F_TOKEN = 8
F_FUTURE = 3

_CAT_IDX = [FEATURE_NAMES.index(c) for c in CATEGORICAL]
_NUM_IDX = [i for i in range(len(FEATURE_NAMES)) if i not in _CAT_IDX]
N_STATIC = 2 * len(_NUM_IDX)   # value and a missing flag for each numeric feature
VOCAB = {"platform_id": 16, "country_id": 64}


def n_complete_blocks(origin: int) -> int:
    """Number of complete blocks on the fixed grid that go to the history encoder.

    Chosen so the weekly part that follows covers RECENT to RECENT + BLOCK - 1 weeks."""
    return max((origin - RECENT + 1) // BLOCK, 0)


def _week_feats(store: SequenceStore, rows: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """[n, hi - lo, 4]: active, log1p(hours), log1p(spend), spend > 0 for weeks lo .. hi-1."""
    h = store.hours[rows, lo:hi]
    s = store.spend[rows, lo:hi]
    return np.stack([(h > 0).astype(np.float32), np.log1p(h), np.log1p(s), (s > 0).astype(np.float32)], axis=-1)


def _left_align(x: np.ndarray, start: np.ndarray, length: np.ndarray, width: int) -> np.ndarray:
    """Move each row's slice x[i, start[i] : start[i] + length[i]] to the front; zero-pad to ``width``."""
    n = x.shape[0]
    idx = start[:, None] + np.arange(width)[None, :]
    valid = np.arange(width)[None, :] < length[:, None]
    idx = np.clip(idx, 0, x.shape[1] - 1)
    out = np.take_along_axis(x, idx[:, :, None], axis=1)
    out[~valid] = 0.0
    return out


def block_inputs(store: SequenceStore, rows: np.ndarray, origin: int):
    """Blocks [n, NB, F_BLOCK] left-aligned at each user's first block, and their lengths."""
    nb = n_complete_blocks(origin)
    n = len(rows)
    if nb == 0:
        return np.zeros((n, 1, F_BLOCK), np.float32), np.zeros(n, np.int64)
    wf = _week_feats(store, rows, 0, nb * BLOCK).reshape(n, nb, BLOCK * 4)
    sw = store.season_week[: nb * BLOCK].reshape(nb, BLOCK)
    cal = np.concatenate([(sw == 1).astype(np.float32), sw[:, :1].astype(np.float32) / 10.0], axis=1)
    x = np.concatenate([wf, np.broadcast_to(cal[None], (n, nb, BLOCK + 1))], axis=-1).astype(np.float32)
    fb = store.first_week[rows] // BLOCK
    length = np.clip(nb - fb, 0, None)
    start = np.where(length > MAX_BLOCKS, nb - MAX_BLOCKS, fb)       # keep the latest blocks
    length = np.minimum(length, MAX_BLOCKS)
    width = max(int(length.max()) if n else 1, 1)
    return _left_align(x, start, length, width), length.astype(np.int64)


def weekly_inputs(store: SequenceStore, rows: np.ndarray, origin: int):
    """Recent weeks [n, MAX_WEEKLY, F_WEEK] left-aligned, and their lengths (>= 1)."""
    lo = n_complete_blocks(origin) * BLOCK
    n = len(rows)
    wf = _week_feats(store, rows, lo, origin + 1)
    sw = store.season_week[lo:origin + 1]
    cal = np.stack([sw.astype(np.float32) / 10.0, (sw == 1).astype(np.float32)], axis=-1)
    x = np.concatenate([wf, np.broadcast_to(cal[None], (n, len(sw), 2))], axis=-1).astype(np.float32)
    start = np.clip(store.first_week[rows] - lo, 0, None)
    length = (origin + 1 - lo) - start
    return _left_align(x, start, length, MAX_WEEKLY), length.astype(np.int64)


def episode_tokens(store: SequenceStore, rows: np.ndarray, origin: int, max_tokens: int = MAX_TOKENS):
    """Spell tokens [n, NT, F_TOKEN] (oldest first, latest ``max_tokens`` kept) and their counts."""
    n, t = len(rows), origin
    H = store.hours[rows, :t + 1]
    S = store.spend[rows, :t + 1]
    act = H > 0
    wk = np.arange(t + 1, dtype=np.int32)
    idx = np.where(act, wk, -1).astype(np.int32)
    prev = np.concatenate([np.full((n, 1), -1, np.int32), np.maximum.accumulate(idx, axis=1)[:, :-1]], axis=1)
    gap = np.where(act & (prev >= 0), wk - prev - 1, 0)
    start = act & ((prev < 0) | (gap >= SPELL_GAP))
    sid = np.cumsum(start, axis=1) - 1                              # spell index of each active week
    n_sp = start.sum(axis=1)
    width = max(int(n_sp.max()) if n else 1, 1)
    flat = (np.arange(n)[:, None] * width + sid)[act]
    w_act = np.broadcast_to(wk, act.shape)[act]

    def agg(values, how="sum"):
        out = np.zeros(n * width, np.float64)
        if how == "sum":
            np.add.at(out, flat, values)
        elif how == "max":
            out[:] = -1
            np.maximum.at(out, flat, values)
        return out.reshape(n, width)

    first_w = np.full(n * width, 10 ** 6, np.float64)
    np.minimum.at(first_w, flat, w_act)
    first_w = first_w.reshape(n, width)
    last_w = agg(w_act, "max")
    lapse = agg(np.where(start, gap, 0)[act])
    sw_start = store.season_week[np.clip(first_w, 0, t).astype(int)]
    exists = np.arange(width)[None, :] < n_sp[:, None]
    span = np.where(exists, last_w - first_w + 1, 0.0)             # slots past a user's last spell are empty
    since = np.where(exists, t - last_w, 0.0)
    tok = np.stack([np.log1p(span), np.log1p(agg(np.ones(len(flat)))), np.log1p(agg(H[act])),
                    np.log1p(agg(S[act])), np.log1p(lapse), sw_start / 10.0, (sw_start <= 2).astype(float),
                    np.log1p(since)], axis=-1)
    tok = np.where(exists[:, :, None], tok, 0.0).astype(np.float32)
    length = np.minimum(n_sp, max_tokens)
    return _left_align(tok, n_sp - length, length, max(int(length.max()) if n else 1, 1)), length.astype(np.int64)


def static_inputs(store: SequenceStore, rows: np.ndarray, origin: int, known_window="current_season"):
    """Numeric static features [n, N_STATIC] (compressed, with missing flags) and categorical ids [n, 2]."""
    X = tabular_features(store, rows, origin, known_window=known_window)
    num = X[:, _NUM_IDX]
    miss = np.isnan(num)
    num = np.nan_to_num(num)
    num = np.sign(num) * np.log1p(np.abs(num))                     # heavy-tailed counts and amounts
    cat = np.stack([np.clip(np.nan_to_num(X[:, i]), 0, VOCAB[c] - 1) for i, c in zip(_CAT_IDX, CATEGORICAL)],
                   axis=1).astype(np.int64)
    return np.concatenate([num, miss.astype(np.float32)], axis=1).astype(np.float32), cat


def future_inputs(store: SequenceStore, origin: int, horizon: int = 52, known_window="current_season"):
    """Projected calendar for the forecast weeks, [horizon, F_FUTURE]. The same for every user."""
    sw, known = project_calendar(store.season_week, store.season_idx, origin, horizon, known_window)
    return np.stack([sw / 10.0, (sw == 1).astype(float), known.astype(float)], axis=1).astype(np.float32)


def build_inputs(store: SequenceStore, rows: np.ndarray, origin: int, known_window="current_season",
                 tokens: bool = False) -> dict[str, np.ndarray]:
    """Everything a sequence model needs for ``rows`` at ``origin``."""
    rows = np.asarray(rows)
    out = {}
    out["blocks"], out["block_len"] = block_inputs(store, rows, origin)
    out["weekly"], out["weekly_len"] = weekly_inputs(store, rows, origin)
    if tokens:
        out["tokens"], out["token_len"] = episode_tokens(store, rows, origin)
    out["static"], out["cat"] = static_inputs(store, rows, origin, known_window)
    out["future"] = future_inputs(store, origin, known_window=known_window)
    return out
