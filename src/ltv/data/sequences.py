"""In-memory store of user histories, built from the ``user_sequences`` table.

A ``SequenceStore`` holds one shard of users as dense ``[n_users, n_weeks]`` arrays, indexed by
global ``week_idx`` (weeks before a user's first week are zeros). Everything a sequence-based
model needs at an origin is *sliced* from these arrays: inputs from weeks <= origin, labels
from weeks origin+1 .. origin+H. One source for both is what keeps features and labels from
drifting apart; tests/test_sequences.py checks the slices against the SQL ``labels`` table.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CUM_HORIZONS = (4, 13, 26, 39, 52)


@dataclass
class SequenceStore:
    user_ids: np.ndarray          # [n] str
    split: np.ndarray             # [n] 'train' / 'val' / 'test'
    first_week: np.ndarray        # [n] int32
    hours: np.ndarray             # [n, W] float32
    spend: np.ndarray             # [n, W] float32
    platform: np.ndarray          # [n, W] int16, 0 = no row that week
    country: np.ndarray           # [n, W] int16, 0 = no row that week
    season_week: np.ndarray       # [Wcal] int16 (the calendar may run past the data)
    season_idx: np.ndarray        # [Wcal] int16

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_weeks(self) -> int:
        return self.hours.shape[1]

    @property
    def last_week(self) -> int:
        return self.n_weeks - 1

    # -- construction -------------------------------------------------------------
    @classmethod
    def from_frames(cls, seq: pd.DataFrame, calendar: pd.DataFrame) -> "SequenceStore":
        """``seq`` is user_sequences as pandas; ``calendar`` is calendar_weeks as pandas."""
        cal = calendar.sort_values("week_idx")
        if not (cal["week_idx"].to_numpy() == np.arange(len(cal))).all():
            raise ValueError("calendar_weeks has gaps in week_idx; fix the calendar before modeling")
        n = len(seq)
        W = int(seq["last_data_week"].iloc[0]) + 1 if n else int(cal["has_data"].sum())
        first = seq["first_week"].to_numpy(dtype=np.int32)
        store = cls(
            user_ids=seq["user_id"].to_numpy(dtype=object).astype(str),
            split=seq["split"].to_numpy(dtype=object).astype(str),
            first_week=first,
            hours=np.zeros((n, W), np.float32), spend=np.zeros((n, W), np.float32),
            platform=np.zeros((n, W), np.int16), country=np.zeros((n, W), np.int16),
            season_week=cal["season_week"].to_numpy(dtype=np.int16),
            season_idx=cal["season_idx"].to_numpy(dtype=np.int16),
        )
        for col, dst in (("hours", store.hours), ("spend", store.spend),
                         ("platform_id", store.platform), ("country_id", store.country)):
            for i, arr in enumerate(seq[col].to_numpy()):
                dst[i, first[i]:first[i] + len(arr)] = arr
        return store

    def save(self, path: str | Path) -> None:
        """Write a compressed .npz through local disk (safe on Unity Catalog volumes; see ltv.io)."""
        from ltv.io import write_via_local
        write_via_local(path, lambda p: np.savez_compressed(
            p, user_ids=self.user_ids, split=self.split, first_week=self.first_week, hours=self.hours,
            spend=self.spend, platform=self.platform, country=self.country,
            season_week=self.season_week, season_idx=self.season_idx))

    @classmethod
    def load(cls, path: str | Path) -> "SequenceStore":
        z = np.load(path, allow_pickle=False)
        return cls(**{k: z[k] for k in z.files})

    def subset(self, rows: np.ndarray) -> "SequenceStore":
        return SequenceStore(self.user_ids[rows], self.split[rows], self.first_week[rows],
                             self.hours[rows], self.spend[rows], self.platform[rows],
                             self.country[rows], self.season_week, self.season_idx)

    # -- membership ---------------------------------------------------------------
    def population(self, origin: int, lookback: int = 104) -> np.ndarray:
        """Row indices of users in the population at ``origin``: any active or spend week in (origin - lookback, origin]."""
        lo = max(origin - lookback + 1, 0)
        recent = ((self.hours[:, lo:origin + 1] > 0) | (self.spend[:, lo:origin + 1] > 0)).any(axis=1)
        return np.flatnonzero(recent & (self.first_week <= origin))

    def rows_for(self, origin: int, split: str | None = None, lookback: int = 104) -> np.ndarray:
        rows = self.population(origin, lookback)
        if split is not None:
            rows = rows[self.split[rows] == split]
        return rows

    # -- labels -------------------------------------------------------------------
    def labels(self, rows: np.ndarray, origin: int, horizon: int = 52) -> dict[str, np.ndarray]:
        """The five targets and sub-horizon breakdowns, sliced from weeks origin+1 .. origin+horizon."""
        if origin + horizon > self.last_week:
            raise ValueError(f"origin {origin} has no full {horizon}-week label window")
        h = self.hours[rows, origin + 1:origin + 1 + horizon]
        s = self.spend[rows, origin + 1:origin + 1 + horizon]
        act = h > 0
        ch, cs = np.cumsum(h, axis=1, dtype=np.float64), np.cumsum(s, axis=1, dtype=np.float64)
        idx = [c - 1 for c in CUM_HORIZONS if c <= horizon]
        return {
            "any_active_52": act.any(axis=1).astype(np.float32),
            "active_weeks_52": act.sum(axis=1).astype(np.float32),
            "hours_52": ch[:, -1].astype(np.float32),
            "any_spend_52": (s > 0).any(axis=1).astype(np.float32),
            "spend_52": cs[:, -1].astype(np.float32),
            "cum_hours": ch[:, idx].astype(np.float32),
            "cum_spend": cs[:, idx].astype(np.float32),
            "weekly_active": act.astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Reading the table in shards
# ---------------------------------------------------------------------------

def load_store(spark, cfg, shard: int | None = None, n_shards: int = 1) -> SequenceStore:
    """Read ``user_sequences`` (or one hash shard of it) into a SequenceStore."""
    seq = spark.table(cfg.table("user_sequences"))
    if n_shards > 1:
        seq = seq.where(f"PMOD(XXHASH64(user_id, 'shard'), {int(n_shards)}) = {int(shard)}")
    cal = spark.table(cfg.table("calendar_weeks")).toPandas()
    return SequenceStore.from_frames(seq.toPandas(), cal)


def n_shards_for(n_users: int, shard_users: int) -> int:
    return max(1, int(np.ceil(n_users / max(shard_users, 1))))


def export_shards(spark, cfg, out_dir: str | Path) -> list[Path]:
    """Write the sample as NumPy shards to ``out_dir`` and return their paths.

    Training reads one shard at a time, so memory stays flat as the sample grows.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_users = spark.table(cfg.table("user_sequences")).count()
    k = n_shards_for(n_users, int(cfg["training"]["seq_shard_users"]))
    paths = []
    for i in range(k):
        p = out / f"shard_{i:04d}.npz"
        load_store(spark, cfg, i, k).save(p)
        paths.append(p)
    return paths
