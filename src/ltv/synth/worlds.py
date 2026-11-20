"""Synthetic worlds for the simulation study, generated chunk by chunk into sequence shards.

Two processes with the same output format, so the bake-off code cannot tell them apart:

``markov``  The generator of ``ltv.synth.generator``: a hidden Markov process with four user
            types and three spender classes. Candidate B2 has the same shape, so this world is
            the best case for the HMM. It has an exact oracle.

``rhythm``  A process that is deliberately not a hidden Markov model. Every user has their own
            continuous traits: a typical spell length, a typical lapse length, a chance that
            each lapse is permanent, an attraction to season launches, a play intensity and a
            spend propensity with a heavy tail. Lapse lengths are peaked around the personal
            typical lapse (gamma, shape 4) rather than geometric, hours fade over the last
            weeks of a spell, and a user's traits drift from spell to spell. No finite set of
            types and states reproduces it.

Neither world is the real product. Running both shows which conclusions depend on the
process; anything that flips between them cannot be decided on synthetic data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ltv.synth import generator as g

WORLDS = ("markov", "rhythm")


def _arrivals(n_users: int, n_weeks: int, sw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """First week per user: a launch cohort, an early wave, then arrivals that favour season launches."""
    w = np.ones(n_weeks)
    w[sw[:n_weeks] <= 2] = 4.0
    w *= np.linspace(1.3, 0.7, n_weeks)
    w[:9] = 0.0
    w[n_weeks - 4:] = 0.0
    w /= w.sum()
    u = rng.random(n_users)
    return np.where(u < 0.15, 0, np.where(u < 0.25, rng.integers(1, 9, n_users),
                                           rng.choice(n_weeks, size=n_users, p=w)))


def _gamma_weeks(mean: np.ndarray, shape: float, rng: np.random.Generator) -> np.ndarray:
    return np.maximum(np.rint(rng.gamma(shape, mean / shape)), 1).astype(np.int32)


def simulate_rhythm(n_users: int, n_weeks: int, sw: np.ndarray, rng: np.random.Generator) -> dict:
    """The semi-Markov 'personal rhythm' world. Returns the same dense arrays as ``generator.simulate``."""
    z = rng.normal(size=(n_users, 5))
    engage = z[:, 0]
    spell_mean = np.exp(1.5 + 0.5 * engage + 0.5 * z[:, 1])                  # median about 4.5 weeks
    lapse_mean = np.exp(2.6 - 0.25 * engage + 0.8 * z[:, 2])                 # median about 13 weeks, wide
    quit_p = np.clip(rng.beta(1.2, 6.0, n_users) * np.exp(-0.4 * engage), 0.01, 0.9)
    launch_aff = rng.beta(0.6, 1.8, n_users)                                 # pull of a season launch
    p_play = np.clip(0.55 + 0.4 / (1 + np.exp(-(engage + z[:, 3]))), 0.3, 0.98)
    mu = 1.3 + 0.5 * engage + 0.4 * z[:, 3]                                  # log hours level
    payer = rng.random(n_users) < 0.4
    q = np.where(payer, np.minimum(np.exp(-3.4 + 1.2 * z[:, 4] + 0.3 * engage), 0.8), 0.0)
    amt_mu = 2.0 + 0.45 * np.maximum(z[:, 4], 0)

    first = _arrivals(n_users, n_weeks, sw, rng)
    main_platform = rng.choice(3, size=n_users, p=[0.5, 0.3, 0.2])
    zipf = 1.0 / np.arange(1, 26)
    country = rng.choice(25, size=n_users, p=zipf / zipf.sum())

    mode = np.full(n_users, -1, np.int8)            # -1 not started, 1 in a spell, 2 lapsed, 3 gone
    age = np.zeros(n_users, np.int32)               # weeks into the current spell
    length = np.zeros(n_users, np.int32)            # length of the current spell
    rem = np.zeros(n_users, np.int32)               # weeks left in the current lapse
    n_lapses = np.zeros(n_users, np.int32)
    states = np.full((n_users, n_weeks), -1, np.int8)
    hours = np.zeros((n_users, n_weeks), np.float32)
    spend = np.zeros((n_users, n_weeks), np.float32)
    platform = np.zeros((n_users, n_weeks), np.int8)

    def start_spell(idx: np.ndarray) -> None:
        mode[idx], age[idx] = 1, 0
        length[idx] = _gamma_weeks(spell_mean[idx], 1.5, rng)

    for w in range(n_weeks):
        launch = 1.0 if sw[w] == 1 else (0.4 if sw[w] == 2 else 0.0)
        # spells that have run their length end in a lapse or for good
        ended = np.flatnonzero((mode == 1) & (age >= length))
        if len(ended):
            boost = np.where(n_lapses[ended] == 0, 2.0, 1.0)      # the first lapse is the most dangerous
            gone = rng.random(len(ended)) < np.minimum(quit_p[ended] * boost + 0.1 * (boost > 1), 0.95)
            mode[ended] = np.where(gone, 3, 2)
            rem[ended] = _gamma_weeks(lapse_mean[ended], 4.0, rng)
            n_lapses[ended] += 1
            # traits drift from spell to spell: a little less play, slightly longer lapses
            mu[ended] += rng.normal(-0.03, 0.15, len(ended))
            lapse_mean[ended] *= np.exp(rng.normal(0.05, 0.2, len(ended)))
        # lapsed users return on their own clock or at a launch; the gone rarely come back at a launch
        lapsed = np.flatnonzero(mode == 2)
        if len(lapsed):
            rem[lapsed] -= 1
            back = (rem[lapsed] <= 0) | (rng.random(len(lapsed)) < launch * launch_aff[lapsed])
            start_spell(lapsed[back])
        if launch:
            gone_idx = np.flatnonzero(mode == 3)
            back = rng.random(len(gone_idx)) < launch * 0.004 * (1 + 5 * launch_aff[gone_idx])
            start_spell(gone_idx[back])
        start_spell(np.flatnonzero(first == w))

        live = np.flatnonzero(mode >= 1)
        states[live, w] = mode[live]
        on = live[mode[live] == 1]
        a, left = age[on], length[on] - age[on]
        act = (rng.random(len(on)) < p_play[on]) | (a == 0)
        level = mu[on] + 0.3 * (a < 2) + 0.15 * (launch > 0) - 0.45 * np.maximum(0, 3 - left)   # fade before a lapse
        h = np.exp(level + 0.55 * rng.normal(size=len(on)))
        hours[on, w] = np.where(act, np.minimum(h, g.MAX_HOURS), 0.0)
        qs = np.minimum(q[on] * (1 + 1.0 * (launch > 0)) * (1 + 0.8 * (a == 0)), 0.9)
        sp = act & (rng.random(len(on)) < qs)
        spend[on, w] = np.where(sp, np.round(np.exp(amt_mu[on] + 0.9 * rng.normal(size=len(on))), 2), 0.0)
        off = live[mode[live] == 2]                                 # rare spend-only weeks while lapsed
        sp_off = rng.random(len(off)) < q[off] * 0.02
        spend[off, w] = np.where(sp_off, np.round(np.exp(amt_mu[off] + 0.9 * rng.normal(size=len(off))), 2), 0.0)
        switch = rng.random(len(live)) < 0.03
        platform[live, w] = np.where(switch, rng.integers(0, 3, len(live)), main_platform[live])
        age[on] += 1

    return dict(user_type=np.digitize(lapse_mean, [8, 20]).astype(np.int8), spender=payer.astype(np.int8),
                user_effect=mu, first_week=first, states=states, hours=hours, spend=spend,
                platform=platform, country=country)


SIMULATORS = {"markov": g.simulate, "rhythm": simulate_rhythm}


def world_calendar(n_weeks: int, seed: int, future_weeks: int = 20) -> pd.DataFrame:
    return g.make_calendar(n_weeks, future_weeks, np.random.default_rng([seed, 0]))


def store_from_sim(sim: dict, calendar: pd.DataFrame, offset: int = 0):
    """A SequenceStore from dense simulated arrays. Users are split 80/10/10 by position."""
    from ltv.data.sequences import SequenceStore
    n, _ = sim["hours"].shape
    pos = np.arange(offset, offset + n)
    present = (sim["hours"] > 0) | (sim["spend"] > 0)
    return SequenceStore(
        user_ids=np.char.add("u", np.char.zfill(pos.astype(str), 8)),
        split=np.where(pos % 10 < 8, "train", np.where(pos % 10 == 8, "val", "test")),
        first_week=sim["first_week"].astype(np.int32),
        hours=sim["hours"].astype(np.float32), spend=sim["spend"].astype(np.float32),
        platform=np.where(present, sim["platform"].astype(np.int16) + 1, 0).astype(np.int16),
        country=np.where(present, sim["country"].astype(np.int16)[:, None] + 1, 0).astype(np.int16),
        season_week=calendar["season_week_number"].to_numpy(np.int16),
        season_idx=pd.factorize(calendar["season"])[0].astype(np.int16))


def write_world(world: str, n_users: int, out_dir: str | Path, n_weeks: int = 320, seed: int = 7,
                shard_users: int = 100_000) -> list[Path]:
    """Generate ``n_users`` histories as ``shard_XXXXX.npz`` files under ``out_dir/shards``.

    Memory stays at one shard. Each shard also gets ``truth_XXXXX.npz`` with what the
    simulation knows and the models do not (types and weekly hidden states), for the oracle.
    Existing shards are kept, so an interrupted call continues.
    """
    if world not in SIMULATORS:
        raise ValueError(f"unknown world {world!r}; choose from {WORLDS}")
    out = Path(out_dir) / "shards"
    out.mkdir(parents=True, exist_ok=True)
    cal = world_calendar(n_weeks, seed)
    cal.to_csv(Path(out_dir) / "calendar.csv", index=False)
    sw = cal["season_week_number"].to_numpy()
    paths = []
    for i, start in enumerate(range(0, n_users, shard_users)):
        path = out / f"shard_{i:05d}.npz"
        paths.append(path)
        if path.exists():
            continue
        n = min(shard_users, n_users - start)
        sim = SIMULATORS[world](n, n_weeks, sw, np.random.default_rng([seed, 1, i]))
        store = store_from_sim(sim, cal, start)
        np.savez_compressed(out / f"truth_{i:05d}.npz", user_type=sim["user_type"], spender=sim["spender"],
                            user_effect=sim["user_effect"], first_week=sim["first_week"], states=sim["states"])
        tmp = out / f"tmp_{i:05d}.npz"
        store.save(tmp)
        tmp.rename(path)
    return paths


def load_truth(shard_path: str | Path) -> dict:
    """The truth file written next to a shard, as the dict ``generator.oracle_predict`` expects."""
    from ltv.data.sequences import SequenceStore
    p = Path(shard_path)
    z = np.load(p.with_name(p.name.replace("shard_", "truth_")))
    t = {k: z[k] for k in z.files}
    store = SequenceStore.load(p)
    t.update(user_ids=store.user_ids, hours=store.hours, spend=store.spend, n_weeks=store.n_weeks)
    return t
