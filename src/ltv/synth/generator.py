"""Synthetic weekly activity data with known ground truth.

THIS IS A TEST FIXTURE. It exists so the pipeline can be executed and checked before it
meets real data. Nothing computed from it may be used to compare, tune or choose models:
the generator is itself a hidden Markov process, so it favours the HMM by construction.
See docs/SYNTHETIC_DATA.md.

What is planted
---------------
* Four user types that never change: steady players, serial returners (lapse for months,
  then come back, repeatedly), launch returners (come back mostly in the first two weeks of
  a season) and one-time players.
* Five weekly states per user: two active levels (hi, lo) and a chain of three dormancy
  states (d1 -> d2 -> d3), so the chance of returning depends on how long the lapse has run.
* Seasons of about two months with varying length; launches raise return probabilities.
* Heavy-tailed spend with three spender classes (none, normal, whale) and rare spend-only
  weeks during dormancy.
* Optionally, one data defect per Phase 0 audit check (``plant_defects``).

Because the process is known, ``oracle_predict`` returns the exact 52-week expectations
given each user's true state. Tests use it to check metrics and model recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TYPES = ("steady", "serial_returner", "launch_returner", "one_time")
STATES = ("hi", "lo", "d1", "d2", "d3")
SPENDER = ("none", "normal", "whale")
N_T, N_S = len(TYPES), len(STATES)
BASE_DATE = np.datetime64("2020-09-07")          # a Monday; week_idx 0

# --- process parameters ----------------------------------------------------------
TYPE_MIX = np.array([0.25, 0.20, 0.20, 0.35])
SPENDER_MIX = np.array([0.65, 0.30, 0.05])
P_ACTIVE = np.array([0.93, 0.55, 0.0, 0.0, 0.0])
HOURS_MU = np.array([2.3, 0.9, 0.0, 0.0, 0.0])
HOURS_SIGMA = 0.6
USER_SIGMA = 0.35
Q_SPEND = np.array([0.10, 0.03, 0.004, 0.001, 0.0003])
SPENDER_Q_MULT = np.array([0.01, 1.0, 2.5])
SPEND_M = 2.3
SPEND_S = 1.0
WHALE_BONUS = 1.5
MAX_HOURS = 120.0

# per type: (hi_stay, hi_lo, lo_hi, lo_stay, [return d1,d2,d3], [deepen d1,d2], launch mult wk1, wk2)
_TYPE_SPEC = {
    "steady":          (0.95, 0.04, 0.08, 0.84, (0.12, 0.05, 0.015), (0.10, 0.06), 1.5, 1.2),
    "serial_returner": (0.88, 0.07, 0.06, 0.78, (0.05, 0.06, 0.04), (0.12, 0.03), 1.5, 1.2),
    "launch_returner": (0.80, 0.10, 0.04, 0.66, (0.010, 0.008, 0.006), (0.15, 0.05), 25.0, 12.0),
    "one_time":        (0.80, 0.12, 0.03, 0.67, (0.02, 0.004, 0.0008), (0.20, 0.10), 2.0, 1.5),
}
RETURN_TO_LO = 0.75


def transition_tensor() -> np.ndarray:
    """A[type, bucket, from_state, to_state]; bucket 0 = launch week, 1 = season week 2, 2 = other."""
    A = np.zeros((N_T, 3, N_S, N_S))
    for ti, name in enumerate(TYPES):
        hi_stay, hi_lo, lo_hi, lo_stay, ret, deep, m1, m2 = _TYPE_SPEC[name]
        for b, mult in enumerate((m1, m2, 1.0)):
            A[ti, b, 0] = [hi_stay, hi_lo, 1 - hi_stay - hi_lo, 0, 0]
            A[ti, b, 1] = [lo_hi, lo_stay, 1 - lo_hi - lo_stay, 0, 0]
            for di, s in enumerate((2, 3, 4)):
                r = min(ret[di] * mult, 0.9)
                d = deep[di] if di < 2 else 0.0
                d = min(d, 1 - r)
                row = np.zeros(N_S)
                row[1] = r * RETURN_TO_LO
                row[0] = r * (1 - RETURN_TO_LO)
                if di < 2:
                    row[s + 1] = d
                row[s] = 1 - r - d
                A[ti, b, s] = row
    assert np.allclose(A.sum(-1), 1.0)
    return A


@dataclass
class SynthData:
    activity: pd.DataFrame            # sparse source format: user_id, week_start, hours, spend, platform, country
    calendar: pd.DataFrame            # week_start, season, season_week_number
    truth: dict = field(default_factory=dict)
    defects: dict = field(default_factory=dict)


def make_calendar(n_weeks: int, future_weeks: int, rng: np.random.Generator) -> pd.DataFrame:
    """Seasons of 7 to 11 weeks. The table runs ``future_weeks`` past the data."""
    total = n_weeks + future_weeks
    season, season_week = [], []
    s = 0
    while len(season) < total:
        length = int(rng.choice([7, 8, 8, 9, 9, 10, 11]))
        s += 1
        for w in range(1, length + 1):
            season.append(f"S{s:02d}")
            season_week.append(w)
    weeks = BASE_DATE + np.arange(total) * np.timedelta64(7, "D")
    return pd.DataFrame({"week_start": pd.to_datetime(weeks).date,
                         "season": season[:total],
                         "season_week_number": np.array(season_week[:total], dtype=np.int32)})


def _bucket(season_week: np.ndarray) -> np.ndarray:
    return np.where(season_week == 1, 0, np.where(season_week == 2, 1, 2))


def generate(n_users: int = 50_000, n_weeks: int = 320, seed: int = 7, future_weeks: int = 20,
             plant: bool = False) -> SynthData:
    """Simulate the process and return it in the sparse source format."""
    rng = np.random.default_rng(seed)
    cal = make_calendar(n_weeks, future_weeks, rng)
    sw = cal["season_week_number"].to_numpy()
    bucket = _bucket(sw)
    A = transition_tensor()

    utype = rng.choice(N_T, size=n_users, p=TYPE_MIX)
    spender = rng.choice(len(SPENDER), size=n_users, p=SPENDER_MIX)
    ueff = rng.normal(0.0, USER_SIGMA, size=n_users)

    # Arrival weeks: a launch cohort, an early wave, then arrivals that favour season launches.
    w_arr = np.ones(n_weeks)
    w_arr[sw[:n_weeks] <= 2] = 4.0
    w_arr *= np.linspace(1.3, 0.7, n_weeks)
    w_arr[: 9] = 0.0
    # Leave room after the last arrival so every user has some history.
    w_arr[n_weeks - 4:] = 0.0
    w_arr /= w_arr.sum()
    u = rng.random(n_users)
    first = np.where(u < 0.15, 0,
                     np.where(u < 0.25, rng.integers(1, 9, n_users),
                              rng.choice(n_weeks, size=n_users, p=w_arr)))

    main_platform = rng.choice(3, size=n_users, p=[0.5, 0.3, 0.2])
    zipf = 1.0 / np.arange(1, 26)
    country = rng.choice(25, size=n_users, p=zipf / zipf.sum())

    state = np.full(n_users, -1, dtype=np.int8)
    states = np.full((n_users, n_weeks), -1, dtype=np.int8)
    hours = np.zeros((n_users, n_weeks), dtype=np.float32)
    spend = np.zeros((n_users, n_weeks), dtype=np.float32)
    platform = np.zeros((n_users, n_weeks), dtype=np.int8)

    for w in range(n_weeks):
        started = state >= 0
        if started.any():
            idx = np.flatnonzero(started)
            P = A[utype[idx], bucket[w], state[idx]]                    # [n, S]
            cdf = np.cumsum(P, axis=1)
            draw = rng.random(len(idx))[:, None]
            state[idx] = (draw > cdf).sum(axis=1).clip(0, N_S - 1).astype(np.int8)
        new = first == w
        if new.any():
            state[new] = rng.integers(0, 2, new.sum()).astype(np.int8)
        live = np.flatnonzero(state >= 0)
        st = state[live]
        states[live, w] = st
        act = rng.random(len(live)) < P_ACTIVE[st]
        act |= first[live] == w                                         # the first week is always active
        h = np.exp(HOURS_MU[st] + ueff[live] + HOURS_SIGMA * rng.normal(size=len(live)))
        hours[live, w] = np.where(act, np.minimum(h, MAX_HOURS), 0.0)
        q = np.minimum(Q_SPEND[st] * SPENDER_Q_MULT[spender[live]], 1.0)
        sp = rng.random(len(live)) < q
        amt = np.exp(SPEND_M + WHALE_BONUS * (spender[live] == 2) + SPEND_S * rng.normal(size=len(live)))
        spend[live, w] = np.where(sp, np.round(amt, 2), 0.0)
        switch = rng.random(len(live)) < 0.03
        platform[live, w] = np.where(switch, rng.integers(0, 3, len(live)), main_platform[live])

    ui, wi = np.nonzero((hours > 0) | (spend > 0))
    weeks = BASE_DATE + np.arange(n_weeks) * np.timedelta64(7, "D")
    plat_names = np.array(["pc", "console", "mobile"])
    country_names = np.array([f"C{c:02d}" for c in range(25)])
    activity = pd.DataFrame({
        "user_id": np.char.add("u", np.char.zfill(ui.astype(str), 7)),
        "week_start": pd.to_datetime(weeks[wi]).date,
        "hours": hours[ui, wi].astype(np.float64).round(3),
        "spend": spend[ui, wi].astype(np.float64).round(2),
        "platform": plat_names[platform[ui, wi]],
        "country": country_names[country[ui]],
    })
    truth = dict(user_type=utype, spender=spender, user_effect=ueff, first_week=first,
                 states=states, hours=hours, spend=spend, n_weeks=n_weeks,
                 season_week=sw, user_ids=np.char.add("u", np.char.zfill(np.arange(n_users).astype(str), 7)))
    data = SynthData(activity=activity, calendar=cal, truth=truth)
    if plant:
        plant_defects(data, rng)
    return data


# ---------------------------------------------------------------------------
# Planted defects: one per Phase 0 audit check
# ---------------------------------------------------------------------------

def plant_defects(data: SynthData, rng: np.random.Generator) -> None:
    """Corrupt ``data.activity`` in known ways and record what was done in ``data.defects``.

    The clean pipeline must remove or neutralise every one of these, and each audit query
    must report its defect. tests/test_audit.py checks both.
    """
    act = data.activity
    n_weeks = data.truth["n_weeks"]
    weeks = np.sort(act["week_start"].unique())
    defects: dict = {}

    # 1. Pipeline outage: one week loses most of its rows.
    outage_week = weeks[n_weeks // 2 + 3]
    in_week = np.flatnonzero((act["week_start"] == outage_week).to_numpy())
    drop = rng.choice(in_week, size=int(len(in_week) * 0.85), replace=False)
    act = act.drop(index=act.index[drop]).reset_index(drop=True)
    defects["outage_week"] = str(outage_week)

    extra = []
    # 2. Rows with zero hours and zero spend.
    z = act.sample(300, random_state=1).copy()
    z["hours"], z["spend"] = 0.0, 0.0
    z["week_start"] = weeks[5]
    z["user_id"] = [f"zero{i:05d}" for i in range(len(z))]
    extra.append(z)
    defects["zero_rows"] = len(z)

    # 3. Negative spend (refunds booked as negative rows).
    neg = act.sample(150, random_state=2).copy()
    neg["hours"], neg["spend"] = 0.0, -9.99
    extra.append(neg)
    defects["negative_spend_rows"] = len(neg)

    # 4. Duplicate user-weeks on a second platform.
    dup = act[act["hours"] > 1].sample(200, random_state=3).copy()
    dup["hours"] = dup["hours"] * 0.25
    dup["spend"] = 0.0
    dup["platform"] = "handheld"
    extra.append(dup)
    defects["duplicate_user_weeks"] = len(dup)

    # 5. Test accounts with impossible hours.
    t_users = [f"test{i:03d}" for i in range(20)]
    t_weeks = weeks[40:80]
    tacc = pd.DataFrame({"user_id": np.repeat(t_users, len(t_weeks)),
                         "week_start": np.tile(t_weeks, len(t_users)),
                         "hours": 400.0, "spend": 0.0, "platform": "pc", "country": "C00"})
    extra.append(tacc)
    defects["suspect_users"] = len(t_users)

    # 6. Rows whose week_start is not a calendar week (shifted two days).
    mis = act.sample(100, random_state=4).copy()
    mis["week_start"] = (pd.to_datetime(mis["week_start"]) + pd.Timedelta(days=2)).dt.date
    extra.append(mis)
    defects["misaligned_rows"] = len(mis)

    # 7. Missing platform and country.
    nul = act.sample(100, random_state=5).copy()
    nul["user_id"] = [f"null{i:05d}" for i in range(len(nul))]
    nul["platform"], nul["country"] = None, None
    extra.append(nul)
    defects["null_dimension_rows"] = len(nul)

    data.activity = pd.concat([act] + extra, ignore_index=True)
    data.defects = defects


# ---------------------------------------------------------------------------
# Oracle: exact expectations under the true process
# ---------------------------------------------------------------------------

def oracle_predict(truth: dict, season_week: np.ndarray, origin: int, horizon: int = 52,
                   cum_horizons=(4, 13, 26, 39, 52)) -> pd.DataFrame:
    """Exact 52-week expectations for every user alive at ``origin``, given their true state.

    ``season_week`` is the calendar the oracle is allowed to see for weeks origin+1 ..
    origin+horizon. Pass the real calendar for the true oracle, or a projected one to see
    what a perfect model limited to the projected calendar would say.
    """
    A = transition_tensor()
    bucket = _bucket(np.asarray(season_week))
    st0 = truth["states"][:, origin]
    alive = np.flatnonzero(st0 >= 0)
    n = len(alive)
    out = {k: np.zeros(n) for k in ("p_reengage", "exp_active_weeks", "exp_hours_base", "p_spend",
                                    "exp_spend")}
    weekly = np.zeros((n, horizon), dtype=np.float32)
    cum_h = np.zeros((n, len(cum_horizons)))
    cum_s = np.zeros((n, len(cum_horizons)))
    utype, spender = truth["user_type"][alive], truth["spender"][alive]
    hours_mean = np.exp(HOURS_MU + 0.5 * HOURS_SIGMA ** 2) * P_ACTIVE          # per state, user effect 0
    for t in range(N_T):
        for c in range(len(SPENDER)):
            q = np.minimum(Q_SPEND * SPENDER_Q_MULT[c], 1.0)
            amt = np.exp(SPEND_M + WHALE_BONUS * (c == 2) + 0.5 * SPEND_S ** 2)
            for s in range(N_S):
                m = np.flatnonzero((utype == t) & (spender == c) & (st0[alive] == s))
                if len(m) == 0:
                    continue
                pi = np.zeros(N_S); pi[s] = 1.0
                no_act = pi.copy(); no_sp = pi.copy()
                e_aw = e_h = e_s = 0.0
                wk = np.zeros(horizon)
                ch, cs = [], []
                for h in range(1, horizon + 1):
                    Ah = A[t, bucket[h - 1]]
                    pi = pi @ Ah
                    no_act = (no_act @ Ah) * (1 - P_ACTIVE)
                    no_sp = (no_sp @ Ah) * (1 - q)
                    wk[h - 1] = pi @ P_ACTIVE
                    e_aw += wk[h - 1]
                    e_h += pi @ hours_mean
                    e_s += (pi @ q) * amt
                    if h in cum_horizons:
                        ch.append(e_h); cs.append(e_s)
                out["p_reengage"][m] = 1 - no_act.sum()
                out["exp_active_weeks"][m] = e_aw
                out["exp_hours_base"][m] = e_h
                out["p_spend"][m] = 1 - no_sp.sum()
                out["exp_spend"][m] = e_s
                weekly[m] = wk
                cum_h[m] = ch
                cum_s[m] = cs
    scale = np.exp(truth["user_effect"][alive])
    df = pd.DataFrame({"user_id": truth["user_ids"][alive], "origin_week": origin,
                       "p_reengage": out["p_reengage"], "exp_active_weeks": out["exp_active_weeks"],
                       "exp_hours": out["exp_hours_base"] * scale, "p_spend": out["p_spend"],
                       "exp_spend": out["exp_spend"]})
    df["cum_hours"] = list(cum_h * scale[:, None])
    df["cum_spend"] = list(cum_s)
    df["weekly_p_active"] = list(weekly)
    return df
