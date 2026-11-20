"""Example players, ready to plot: weekly history overlaid with each model's weekly outputs.

The evaluation tables say how the models compare on average. This module builds the other
view: individual players, one row per player, week and series, so a person can plot a
history and see what each model said about it.

One long table (``example_weekly``), three views, one ``hours`` column to overlay:

``actual``  the player's real weeks, from their first week to the end of the data.
``path``    the forecast a model made at one origin for each of the next 52 weeks: expected
            hours in that week, the probability of being active in that week, expected spend.
``as_of``   the model's opinion as it stood at the end of each week of a rolling window:
            P(re-engage in the next 52 weeks), the 52-week totals, and the near-term rate
            (expected hours per week over the next 4 weeks). This is the view that shows
            what a model does during a lapse and when the player returns.

Models enter as plain functions ``score(store, rows, origin) -> Preds``, so the same code
serves the Databricks pipeline (``ltv.pipeline.build_examples``), the simulation and tests.
Nothing here reads a week after the origin being scored: each as-of row is a fresh score
of the history up to that week.
"""
from __future__ import annotations

import zlib
from typing import Callable

import numpy as np
import pandas as pd

from ltv.data.sequences import CUM_HORIZONS, SequenceStore
from ltv.models.base import TARGETS, Preds

VIEWS = ("actual", "path", "as_of")
AS_OF_COLS = ("p_reengage_52", "exp_active_weeks_52", "exp_hours_52", "p_spend_52", "exp_spend_52")
WEEKLY_COLUMNS = ["user_id", "example_group", "series", "view", "week_idx", "week_start", "origin_week",
                  "weeks_from_origin", "hours", "active", "spend", "cum_hours", *AS_OF_COLS,
                  "fit", "calibrated", "after_training_period"]
WEEKLY_SCHEMA = ("user_id string, example_group string, series string, view string, week_idx int, week_start date, "
                 "origin_week int, weeks_from_origin int, hours double, active double, spend double, cum_hours double, "
                 "p_reengage_52 double, exp_active_weeks_52 double, exp_hours_52 double, p_spend_52 double, "
                 "exp_spend_52 double, fit string, calibrated boolean, after_training_period boolean")

# Groups a person would want to look at, most specific first: a player lands in the first one that fits.
GROUPS = (
    ("serial returner, long dormant", lambda f: (f["n_returns_26"] >= 2) & (f["wk_since_active"] >= 27)),
    ("returned once, long dormant", lambda f: (f["n_returns_26"] == 1) & (f["wk_since_active"] >= 27)),
    ("first long lapse", lambda f: (f["n_returns_26"] == 0) & (f["wk_since_active"] >= 27)),
    ("serial returner, playing now", lambda f: (f["n_returns_26"] >= 2) & (f["wk_since_active"] == 0)),
    ("lapsed 5 to 26 weeks", lambda f: f["wk_since_active"].between(5, 26)),
    ("new player", lambda f: (f["tenure_weeks"] < 13) & (f["wk_since_active"] <= 4)),
    ("playing, hours falling", lambda f: (f["wk_since_active"] == 0) & (f["aw_52"] >= 13) & (_recent_ratio(f) < 0.5)),
    ("playing, hours rising", lambda f: (f["wk_since_active"] == 0) & (f["aw_52"] >= 13) & (_recent_ratio(f) > 1.5)),
    ("high spender", lambda f: _high_spender(f)),
    ("playing, steady", lambda f: (f["wk_since_active"] == 0) & (f["aw_52"] >= 13)),
)


def _recent_ratio(f: pd.DataFrame) -> pd.Series:
    """Hours per active week in the last 4 weeks against the player's own 52-week norm."""
    recent = f["hrs_4"] / f["aw_4"].where(f["aw_4"] > 0)
    norm = f["hrs_52"] / f["aw_52"].where(f["aw_52"] > 0)
    return (recent / norm).fillna(1.0)


def _high_spender(f: pd.DataFrame) -> pd.Series:
    """In the top tenth of players who spent anything in the last 52 weeks."""
    paid = f["spd_52"][f["spd_52"] > 0]
    return f["spd_52"] >= paid.quantile(0.9) if len(paid) else pd.Series(False, index=f.index)


def _stable_order(user_ids: np.ndarray, seed: int) -> np.ndarray:
    """A fixed pseudo-random order of users that does not depend on the order they arrive in."""
    key = np.array([zlib.crc32(f"{seed}:{u}".encode()) for u in user_ids], dtype=np.int64)
    return np.lexsort((user_ids.astype(str), key))


def pick_examples(frame: pd.DataFrame, per_group: int = 20, seed: int = 0, user_ids=None,
                  disagreement: pd.Series | None = None) -> pd.DataFrame:
    """Choose example players from a features frame at one origin.

    ``frame`` has ``user_id`` and the tabular features. ``per_group`` players are taken from
    each group in ``GROUPS``, in a fixed pseudo-random order, so the same data gives the
    same players. ``user_ids`` adds players by id (group ``requested``). ``disagreement`` is
    an optional score per user_id (for example |A1 - C1| on P(re-engage)): the top
    ``per_group`` become the group ``models disagree most``.

    Returns user_id, example_group; one row per player.
    """
    f = frame.reset_index(drop=True)
    ids = f["user_id"].to_numpy().astype(str)
    taken: dict[str, str] = {}
    known = set(ids)
    for u in (user_ids or []):
        if str(u) in known:
            taken[str(u)] = "requested"
    if disagreement is not None and len(disagreement):
        d = disagreement[disagreement.index.astype(str).isin(known)].sort_values(ascending=False, kind="stable")
        for u in d.index.astype(str)[:per_group]:
            taken.setdefault(u, "models disagree most")
    order = _stable_order(ids, seed)
    free = np.ones(len(f), bool)
    for name, rule in GROUPS:
        m = np.asarray(rule(f), bool) & free
        free &= ~m                                   # a player belongs to the first group that fits
        n = 0
        for i in order[m[order]]:
            if n >= per_group:
                break
            if ids[i] not in taken:
                taken[ids[i]] = name
                n += 1
    return pd.DataFrame({"user_id": list(taken), "example_group": list(taken.values())})


def weekly_path(cum: np.ndarray, weekly_p: np.ndarray | None) -> np.ndarray:
    """Spread cumulative amounts at the horizons (4, 13, 26, 39, 52) over the 52 weeks.

    Each bucket's amount is shared between its weeks in proportion to the model's weekly
    P(active), or evenly where the model gives none. The result sums to the 52-week total
    and never goes negative.
    """
    cum = np.asarray(cum, float)
    n, horizon = len(cum), CUM_HORIZONS[-1]
    inc = np.maximum(np.diff(np.concatenate([np.zeros((n, 1)), cum], axis=1), axis=1), 0.0)
    out = np.zeros((n, horizon))
    lo = 0
    for b, hi in enumerate(CUM_HORIZONS):
        w = np.ones((n, hi - lo)) if weekly_p is None else np.asarray(weekly_p, float)[:, lo:hi]
        tot = w.sum(axis=1, keepdims=True)
        share = np.where(tot > 0, w / np.maximum(tot, 1e-12), 1.0 / (hi - lo))
        out[:, lo:hi] = inc[:, b:b + 1] * share
        lo = hi
    return out


def _calibrate(p: Preds, recal) -> Preds:
    """Apply the shared calibration layer to a ``Preds``; arrays are rescaled with their totals."""
    if recal is None:
        return p
    raw = pd.DataFrame({t: getattr(p, t) for t in TARGETS})
    cal = pd.DataFrame({t: recal.maps[t].predict(raw[t].to_numpy(float)) for t in TARGETS})
    def scaled(arr, total, cap=None):
        """Rescale an array with its total. A model that said exactly zero gets the calibrated
        total spread evenly over time, so paths always add up to the totals shown."""
        if arr is None:
            return None
        arr, new, old = np.asarray(arr, float), cal[total].to_numpy(), raw[total].to_numpy(float)
        even = (np.asarray(CUM_HORIZONS, float) / CUM_HORIZONS[-1] if arr.shape[1] == len(CUM_HORIZONS)
                else np.full(arr.shape[1], 1.0 / arr.shape[1]))
        out = np.where((old > 1e-12)[:, None], arr * (new / np.maximum(old, 1e-12))[:, None], new[:, None] * even)
        return np.clip(out, 0.0, cap)
    return Preds(p.user_ids, p.origin, cal["p_reengage"].to_numpy(), cal["exp_active_weeks"].to_numpy(),
                 cal["exp_hours"].to_numpy(), cal["p_spend"].to_numpy(), cal["exp_spend"].to_numpy(),
                 cum_hours=scaled(p.cum_hours, "exp_hours"), cum_spend=scaled(p.cum_spend, "exp_spend"),
                 weekly_p_active=scaled(p.weekly_p_active, "exp_active_weeks", 1.0))


def actual_rows(store: SequenceStore, rows: np.ndarray, origin: int) -> pd.DataFrame:
    """One row per player and week, from the player's first week to the last week of data."""
    parts = []
    for r in rows:
        first = int(store.first_week[r])
        w = np.arange(first, store.n_weeks)
        h, s = store.hours[r, first:].astype(float), store.spend[r, first:].astype(float)
        after = w > origin
        cum = np.where(after, np.cumsum(np.where(after, h, 0.0)), np.nan)
        parts.append(pd.DataFrame({"user_id": store.user_ids[r], "series": "actual", "view": "actual", "week_idx": w,
                                   "origin_week": origin, "hours": h, "active": (h > 0).astype(float), "spend": s,
                                   "cum_hours": cum}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _even(total: np.ndarray) -> np.ndarray:
    """Cumulative amounts at the horizons for a model that gives only a 52-week total: an even spread."""
    return np.asarray(total, float)[:, None] * (np.asarray(CUM_HORIZONS, float) / CUM_HORIZONS[-1])


def path_rows(p: Preds, model_id: str) -> pd.DataFrame:
    """The forecast made at ``p.origin`` for each of the next 52 weeks.

    A model that predicts only 52-week totals (A1) has no shape over time: its path is the
    total spread evenly, a flat line. A2 and the sequence models give the shape.
    """
    n, horizon = len(p.user_ids), CUM_HORIZONS[-1]
    hours = weekly_path(p.cum_hours if p.cum_hours is not None else _even(p.exp_hours), p.weekly_p_active)
    spend = weekly_path(p.cum_spend if p.cum_spend is not None else _even(p.exp_spend), p.weekly_p_active)
    active = (np.asarray(p.weekly_p_active, float) if p.weekly_p_active is not None
              else np.repeat(np.asarray(p.exp_active_weeks, float)[:, None] / horizon, horizon, axis=1))
    week = p.origin + 1 + np.arange(horizon)
    return pd.DataFrame({"user_id": np.repeat(p.user_ids, horizon), "series": model_id, "view": "path",
                         "week_idx": np.tile(week, n), "origin_week": p.origin, "hours": hours.ravel(),
                         "active": active.ravel(), "spend": spend.ravel(),
                         "cum_hours": np.cumsum(hours, axis=1).ravel()})


def as_of_rows(p: Preds, model_id: str) -> pd.DataFrame:
    """The model's opinion at the end of week ``p.origin``: 52-week outputs and the near-term rate.

    The near-term columns are empty for a model that predicts only 52-week totals (A1)."""
    first = CUM_HORIZONS[0]
    near = lambda cum: np.asarray(cum, float)[:, 0] / first if cum is not None else np.nan
    return pd.DataFrame({
        "user_id": p.user_ids, "series": model_id, "view": "as_of", "week_idx": p.origin, "origin_week": p.origin,
        "hours": near(p.cum_hours),                                       # expected hours per week, next 4 weeks
        "active": (np.asarray(p.weekly_p_active, float)[:, 0] if p.weekly_p_active is not None else np.nan),
        "spend": near(p.cum_spend),
        "p_reengage_52": p.p_reengage, "exp_active_weeks_52": p.exp_active_weeks, "exp_hours_52": p.exp_hours,
        "p_spend_52": p.p_spend, "exp_spend_52": p.exp_spend})


def example_frames(store: SequenceStore, rows: np.ndarray, scorers: dict[str, Callable], origin: int,
                   as_of_origins, recal: dict | None = None, fit: str = "final_1",
                   last_train_origin: int | None = None) -> pd.DataFrame:
    """All three views for the players in ``rows`` of one store.

    ``scorers`` maps a model id to ``score(store, rows, origin) -> Preds``. ``origin`` is
    where the forecast path starts; ``as_of_origins`` are the weeks of the rolling view.
    ``recal`` optionally maps a model id to a fitted ``Recalibrator``. ``last_train_origin``
    marks which as-of weeks lie after everything the fit was trained on.
    """
    if not len(rows):
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    recal = recal or {}
    parts = [actual_rows(store, rows, origin).assign(calibrated=False, after_training_period=None)]
    for model_id, score in scorers.items():
        rc = recal.get(model_id)
        parts.append(path_rows(_calibrate(score(store, rows, origin), rc), model_id)
                     .assign(calibrated=rc is not None, after_training_period=True))
        for t in as_of_origins:
            live = rows[store.first_week[rows] <= t]            # a player has no opinion before they exist
            if not len(live):
                continue
            parts.append(as_of_rows(_calibrate(score(store, live, int(t)), rc), model_id).assign(
                calibrated=rc is not None,
                after_training_period=(None if last_train_origin is None else bool(t > last_train_origin))))
    out = pd.concat(parts, ignore_index=True)
    out["fit"] = fit
    out["weeks_from_origin"] = out["week_idx"] - origin
    for c in WEEKLY_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out


def finish(weekly: pd.DataFrame, groups: pd.DataFrame, week_start: pd.Series | None = None) -> pd.DataFrame:
    """Attach the example group and the calendar date of each week; order the columns."""
    out = weekly.drop(columns=["example_group", "week_start"], errors="ignore").merge(groups, on="user_id", how="left")
    out["week_start"] = out["week_idx"].map(week_start) if week_start is not None else pd.NaT
    out = out[WEEKLY_COLUMNS].sort_values(["example_group", "user_id", "series", "view", "week_idx"])
    for c in ("week_idx", "origin_week", "weeks_from_origin"):
        out[c] = out[c].astype(np.int32)
    flag = out["after_training_period"].astype(object)
    out["after_training_period"] = flag.where(flag.notna(), None)
    out["calibrated"] = out["calibrated"].astype(bool)
    return out.reset_index(drop=True)


def user_summary(frame: pd.DataFrame, groups: pd.DataFrame, origin: int) -> pd.DataFrame:
    """One row per example player: who they are at the origin and what happened next."""
    keep = ["user_id", "tenure_weeks", "wk_since_active", "n_returns_26", "aw_52", "hrs_52", "spd_52"]
    labels = [c for c in ("any_active_52", "active_weeks_52", "hours_52", "any_spend_52", "spend_52") if c in frame]
    out = groups.merge(frame[keep + labels], on="user_id", how="left")
    out.insert(2, "origin_week", np.int32(origin))
    return out.rename(columns={c: f"actual_{c}" for c in labels})


# ---------------------------------------------------------------------------
# A quick look inside the notebook (the table itself is what goes to a plotting tool)
# ---------------------------------------------------------------------------

# One fixed color per model, so a model keeps its color whichever others are shown.
SERIES_COLORS = {"A1": "#2a78d6", "A2": "#eb6834", "C1": "#1baf7a", "C1e": "#eda100",
                 "A1tw": "#e87ba4", "C2": "#008300", "R1": "#6250d6", "R2": "#e34948"}
_INK, _MUTED, _GRID, _SURFACE = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def plot_player(weekly: pd.DataFrame, user_id: str, models=None, weeks_before: int = 104):
    """Two stacked panels for one player. Returns a matplotlib Figure.

    Top: actual weekly hours, and each model's forecast path from the origin.
    Bottom: each model's P(re-engage in the next 52 weeks), as it stood at the end of each week.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    d = weekly[weekly["user_id"] == user_id]
    if d.empty:
        raise ValueError(f"no example rows for user {user_id!r}")
    origin = int(d["origin_week"][d["view"] == "path"].iloc[0]) if (d["view"] == "path").any() else int(d["origin_week"].iloc[0])
    lo = origin - weeks_before
    d = d[d["week_idx"] >= lo]
    series = [m for m in (models or sorted(set(d["series"]) - {"actual"}, key=lambda m: list(SERIES_COLORS).index(m)
                                           if m in SERIES_COLORS else 99))]
    x = "week_start" if d["week_start"].notna().all() else "week_idx"
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, height_ratios=[3, 2], facecolor=_SURFACE)
    act = d[d["view"] == "actual"].sort_values("week_idx")
    top.fill_between(act[x], act["hours"], step="mid", color=_MUTED, alpha=0.35, linewidth=0, label="actual")
    ends = {top: [], bottom: []}
    for m in series:
        color = SERIES_COLORS.get(m, _INK)
        path = d[(d["series"] == m) & (d["view"] == "path")].sort_values("week_idx")
        if len(path):
            top.plot(path[x], path["hours"], color=color, linewidth=2, label=m)
            ends[top].append((m, path[x].iloc[-1], float(path["hours"].iloc[-1])))
        roll = d[(d["series"] == m) & (d["view"] == "as_of")].sort_values("week_idx")
        if len(roll):
            bottom.plot(roll[x], roll["p_reengage_52"], color=color, linewidth=2, label=m)
            ends[bottom].append((m, roll[x].iloc[-1], float(roll["p_reengage_52"].iloc[-1])))
    bottom.set_ylim(0, 1)
    top.set_ylim(bottom=0)
    at = act.loc[act["week_idx"] == origin, x]
    for ax in (top, bottom):
        ax.set_facecolor(_SURFACE)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=_MUTED, length=0, labelsize=9)
        if len(at):
            ax.axvline(at.iloc[0], color=_INK, linewidth=1, linestyle=(0, (3, 3)))
        # Name each line at its right-hand end; lines that finish together share one label.
        gap, placed = 0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0]), []
        for m, xe, ye in sorted(ends[ax], key=lambda e: e[2]):
            if placed and ye - placed[-1][2] < gap:
                placed[-1][0] += f", {m}"
            else:
                placed.append([m, xe, ye])
        for label, xe, ye in placed:
            ax.annotate(label, (xe, ye), xytext=(6, 0), textcoords="offset points", color=_INK, fontsize=9, va="center")
    if len(at):
        bottom.annotate("forecast made here", (at.iloc[0], 0.04), xytext=(5, 0), textcoords="offset points",
                        color=_INK, fontsize=9, va="bottom")
    group = d["example_group"].iloc[0]
    fig.suptitle(f"Player {user_id}  ·  {group}", x=0.01, ha="left", color="#0b0b0b", fontsize=12)
    top.set_title("Hours per week: actual, and each model's forecast made at the origin", loc="left", color=_INK, fontsize=9)
    bottom.set_title("P(re-engage within 52 weeks), as each model saw it at the end of each week", loc="left",
                     color=_INK, fontsize=9)
    top.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=_INK, ncols=1 + len(series))
    fig.tight_layout()
    return fig
