"""The one evaluation every candidate goes through.

Input: a pandas frame of predictions joined to labels and features (one row per model x
origin x user). Output: long tables of metrics, coherence violations, interval coverage and
paired model comparisons. Nothing here knows which model produced a row.

How a "win" is decided (docs/DESIGN.md, "Model comparison"). For each target, models are
ranked on the primary metric among 'test'-split users at each test origin. The leader wins
the target only if the 95% bootstrap interval of its paired difference from the runner-up
excludes zero at BOTH test origins. The bootstrap resamples users, and uses the same
resampled users for both models, which is what makes it paired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ltv.eval import metrics as M
from ltv.eval.calibration import Recalibrator, SegmentConformal
from ltv.eval.segments import segment_frame, segment_label
from ltv.models.base import LABEL_OF, TARGETS

SEGMENT_DIMS = ("dormancy", "tenure", "prior_spender", "returner", "returner_x_dormancy", "platform", "country")
MIN_SEGMENT = 30


def _rows(target: str, y, pred, base: dict, dim: str, seg: str) -> list[dict]:
    return [{**base, "target": target, "segment_dim": dim, "segment": seg, "n": int(len(y)), "metric": k,
             "value": float(v)} for k, v in M.target_metrics(target, y, pred).items()]


def metric_table(df: pd.DataFrame, base: dict, top_countries: int = 10) -> pd.DataFrame:
    """Every metric for one model on one group of users, overall and by segment."""
    seg = segment_frame(df, top_countries)
    out: list[dict] = []
    for t in TARGETS:
        y, p = df[LABEL_OF[t]].to_numpy(float), df[t].to_numpy(float)
        out += _rows(t, y, p, base, "all", "all")
        for dim in SEGMENT_DIMS:
            codes = seg[dim].to_numpy()
            for c in np.unique(codes):
                m = codes == c
                if m.sum() >= MIN_SEGMENT:
                    out += _rows(t, y[m], p[m], base, dim, segment_label(dim, int(c)))
    return pd.DataFrame(out)


def _stack(col: pd.Series) -> np.ndarray | None:
    v = col.to_numpy()
    if len(v) == 0 or v[0] is None or (isinstance(v[0], float) and np.isnan(v[0])):
        return None
    return np.stack([np.asarray(a, float) for a in v])


def coherence_table(df: pd.DataFrame, base: dict, tol: float = 1e-6) -> pd.DataFrame:
    """Share of users violating each coherence rule. Rules that need arrays are skipped when absent."""
    p_re, aw = df["p_reengage"].to_numpy(float), df["exp_active_weeks"].to_numpy(float)
    eh = df["exp_hours"].to_numpy(float)
    checks = {
        "active_weeks_outside_0_52": (aw < -tol) | (aw > 52 + tol),
        "p_reengage_above_active_weeks": p_re > aw + 1e-3,                 # P(any) <= E[count]
        "active_weeks_above_52_p_reengage": aw > 52 * p_re + 1e-3,         # E[count] <= 52 P(any)
        "hours_without_reengage": (p_re < 0.01) & (eh > 0.05 * max(float(np.mean(eh)), 1e-9)),
    }
    wk = _stack(df["weekly_p_active"])
    if wk is not None:
        checks["p_reengage_below_a_weekly_p"] = p_re < wk.max(axis=1) - 1e-3
        checks["p_reengage_above_sum_weekly_p"] = p_re > wk.sum(axis=1) + 1e-3
        checks["weekly_sum_not_active_weeks"] = np.abs(wk.sum(axis=1) - aw) > 1e-2 * np.maximum(aw, 1.0)
    for name, total in (("cum_hours", "exp_hours"), ("cum_spend", "exp_spend")):
        c = _stack(df[name])
        if c is not None:
            checks[f"{name}_decreases"] = (np.diff(c, axis=1) < -1e-6).any(axis=1)
            tot = df[total].to_numpy(float)
            checks[f"{name}_last_not_total"] = np.abs(c[:, -1] - tot) > 1e-2 * np.maximum(tot, 1e-6)
    return pd.DataFrame([{**base, "check": k, "n": int(len(df)), "violations": int(v.sum()),
                          "share": float(v.mean()) if len(df) else 0.0} for k, v in checks.items()])


def interval_table(val: pd.DataFrame, test: pd.DataFrame, base: dict, levels=(0.8, 0.95)) -> pd.DataFrame:
    """Coverage of the shared conformal intervals, overall and by dormancy segment, plus the
    coverage of a model's own 10%-90% quantiles where it has them."""
    out = []
    seg_v, seg_t = segment_frame(val)["dormancy"].to_numpy(), segment_frame(test)["dormancy"].to_numpy()
    for t in ("exp_hours", "exp_spend"):
        y_t = test[LABEL_OF[t]].to_numpy(float)
        conf = SegmentConformal(levels).fit(val[t].to_numpy(float), val[LABEL_OF[t]].to_numpy(float), seg_v)
        for lvl in levels:
            lo, hi = conf.interval(test[t].to_numpy(float), seg_t, lvl)
            out.append({**base, "target": t, "kind": "conformal", "level": lvl, "segment": "all",
                        "n": int(len(y_t)), "coverage": M.coverage(y_t, lo, hi), "mean_width": float(np.mean(hi - lo))})
            for c in np.unique(seg_t):
                m = seg_t == c
                if m.sum() >= MIN_SEGMENT:
                    out.append({**base, "target": t, "kind": "conformal", "level": lvl,
                                "segment": segment_label("dormancy", int(c)), "n": int(m.sum()),
                                "coverage": M.coverage(y_t[m], lo[m], hi[m]), "mean_width": float(np.mean(hi[m] - lo[m]))})
        q = _stack(test["q_hours" if t == "exp_hours" else "q_spend"])
        if q is not None:
            out.append({**base, "target": t, "kind": "model_quantiles", "level": 0.8, "segment": "all",
                        "n": int(len(y_t)), "coverage": M.coverage(y_t, q[:, 0], q[:, 2]),
                        "mean_width": float(np.mean(q[:, 2] - q[:, 0]))})
    return pd.DataFrame(out)


def paired_bootstrap(target: str, y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, draws: int = 200,
                     seed: int = 0) -> dict:
    """Bootstrap the difference in the primary metric (a minus b), resampling users.

    ``better`` is +1 if a is significantly better, -1 if b is, 0 if the interval contains zero.
    """
    name, direction = M.PRIMARY[target]
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        diffs[i] = M.primary_value(target, y[idx], pred_a[idx]) - M.primary_value(target, y[idx], pred_b[idx])
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = (np.quantile(diffs, [0.025, 0.975]) if len(diffs) else (np.nan, np.nan))
    point = M.primary_value(target, y, pred_a) - M.primary_value(target, y, pred_b)
    sign = -1.0 if direction == "lower" else 1.0            # make "positive = a better"
    better = 1 if sign * lo > 0 and sign * hi > 0 else (-1 if sign * lo < 0 and sign * hi < 0 else 0)
    return {"metric": name, "diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "a_better": int(better)}


def family(model_id: str) -> str:
    """Candidates A, B and C come in variants ('A1', 'A1tw', 'A2'); any other model stands alone."""
    return model_id[0] if model_id[:1] in ("A", "B", "C") and model_id[1:2].isdigit() else model_id


def pairwise_table(test: pd.DataFrame, models: list[str], draws: int = 200, seed: int = 0) -> pd.DataFrame:
    """For each test origin and target: the leader against the best model of another family,
    and every other model against the leader, on the primary metric.

    The leader is not compared with its own variants. They can share a head outright (A1,
    A1tw and A2 use one P(re-engage) model), and a tie with a sibling says nothing about
    which family wins.
    """
    out = []
    for origin, g in test.groupby("origin_week"):
        wide = {m: d.set_index("user_id") for m, d in g.groupby("model_id") if m in models}
        common = None
        for d in wide.values():
            common = d.index if common is None else common.intersection(d.index)
        if common is None or len(common) < 50:
            continue
        for t in TARGETS:
            name, direction = M.PRIMARY[t]
            y = next(iter(wide.values())).loc[common, LABEL_OF[t]].to_numpy(float)
            score = {m: M.primary_value(t, y, d.loc[common, t].to_numpy(float)) for m, d in wide.items()}
            order = sorted(score, key=lambda m: score[m], reverse=(direction == "higher"))
            for rank, a in enumerate(order):
                if rank == 0:                                # leader vs the best model of another family
                    b = next((m for m in order[1:] if family(m) != family(a)), None)
                else:                                        # the others vs the leader
                    b = order[0]
                if b is None:
                    continue
                r = paired_bootstrap(t, y, wide[a].loc[common, t].to_numpy(float),
                                     wide[b].loc[common, t].to_numpy(float), draws, seed)
                out.append({"origin_week": int(origin), "target": t, "rank": rank + 1, "model_a": a,
                            "model_b": b, "value_a": float(score[a]), "value_b": float(score[b]),
                            "n": int(len(common)), **r})
    return pd.DataFrame(out)


def winners(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Per target: the leader at each test origin, and whether it wins: the same model leads at
    both origins and beats the best model of another family significantly at both."""
    rows = []
    for t, g in pairwise[pairwise["rank"] == 1].groupby("target"):
        leaders = g.set_index("origin_week")
        same = leaders["model_a"].nunique() == 1
        sig = bool((leaders["a_better"] == 1).all())
        rows.append({"target": t, "metric": leaders["metric"].iloc[0],
                     "leaders": ", ".join(f"{o}: {m}" for o, m in leaders["model_a"].items()),
                     "runners_up": ", ".join(f"{o}: {m}" for o, m in leaders["model_b"].items()),
                     "winner": leaders["model_a"].iloc[0] if (same and sig and len(leaders) >= 2) else "no clear winner",
                     "origins": int(len(leaders))})
    return pd.DataFrame(rows)


CALIBRATIONS = ("tune_fit", "same_fit")


def evaluate(frame: pd.DataFrame, schedule, draws: int = 200, seed: int = 0, top_countries: int = 10,
             calibration: str = "tune_fit") -> dict:
    """Run the whole evaluation. ``frame`` holds predictions joined to labels and features, for
    all models, at the validation and test origins.

    ``calibration`` says which predictions fit the shared layer for a final fit:

    - ``"tune_fit"`` (plan of record): the candidate's ``tune`` fit, 'val' users, validation origin.
    - ``"same_fit"``: the final fit's own predictions for 'val' users at its last training
      origin, whose labels end before the test origin. The frame must hold those rows.
      Simulation only so far (``ltv.sim``); the Databricks pipeline does not score them. See
      the open question on calibration in ``docs/DECISIONS.md``.

    Returns a dict of DataFrames: metrics, coherence, intervals, pairwise, winners.
    """
    if calibration not in CALIBRATIONS:
        raise ValueError(f"calibration must be one of {CALIBRATIONS}")
    tests = {schedule.test_1: "final_1", schedule.test_2: "final_2"}
    metrics, coher, inter, cal_frames = [], [], [], []
    for model, g in frame.groupby("model_id"):
        val = g[(g["origin_week"] == schedule.validation) & (g["fit"] == "tune") & (g["split"] == "val")]
        if len(val):
            metrics.append(metric_table(val, {"model_id": model, "origin_week": schedule.validation,
                                              "split": "val", "calibrated": False}, top_countries))
        for origin, fit in tests.items():
            at = g[(g["origin_week"] == origin) & (g["fit"] == fit)]
            cal = val
            if calibration == "same_fit":
                cal = g[(g["origin_week"] == schedule.train_origins(fit)[-1]) & (g["fit"] == fit)
                        & (g["split"] == "val")]
                if not len(cal):
                    raise ValueError(f"{model}/{fit}: no 'val' predictions at its last training origin")
            recal = Recalibrator().fit(cal) if len(cal) >= 100 else None
            for split in ("test", "train"):
                d = at[at["split"] == split]
                if len(d) < MIN_SEGMENT:
                    continue
                base = {"model_id": model, "origin_week": int(origin), "split": split}
                metrics.append(metric_table(d, {**base, "calibrated": False}, top_countries))
                if split == "test":
                    coher.append(coherence_table(d, base))
                if recal is not None:
                    dc = recal.apply(d)
                    metrics.append(metric_table(dc, {**base, "calibrated": True}, top_countries))
                    if split == "test":
                        cal_frames.append(dc)
                        inter.append(interval_table(recal.apply(cal), dc, base))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    test_cal = cat(cal_frames)
    models = sorted(frame["model_id"].unique())
    pw = pairwise_table(test_cal, models, draws, seed) if len(test_cal) else pd.DataFrame()
    return {"metrics": cat(metrics), "coherence": cat(coher), "intervals": cat(inter), "pairwise": pw,
            "winners": winners(pw) if len(pw) else pd.DataFrame()}


def scorecard(res: dict, schedule) -> pd.DataFrame:
    """Models x primary metrics on calibrated predictions, 'test' users, one block per test origin."""
    m = res["metrics"]
    if not len(m):
        return pd.DataFrame()
    keep = m[(m["split"] == "test") & (m["calibrated"]) & (m["segment_dim"] == "all")]
    rows = []
    for t in TARGETS:
        name, _ = M.PRIMARY[t]
        for extra in (name, "aggregate_bias" if t.startswith("exp") else "auc"):
            sel = keep[(keep["target"] == t) & (keep["metric"] == extra)]
            for (model, origin), v in sel.groupby(["model_id", "origin_week"])["value"].first().items():
                rows.append({"model_id": model, "column": f"{t} {extra} @{origin}", "value": v})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(index="model_id", columns="column", values="value")
