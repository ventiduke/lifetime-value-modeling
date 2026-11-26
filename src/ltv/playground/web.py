"""Export the playground as a static site (for GitHub Pages): no server, the models run in the browser.

    python -m ltv.playground --export-web docs/playground

writes ``index.html`` and ``engine.js`` (copies of the files beside this one), ``model.json``
(trees as flat arrays, the GRU's weights, the calibration maps, the calendar, the presets)
and ``players.json`` (a library of simulated players for the Simulate button).

**Only bundles trained on synthetic data can be exported.** A Pages site is public, and a
model trained on real players is not something to publish; ``export_site`` refuses a
bundle whose ``meta.json`` does not say ``"synthetic": true``. Real fits stay in the local
playground (docs/PLAYGROUND.md).

``engine.js`` is a JavaScript copy of the Python inference path. ``fixtures`` builds
players scored by the Python, and tests/test_playground.py checks the JavaScript against
them.
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import numpy as np

from ltv.playground import core

HERE = Path(__file__).parent
SITE_FILES = ("index.html", "engine.js")
A1_HEADS = ("any_active", "active_weeks", "hours", "any_spend", "log_spend_pos")
A2_HEADS = tuple(f"{k}_b{i}" for k in ("aw", "hrs", "spd") for i in range(5))
LINKS = {"binary": "sigmoid", "poisson": "exp", "tweedie": "exp", "regression": "identity"}
MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def _b64(a: np.ndarray, kind: str) -> dict:
    dtype = {"u8": "<u1", "i8": "<i1", "i32": "<i4", "f32": "<f4", "f64": "<f8"}[kind]
    a = np.ascontiguousarray(np.asarray(a).astype(dtype))
    return {"t": kind, "s": list(a.shape), "b": base64.b64encode(a.tobytes()).decode("ascii")}


def export_trees(state: dict, feature_names: list[str]) -> dict:
    """The boosters A1 and A2 need, as flat arrays: one row per split, one value per leaf."""
    import lightgbm as lgb
    feature, flags, threshold, left, right, value = [], [], [], [], [], []
    node_start, leaf_start, cats, boosters, n_trees = [0], [0], [], {}, 0
    for name in A1_HEADS + A2_HEADS:
        if name not in state["models"]:
            raise KeyError(f"the tree suite has no model {name!r}")
        dump = lgb.Booster(model_str=state["models"][name]).dump_model()
        if list(dump["feature_names"]) != list(feature_names):
            raise ValueError("the trees were trained on other features than FEATURE_NAMES")
        obj = dump["objective"].split()
        if obj[0] not in LINKS or (obj[0] == "binary" and "sigmoid:1" not in obj):
            raise ValueError(f"unsupported objective {dump['objective']!r} for {name}")
        boosters[name] = {"first": n_trees, "n": len(dump["tree_info"]), "link": LINKS[obj[0]]}
        for tree in dump["tree_info"]:
            nodes, leaves, stack = {}, {}, [tree["tree_structure"]]
            ref = lambda c: c["split_index"] if "split_index" in c else -c.get("leaf_index", 0) - 1
            while stack:
                nd = stack.pop()
                if "split_index" not in nd:
                    leaves[nd.get("leaf_index", 0)] = nd["leaf_value"]
                    continue
                cat = nd["decision_type"] == "=="
                if not cat and nd["decision_type"] != "<=":
                    raise ValueError(f"unsupported decision type {nd['decision_type']!r}")
                thr = float(nd["threshold"]) if not cat else float(len(cats))
                if cat:
                    cats.append([int(x) for x in str(nd["threshold"]).split("||")])
                nodes[nd["split_index"]] = (nd["split_feature"], int(bool(nd["default_left"])) | MISSING[nd["missing_type"]] << 1
                                            | int(cat) << 3, thr, ref(nd["left_child"]), ref(nd["right_child"]))
                stack += [nd["left_child"], nd["right_child"]]
            for i in range(len(nodes)):
                f, fl, th, lc, rc = nodes[i]
                feature.append(f); flags.append(fl); threshold.append(th); left.append(lc); right.append(rc)
            value += [leaves[i] for i in range(len(leaves))]
            node_start.append(len(feature)); leaf_start.append(len(value))
            n_trees += 1
    child = "i8" if max(np.abs(left + right), default=0) < 127 else "i32"
    return {"boosters": boosters, "cats": cats, "smear_spend": float(state["extras"].get("smear_spend", 1.0)),
            "arrays": {"feature": _b64(feature, "u8"), "flags": _b64(flags, "u8"), "threshold": _b64(threshold, "f64"),
                       "left": _b64(left, child), "right": _b64(right, child), "value": _b64(value, "f64"),
                       "node_start": _b64(node_start, "i32"), "leaf_start": _b64(leaf_start, "i32")}}


def export_gru(state: dict) -> dict:
    from ltv.data import seq_inputs as si
    cfg = state["cfg"]
    if cfg["encoder"] != "gru2res" or cfg["head"] != "ziln" or cfg["layers"] != 1 or not cfg["use_tabular"] or cfg["whale_aux"]:
        raise ValueError("the web engine implements the default C1 only (gru2res encoder, one layer, ZILN heads)")
    return {"cfg": {"d": cfg["d"]}, "cat_idx": list(si._CAT_IDX),
            "weights": {k: _b64(v, "f32") for k, v in state["weights"].items()}}


def export_model(bundle: core.Bundle) -> dict:
    """Everything the engine needs, as one JSON-ready dict."""
    from ltv.data.np_features import FEATURE_NAMES
    from ltv.train.timebox import load_pickle
    if not bundle.meta.get("synthetic"):
        raise PermissionError(
            "refusing to export: this bundle is not marked as trained on synthetic data. A static site is public; "
            "models trained on real players must not be published. Use the local playground for them.")
    trees = load_pickle(bundle.path / "trees" / f"suite_{bundle.fit}.pkl")
    gru = load_pickle(bundle.path / "seq" / "C1" / f"model_{bundle.fit}.pkl")
    models = [m for m in ("A1", "A2", "C1") if m in bundle.scorers]
    cal = {m: {t: {"x": [float(v) for v in mp.x], "v": [float(v) for v in mp.v]} for t, mp in r.maps.items()}
           for m, r in bundle.recal.items() if m in models}
    info = core.info(bundle)
    return {
        "meta": {"fit": bundle.fit, "origin_week": bundle.origin, "known_window": bundle.known_window, "models": models,
                 "synthetic": True, "world": bundle.meta.get("world"), "users": bundle.meta.get("users"),
                 "feature_names": list(FEATURE_NAMES), "labels": core.MODEL_LABELS, "worlds": core.WORLDS, "wants": core.WANTS},
        "calendar": {"season_week": [int(v) for v in bundle.season_week], "season_idx": [int(v) for v in bundle.season_idx]},
        "trees": export_trees(trees, list(FEATURE_NAMES)) if trees is not None and {"A1", "A2"} & set(models) else None,
        "gru": export_gru(gru) if gru is not None and "C1" in models else None,
        "calibration": cal, "presets": info["presets"],
    }


def export_players(bundle: core.Bundle, per_kind: int = 25, seed: int = 100) -> dict:
    """A library of simulated players per world and kind, stored sparsely (week, hours, spend)."""
    out = {}
    for world in core.WORLDS:
        out[world] = {}
        for want in core.WANTS:
            rows = []
            for k in range(per_kind):
                p = core.simulate(bundle, world, want, seed=seed + k)
                sparse = lambda h, s: [[i, a, b] for i, (a, b) in enumerate(zip(h, s)) if a > 0 or b > 0]
                rows.append({"n": len(p["hours"]), "h": sparse(p["hours"], p["spend"]),
                             "f": sparse(p["future_hours"], p["future_spend"]), "seed": p["seed"], "returns": p["returns"]})
            out[world][want] = rows
    return out


def fixtures(bundle: core.Bundle, n: int = 12, weeks: int = 40, step: int = 4) -> list[dict]:
    """Players with the Python's answers, for checking the JavaScript engine against."""
    from ltv.data.np_features import tabular_features
    cases = [(p["name"], p["hours"], p["spend"]) for p in core.presets(bundle.max_weeks)]
    cases += [("one week", [3.0], [0.0]), ("spend only", [0.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 2.5]),
              ("long and busy", [float(1 + (i * 7) % 11) for i in range(bundle.max_weeks)],
               [float((i % 9 == 0) * 4.99) for i in range(bundle.max_weeks)]),
              ("quiet for years", [6.0] * 20 + [0.0] * 150, [0.0] * 170)]
    for k in range(n):
        for world in core.WORLDS:
            p = core.simulate(bundle, world, list(core.WANTS)[k % len(core.WANTS)], seed=500 + k)
            cases.append((f"{world} {k}", p["hours"], p["spend"]))
    out = []
    for name, hours, spend in cases:
        store = core.make_store(bundle, hours, spend)
        x = tabular_features(store, np.array([0]), bundle.origin, known_window=bundle.known_window)[0]
        out.append({"name": name, "hours": [float(v) for v in hours], "spend": [float(v) for v in spend],
                    "features": [None if np.isnan(v) else float(v) for v in x],
                    "forecast": core.forecast(bundle, hours, spend), "timeline": core.timeline(bundle, hours, spend, weeks, step),
                    "weeks": weeks, "step": step})
    return out


def export_site(bundle: core.Bundle, out, per_kind: int = 25, with_fixtures: bool = False) -> Path:
    """Write the static site to ``out``. Returns the folder."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = export_model(bundle)                                   # refuses a non-synthetic bundle before writing anything
    (out / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    (out / "players.json").write_text(json.dumps(export_players(bundle, per_kind), separators=(",", ":")))
    for name in SITE_FILES:
        shutil.copyfile(HERE / name, out / name)
    (out / "mode.json").write_text('{"mode": "static"}')
    if with_fixtures:
        (out / "fixtures.json").write_text(json.dumps(fixtures(bundle)))
    return out
