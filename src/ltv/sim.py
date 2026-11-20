"""Simulation study: the bake-off on millions of synthetic users, without Spark.

Why this exists. The Spark pipeline (``ltv.pipeline``) is the thing that runs on Databricks.
On a small machine local Spark is the bottleneck long before the models are, so this module
drives the *same model, feature, calibration and evaluation code* straight from sequence
shards. It answers three questions before any real data is touched:

1. How fast is each family on CPU, and what does that imply for the time boxes?
2. How do the families behave as training data grows (learning curves)?
3. Which conclusions hold in two different synthetic worlds, and which flip?

What it shares with the pipeline: ``np_features`` (tested equal to the SQL features),
every model class, ``fit_hmm``, ``fit_seq``, the recalibration layer and ``evaluate``.
What it replaces: the tables. Features come from ``frame_from_store`` and predictions stay
in pickles under the output directory.

Every stage is cached on disk, so an interrupted run continues when called again.

    python -m ltv.sim --world rhythm --users 2000000 --out /path/to/dir

Synthetic results never choose the production model (AGENTS.md, rule 3). See
``reports/01_simulation_study.md`` for how to read the output.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ltv.config import Config, load_config
from ltv.data.sequences import SequenceStore
from ltv.data.splits import schedule_from_config
from ltv.models.base import CUM_COLS, LABEL_OF, TARGETS, WEEKLY_COLS, Preds, frame_from_store
from ltv.train.timebox import TimeBox, load_pickle, save_pickle

FITS = ("tune", "final_1", "final_2")
A1_MODELS = ("any_active", "active_weeks", "hours", "any_spend", "log_spend_pos")


@dataclass
class SimSettings:
    eval_shards: int = 8              # val and test users of this many shards are scored and evaluated
    tab_train_shards: int = 10        # shards sampled for the tabular training frame
    tab_train_rows: int = 2_000_000   # user-origin rows in the tabular training frame
    tab_rounds: int = 300             # cap on boosting rounds (early stopping usually ends sooner)
    hmm_train_shards: int = 1         # EM uses the train users of this many shards
    hmm_iters: int = 30               # EM passes for the tune fit
    hmm_final_iters: int = 10         # EM passes for each warm-started final fit
    seq_train_shards: int = 10
    seq_variants: tuple = ("C1",)
    seq_steps: int = 6000             # cap on training steps for the tune fit
    draws: int = 200                  # paired-bootstrap draws
    curves: bool = True               # learning curves on the validation origin
    tree_curve_rows: tuple = (30_000, 100_000, 300_000, 1_000_000)
    hmm_curve_users: tuple = (2_000, 10_000)
    seq_curve_users: tuple = (5_000, 30_000)


class Sim:
    def __init__(self, out_dir: str | Path, cfg: Config, settings: SimSettings | None = None):
        self.out, self.cfg, self.s = Path(out_dir), cfg, settings or SimSettings()
        self.shards = sorted((self.out / "shards").glob("shard_*.npz"))
        if not self.shards:
            raise RuntimeError(f"no shards under {self.out}/shards; run ltv.synth.worlds.write_world first")
        self._store0 = SequenceStore.load(self.shards[0])
        self.sched = schedule_from_config(self._store0.last_week, cfg)
        self.known = cfg["calendar"]["known_window"]
        self.lookback = int(cfg["population_lookback_weeks"])
        self.seed = int(cfg["training"]["seed"])
        self.edges = tuple(cfg["calendar"]["season_week_buckets"])
        self.eval_paths = self.shards[: self.s.eval_shards]
        for d in ("preds", "frames", "models", "results"):
            (self.out / d).mkdir(exist_ok=True)

    # -- bookkeeping ----------------------------------------------------------------
    @contextmanager
    def timed(self, stage: str, **info):
        rec = {"stage": stage, **info}
        t0 = time.time()
        yield rec
        rec["seconds"] = round(time.time() - t0, 2)
        with open(self.out / "timings.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"[sim] {json.dumps(rec, default=float)}", flush=True)

    def _pred_path(self, model: str, fit: str) -> Path:
        return self.out / "preds" / f"{model}_{fit}.pkl"

    def _save_preds(self, model: str, fit: str, p: Preds) -> None:
        p.state = None                                   # not evaluated; large
        save_pickle(self._pred_path(model, fit), p)

    def has(self, model: str, fit: str) -> bool:
        return self._pred_path(model, fit).exists()

    @staticmethod
    def _split(fit: str) -> str:
        return "val" if fit == "tune" else "test"

    def score_rows(self, store: SequenceStore, fit: str) -> np.ndarray:
        return store.rows_for(self.sched.score_origin(fit), self._split(fit), self.lookback)

    # -- frames -----------------------------------------------------------------------
    def score_frame(self, fit: str) -> pd.DataFrame:
        """Features and labels of the evaluated users at the fit's scoring origin."""
        path = self.out / "frames" / f"score_{fit}.pkl"
        if path.exists():
            return pd.read_pickle(path)
        origin = self.sched.score_origin(fit)
        with self.timed("features_score", fit=fit) as rec:
            parts = []
            for p in self.eval_paths:
                store = SequenceStore.load(p)
                parts.append(frame_from_store(store, self.score_rows(store, fit), origin, self.known))
            df = pd.concat(parts, ignore_index=True)
            rec["rows"] = len(df)
        df.to_pickle(path)
        return df

    def train_frame(self, fit: str) -> pd.DataFrame:
        """A sample of train-split user-origin rows over the fit's training origins."""
        path = self.out / "frames" / f"train_{fit}.pkl"
        if path.exists():
            return pd.read_pickle(path)
        origins = self.sched.train_origins(fit)
        paths = self.shards[: self.s.tab_train_shards]
        per = int(np.ceil(self.s.tab_train_rows / (len(paths) * len(origins))))
        rng = np.random.default_rng([self.seed, FITS.index(fit)])
        with self.timed("features_train", fit=fit, origins=len(origins)) as rec:
            parts = []
            for p in paths:
                store = SequenceStore.load(p)
                for o in origins:
                    rows = store.rows_for(o, "train", self.lookback)
                    if len(rows) > per:
                        rows = np.sort(rng.choice(rows, per, replace=False))
                    if len(rows):
                        parts.append(frame_from_store(store, rows, o, self.known))
            df = pd.concat(parts, ignore_index=True)
            rec["rows"] = len(df)
        df.to_pickle(path)
        return df

    def predict_shards(self, fit: str, fn) -> Preds:
        """``fn(store, rows) -> Preds`` over the evaluated users of every eval shard, joined."""
        parts = []
        for p in self.eval_paths:
            store = SequenceStore.load(p)
            rows = self.score_rows(store, fit)
            if len(rows):
                parts.append(fn(store, rows))
        return _concat_preds(parts)

    # -- baselines and trees ------------------------------------------------------------
    def run_baselines(self) -> None:
        from ltv.models.baselines import Persistence, SegmentLookup
        from ltv.models.bgbb import BGBB
        for fit in FITS:
            if all(self.has(m, fit) for m in ("R1", "R2", "R3")):
                continue
            tr, sc, origin = self.train_frame(fit), self.score_frame(fit), self.sched.score_origin(fit)
            for model in (SegmentLookup(), Persistence(), BGBB()):
                with self.timed("baseline", model=model.model_id, fit=fit, train_rows=len(tr), score_rows=len(sc)):
                    model.fit(tr)
                    self._save_preds(model.model_id, fit, model.predict(sc, origin))

    def run_trees(self) -> None:
        from ltv.models.trees import DEFAULT_PARAMS, TreeSuite
        state_path = self.out / "models" / "trees_tune.pkl"
        for fit in FITS:
            if all(self.has(m, fit) for m in ("A1", "A1tw", "A2")):
                continue
            tr, sc, origin = self.train_frame(fit), self.score_frame(fit), self.sched.score_origin(fit)
            if fit == "tune":
                suite = TreeSuite(params=dict(DEFAULT_PARAMS), max_rounds=self.s.tab_rounds, seed=self.seed)
                va = sc
            else:
                st = load_pickle(state_path)
                suite = TreeSuite(params=st["params"], max_rounds=st["max_rounds"], seed=self.seed,
                                  rounds=dict(st["rounds"]))
                va = None
            with self.timed("trees_fit", fit=fit, train_rows=len(tr), models=24) as rec:
                suite.fit(tr, va)
                rec["total_rounds"] = int(sum(suite.rounds.values()))
            if fit == "tune":
                save_pickle(state_path, suite.state())
            with self.timed("trees_score", fit=fit, score_rows=len(sc)):
                self._save_preds("A1", fit, suite.predict_a1(sc, origin))
                self._save_preds("A1tw", fit, suite.predict_a1(sc, origin, tweedie_spend=True))
                self._save_preds("A2", fit, suite.predict_a2(sc, origin))

    # -- HMM ----------------------------------------------------------------------------
    def _hmm_specs(self) -> dict:
        from ltv.models.hmm import spec_b1, spec_b2
        h, nb = self.cfg["hmm"], len(self.edges) + 1
        return {"B1": spec_b1(int(h["b1_states"][0]), nb),
                "B2": spec_b2(int(h["b2_types"][0]), int(h["b2_active_states"]), int(h["b2_dormant_states"]),
                              int(h["b2_spend_classes"]), nb)}

    def run_hmm(self) -> None:
        from ltv.models.hmm import HMM, fit_hmm, predict_hmm
        art = self.out / "models"
        train = self.shards[: self.s.hmm_train_shards]
        kw = dict(edges=self.edges, batch=int(self.cfg["hmm"]["batch_users"]), seed=self.seed)
        for fam, spec in self._hmm_specs().items():
            start = None
            for fit in FITS:
                origin = self.sched.score_origin(fit)
                ckpt = art / f"hmm_{fam}_{fit}.pkl"
                if self.has(fam, fit):
                    start = load_pickle(ckpt)["params"]
                    continue
                iters = self.s.hmm_iters if fit == "tune" else self.s.hmm_final_iters
                with self.timed("hmm_fit", model=fam, fit=fit, states=spec.n_states) as rec:
                    model, info = fit_hmm(spec, train, origin, TimeBox(600), ckpt, max_iters=iters,
                                          start=start if fit != "tune" else None, **kw)
                    rec.update(iters=info["iters"], train_ll=info["train_ll_per_week"],
                               train_users=_n_train(train, origin))
                start = start if fit != "tune" else model.export()
                with self.timed("hmm_score", model=fam, fit=fit) as rec:
                    p = self.predict_shards(fit, lambda st, rows: predict_hmm(
                        model, st, rows, origin, known_window=self.known, edges=self.edges))
                    rec["score_rows"] = len(p.user_ids)
                self._save_preds(fam, fit, p)

    # -- sequence models ------------------------------------------------------------------
    def _seq_kw(self) -> dict:
        s = self.cfg["seq"]
        return dict(batch=int(s["batch_users"]), lr=float(s["lr"]), seed=self.seed, known_window=self.known,
                    lookback=self.lookback, threads=self.cfg["budget"].get("torch_threads"))

    def run_seq(self) -> None:
        from ltv.models.seq import SeqConfig, predict_seq
        from ltv.pipeline import SEQ_VARIANTS
        from ltv.train.seq_fit import fit_seq, prepare_config, validation_batches
        s, art = self.cfg["seq"], self.out / "models"
        train = self.shards[: self.s.seq_train_shards]
        for variant in self.s.seq_variants:
            if all(self.has(variant, f) for f in FITS):
                continue
            base = SeqConfig(d=int(s["d"]), layers=int(s["layers"]), **SEQ_VARIANTS[variant])
            scfg = prepare_config(base, self._store0, self.sched.tune_train, self.known, self.seed,
                                  lookback=self.lookback)
            val = validation_batches(self._store0, self.sched.validation, scfg, self.known, lookback=self.lookback)
            best_step = None
            for fit in FITS:
                origin = self.sched.score_origin(fit)
                with self.timed("seq_fit", model=variant, fit=fit) as rec:
                    if fit == "tune":
                        model, info = fit_seq(scfg, train, self.sched.tune_train, TimeBox(600),
                                              art / f"seq_{variant}_tune.pt", val=val, max_steps=self.s.seq_steps,
                                              eval_every=int(s["eval_every"]), patience=int(s["patience"]),
                                              **self._seq_kw())
                        best_step = max(int(info["best_step"]), 1)
                    else:
                        model, info = fit_seq(scfg, train, self.sched.train_origins(fit), TimeBox(600),
                                              art / f"seq_{variant}_{fit}.pt", fixed_steps=best_step,
                                              **self._seq_kw())
                    rec.update(steps=info["steps"], best_step=info["best_step"],
                               steps_per_second=info["steps_per_second"], batch=int(s["batch_users"]))
                if self.has(variant, fit):
                    continue
                with self.timed("seq_score", model=variant, fit=fit) as rec:
                    p = self.predict_shards(fit, lambda st, rows: predict_seq(model, st, rows, origin, self.known))
                    rec["score_rows"] = len(p.user_ids)
                self._save_preds(variant, fit, p)

    # -- oracle (markov world only) ---------------------------------------------------------
    def run_oracle(self) -> None:
        from ltv.synth.generator import oracle_predict
        from ltv.synth.worlds import load_truth
        for fit in FITS:
            if self.has("ORACLE", fit):
                continue
            origin = self.sched.score_origin(fit)
            parts = []
            for path in self.eval_paths:
                truth = load_truth(path)
                store = SequenceStore.load(path)
                rows = self.score_rows(store, fit)
                o = oracle_predict(truth, store.season_week[origin + 1: origin + 53], origin).set_index("user_id")
                o = o.loc[store.user_ids[rows]]
                parts.append(Preds(store.user_ids[rows], origin, o["p_reengage"].to_numpy(),
                                   o["exp_active_weeks"].to_numpy(), o["exp_hours"].to_numpy(),
                                   o["p_spend"].to_numpy(), o["exp_spend"].to_numpy(),
                                   cum_hours=np.stack(o["cum_hours"].to_numpy()),
                                   cum_spend=np.stack(o["cum_spend"].to_numpy()),
                                   weekly_p_active=np.stack(o["weekly_p_active"].to_numpy())))
            self._save_preds("ORACLE", fit, _concat_preds(parts))

    # -- evaluation -------------------------------------------------------------------------
    def eval_frame(self) -> pd.DataFrame:
        drop = CUM_COLS["hours"] + CUM_COLS["spend"] + WEEKLY_COLS + ["user_id", "origin_week"]
        parts = []
        for fit in FITS:
            sc = self.score_frame(fit)
            side = sc.drop(columns=drop)
            for path in sorted((self.out / "preds").glob(f"*_{fit}.pkl")):
                model = path.name[: -len(f"_{fit}.pkl")]
                p = load_pickle(path)
                if not np.array_equal(p.user_ids, sc["user_id"].to_numpy()):
                    raise RuntimeError(f"{model}/{fit}: predictions are not aligned with the score frame")
                parts.append(pd.concat([p.frame(model, fit, as_lists=False), side], axis=1))
        return pd.concat(parts, ignore_index=True)

    def run_eval(self) -> dict:
        from ltv.eval.evaluate import evaluate, scorecard
        from ltv.pipeline import write_results_report
        with self.timed("evaluate") as rec:
            frame = self.eval_frame()
            rec.update(rows=len(frame), models=int(frame["model_id"].nunique()))
            res = evaluate(frame, self.sched, self.s.draws, self.seed, int(self.cfg["evaluation"]["top_countries"]))
            res["scorecard"] = scorecard(res, self.sched)
        for k in ("metrics", "coherence", "intervals", "pairwise", "winners"):
            res[k].to_csv(self.out / "results" / f"{k}.csv", index=False)
        res["scorecard"].to_csv(self.out / "results" / "scorecard.csv")
        write_results_report(res, self.sched, self.out / "results" / "bakeoff_results.md", synthetic=True)
        return res

    # -- learning curves ----------------------------------------------------------------------
    def _curve_point(self, family: str, size: int, unit: str, p: Preds, seconds: float, sc: pd.DataFrame) -> dict:
        return {"family": family, "size": int(size), "unit": unit, "seconds": round(seconds, 1),
                **primary_metrics(sc, p)}

    def run_curves(self) -> pd.DataFrame:
        """Validation-origin metrics (raw, before recalibration) as the training data grows."""
        from ltv.models.hmm import fit_hmm, predict_hmm
        from ltv.models.seq import SeqConfig, predict_seq
        from ltv.models.trees import DEFAULT_PARAMS, TreeSuite
        from ltv.pipeline import SEQ_VARIANTS
        from ltv.train.seq_fit import fit_seq, prepare_config, validation_batches
        path = self.out / "results" / "learning_curves.pkl"
        done = load_pickle(path, {})
        sc, origin = self.score_frame("tune"), self.sched.validation

        def record(key, point):
            done[key] = point
            save_pickle(path, done)
            print(f"[sim] curve {point}", flush=True)

        tr = self.train_frame("tune")
        for n in self.s.tree_curve_rows:
            if ("A1", n) in done or n >= len(tr):
                continue
            t0 = time.time()
            suite = TreeSuite(params=dict(DEFAULT_PARAMS), max_rounds=self.s.tab_rounds, seed=self.seed)
            suite.fit(tr.sample(n=n, random_state=self.seed), sc, names=A1_MODELS)
            record(("A1", n), self._curve_point("A1", n, "user-origin rows", suite.predict_a1(sc, origin),
                                                time.time() - t0, sc))
        del tr
        kw = dict(edges=self.edges, batch=int(self.cfg["hmm"]["batch_users"]), seed=self.seed)
        spec = self._hmm_specs()["B2"]
        for n in self.s.hmm_curve_users:
            if ("B2", n) in done:
                continue
            t0 = time.time()
            sub = self._store0.subset(np.flatnonzero(self._store0.split == "train")[:n])
            model, _ = fit_hmm(spec, [sub], origin, TimeBox(600), None, max_iters=self.s.hmm_iters, **kw)
            p = self.predict_shards("tune", lambda st, rows: predict_hmm(model, st, rows, origin,
                                                                         known_window=self.known, edges=self.edges))
            record(("B2", n), self._curve_point("B2", n, "train users", p, time.time() - t0, sc))
        s = self.cfg["seq"]
        variant = self.s.seq_variants[0] if self.s.seq_variants else None
        if variant:
            base = SeqConfig(d=int(s["d"]), layers=int(s["layers"]), **SEQ_VARIANTS[variant])
            scfg = prepare_config(base, self._store0, self.sched.tune_train, self.known, self.seed,
                                  lookback=self.lookback)
            val = validation_batches(self._store0, origin, scfg, self.known, lookback=self.lookback)
            for n in self.s.seq_curve_users:
                if (variant, n) in done:
                    continue
                t0 = time.time()
                model, _ = fit_seq(scfg, [self._store0], self.sched.tune_train, TimeBox(600), None, val=val,
                                   max_steps=self.s.seq_steps, eval_every=int(s["eval_every"]),
                                   patience=int(s["patience"]), max_users=n, **self._seq_kw())
                p = self.predict_shards("tune", lambda st, rows: predict_seq(model, st, rows, origin, self.known))
                record((variant, n), self._curve_point(variant, n, "train users (row index below)", p,
                                                       time.time() - t0, sc))
        # the full-size points come from the main run
        for model, unit in (("A1", "user-origin rows"), ("B2", "train users"), (variant, "train users")):
            if model and self.has(model, "tune"):
                size = {"A1": self.s.tab_train_rows, "B2": _n_train(self.shards[: self.s.hmm_train_shards], origin)}.get(
                    model, _n_train(self.shards[: self.s.seq_train_shards], origin))
                done[(model, "main")] = self._curve_point(model, size, unit + " (main run)",
                                                          load_pickle(self._pred_path(model, "tune")), float("nan"), sc)
        save_pickle(path, done)
        df = pd.DataFrame(list(done.values())).sort_values(["family", "size"]).reset_index(drop=True)
        df.to_csv(self.out / "results" / "learning_curves.csv", index=False)
        return df

    def run_all(self, world: str | None = None) -> dict:
        self.run_baselines()
        self.run_trees()
        self.run_hmm()
        self.run_seq()
        if world == "markov":
            self.run_oracle()
        res = self.run_eval()
        if self.s.curves:
            res["curves"] = self.run_curves()
        return res


def primary_metrics(sc: pd.DataFrame, p: Preds) -> dict:
    """The primary metric of each target, plus AUC and aggregate bias, on raw predictions."""
    from ltv.eval.metrics import PRIMARY, target_metrics
    out = {}
    for t in TARGETS:
        m = target_metrics(t, sc[LABEL_OF[t]].to_numpy(float), getattr(p, t))
        out[f"{t}:{PRIMARY[t][0]}"] = m[PRIMARY[t][0]]
        out[f"{t}:{'auc' if 'auc' in m else 'aggregate_bias'}"] = m["auc" if "auc" in m else "aggregate_bias"]
    return out


def _concat_preds(parts: list) -> Preds:
    def cat(name):
        xs = [getattr(p, name) for p in parts]
        return None if xs[0] is None else np.concatenate(xs)
    return Preds(cat("user_ids"), parts[0].origin, cat("p_reengage"), cat("exp_active_weeks"), cat("exp_hours"),
                 cat("p_spend"), cat("exp_spend"), cat("cum_hours"), cat("cum_spend"), cat("weekly_p_active"),
                 cat("q_hours"), cat("q_spend"), None)


def _n_train(shards: list, origin: int) -> int:
    n = 0
    for p in shards:
        st = SequenceStore.load(p)
        n += int(((st.split == "train") & (st.first_week <= origin)).sum())
    return n


def run(world: str, n_users: int, out_dir: str | Path, settings: SimSettings | None = None,
        n_weeks: int = 320, seed: int = 7, overrides: dict | None = None) -> dict:
    """Generate a world (if not there yet) and run the bake-off on it."""
    from ltv.synth.worlds import write_world
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "conf" / "bakeoff.yml", root / "conf" / "synth.yml",
                      overrides={"artifact_path": str(out), **(overrides or {})})
    shard_users = int(cfg["training"]["seq_shard_users"])
    if len(list((out / "shards").glob("shard_*.npz"))) < int(np.ceil(n_users / shard_users)):
        t0 = time.time()
        write_world(world, n_users, out, n_weeks, seed, shard_users)
        with open(out / "timings.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"stage": "generate", "world": world, "users": n_users, "weeks": n_weeks,
                                "seconds": round(time.time() - t0, 1)}) + "\n")
    sim = Sim(out, cfg, settings)
    save_pickle(out / "settings.pkl", {"world": world, "users": n_users, **asdict(sim.s)})
    return sim.run_all(world)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--world", required=True, choices=("markov", "rhythm"))
    ap.add_argument("--users", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    for name, default in asdict(SimSettings()).items():
        if isinstance(default, bool):
            ap.add_argument(f"--{name}", type=lambda v: v.lower() in ("1", "true", "yes"), default=default)
        elif isinstance(default, tuple):
            kind = str if name == "seq_variants" else int
            ap.add_argument(f"--{name}", type=lambda v, k=kind: tuple(k(x) for x in v.split(",") if x), default=default)
        else:
            ap.add_argument(f"--{name}", type=type(default), default=default)
    a = vars(ap.parse_args())
    settings = SimSettings(**{k: a[k] for k in asdict(SimSettings())})
    res = run(a["world"], a["users"], a["out"], settings, a["weeks"], a["seed"])
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res["scorecard"].round(4))
        if "curves" in res:
            print(res["curves"].round(4))


if __name__ == "__main__":
    main()
