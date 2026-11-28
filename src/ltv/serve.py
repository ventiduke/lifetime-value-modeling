"""A fitted model as one MLflow model: histories in, the five 52-week outputs out.

The bake-off saves every fit as it trains (``trees/suite_<fit>.pkl``, ``seq/<variant>/
model_<fit>.pkl``). ``log_fit`` packages one of those saved fits for MLflow **without
retraining**: the fit, its calibration layer, the season calendar and this package's code go
into one ``mlflow.pyfunc`` model, which can be registered in Unity Catalog and loaded
anywhere MLflow runs.

Every model, tree or GRU, has the same contract, so the caller does not change when the
winner does:

    input    one row per player, the columns of the ``user_sequences`` table:
             user_id, first_week, hours[], spend[] (and optionally platform_id[], country_id[]);
             element i of an array is week first_week + i
    params   origin_week   the scoring week (-1 = the last week the arrays reach)
             calibrated    apply the calibration layer (default true)
             season_week, season_idx   the current season calendar, indexed by week, when it
                                       is newer than the one saved with the model
    output   user_id, origin_week, in_population, p_reengage, exp_active_weeks, exp_hours,
             p_spend, exp_spend

The trees get their features rebuilt from the history by ``np_features.py`` (tested equal
to the SQL), so they need no feature table at scoring time. Nothing after ``origin_week``
is read. ``in_population`` is false for a player with no activity or spend in the 104 weeks
up to the origin: the models were not trained on such players.
"""
from __future__ import annotations

import inspect
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

try:                                    # the pipeline imports this module lazily; MLflow is on Databricks
    import mlflow
    from mlflow.pyfunc import PythonModel
except ImportError:                     # pragma: no cover
    mlflow = None
    PythonModel = object

TARGET_COLS = ("p_reengage", "exp_active_weeks", "exp_hours", "p_spend", "exp_spend")
OUTPUT_COLS = ("user_id", "origin_week", "in_population", *TARGET_COLS)
NO_CALENDAR = [-1]                      # params default: "use the calendar saved with the model"
LOOKBACK = 104
TREE_IDS = ("A1", "A1tw", "A2")


def _need_mlflow() -> None:
    if mlflow is None:
        raise ImportError("MLflow is not installed. On Databricks use the ML base environment; locally `pip install mlflow`.")


def _maps_to_json(recal) -> dict:
    return {t: {"x": [float(v) for v in m.x], "v": [float(v) for v in m.v], "prob": bool(m.prob)} for t, m in recal.maps.items()}


def _maps_from_json(d: dict):
    from ltv.eval.calibration import MonotoneMap, Recalibrator
    r = Recalibrator()
    for t, m in d.items():
        mp = MonotoneMap(prob=bool(m["prob"]))
        mp.x, mp.v = np.asarray(m["x"], float), np.asarray(m["v"], float)
        r.maps[t] = mp
    return r


def store_from_rows(rows: pd.DataFrame, season_week, season_idx):
    """A SequenceStore from rows shaped like ``user_sequences``. Returns (store, last week covered)."""
    from ltv.data.sequences import SequenceStore
    n = len(rows)
    first = rows["first_week"].to_numpy(dtype=np.int64)
    hours = [np.asarray(a, np.float32) for a in rows["hours"]]
    spend = [np.asarray(a, np.float32) for a in rows["spend"]]
    if any(len(h) != len(s) for h, s in zip(hours, spend)):
        raise ValueError("hours and spend must have the same number of weeks for every player")
    if (first < 0).any():
        raise ValueError("first_week must be 0 or more")
    last = int(max((f + len(h) - 1 for f, h in zip(first, hours)), default=-1))
    w = last + 1
    H, S = np.zeros((n, w), np.float32), np.zeros((n, w), np.float32)
    P, C = np.zeros((n, w), np.int16), np.zeros((n, w), np.int16)
    for i in range(n):
        f, k = int(first[i]), len(hours[i])
        H[i, f:f + k], S[i, f:f + k] = hours[i], spend[i]
        row = (hours[i] > 0) | (spend[i] > 0)
        for col, dst in (("platform_id", P), ("country_id", C)):
            given = rows[col].iloc[i] if col in rows.columns else None
            dst[i, f:f + k] = np.asarray(given, np.int16) if given is not None and len(given) == k else row.astype(np.int16)
    store = SequenceStore(rows["user_id"].to_numpy(dtype=object).astype(str), np.full(n, "score"), first.astype(np.int32),
                          H, S, P, C, np.asarray(season_week, np.int16), np.asarray(season_idx, np.int16))
    return store, last


class LTVModel(PythonModel):
    """One saved fit. See the module docstring for the contract."""

    def load_context(self, context) -> None:
        from ltv.eval.examples import saved_scorers
        path = Path(context.artifacts["bundle"])
        self.meta = json.loads((path / "meta.json").read_text())
        mid = self.meta["model_id"]
        scorers = saved_scorers(path, self.meta["fit"], [mid], self.meta.get("known_window", "current_season"))
        if mid not in scorers:
            raise FileNotFoundError(f"the model files for {mid} ({self.meta['fit']}) are missing from this MLflow model")
        self.scorer = scorers[mid]
        cal_file = path / "calibration.json"
        self.recal = _maps_from_json(json.loads(cal_file.read_text())) if cal_file.exists() else None
        cal = pd.read_csv(path / "calendar.csv").sort_values("week_idx")
        self.season_week, self.season_idx = cal["season_week"].to_numpy(), cal["season_idx"].to_numpy()

    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        from ltv.eval.examples import _calibrate
        params = dict(params or {})
        rows = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        if not len(rows):
            return pd.DataFrame({c: [] for c in OUTPUT_COLS})
        sw, si = params.get("season_week"), params.get("season_idx")
        given = sw is not None and len(sw) > 0 and list(sw) != NO_CALENDAR
        sw, si = (np.asarray(sw), np.asarray(si)) if given else (self.season_week, self.season_idx)
        if len(sw) != len(si):
            raise ValueError("season_week and season_idx must have the same length")
        store, last = store_from_rows(rows, sw, si)
        origin = int(params.get("origin_week", -1))
        origin = last if origin < 0 else origin
        if origin >= len(sw):
            raise ValueError(f"origin_week {origin} is beyond the season calendar this model holds ({len(sw)} weeks). "
                             "Pass the current calendar in params: season_week and season_idx, indexed by week.")
        if origin > last:                                   # scoring later than the data: the weeks between are quiet
            pad = lambda a: np.concatenate([a, np.zeros((a.shape[0], origin - last), a.dtype)], axis=1)
            store.hours, store.spend, store.platform, store.country = (pad(a) for a in (store.hours, store.spend, store.platform, store.country))
        out = pd.DataFrame({"user_id": store.user_ids, "origin_week": np.int64(origin), "in_population": False,
                            **{t: np.nan for t in TARGET_COLS}})
        alive = np.flatnonzero(store.first_week <= origin)      # a player who has not arrived yet has no forecast
        if len(alive):
            p = self.scorer(store, alive, origin)
            if params.get("calibrated", True) and self.recal is not None:
                p = _calibrate(p, self.recal)
            for t in TARGET_COLS:
                out.loc[alive, t] = np.asarray(getattr(p, t), float)
            lo = max(origin - LOOKBACK + 1, 0)
            recent = ((store.hours[alive, lo:origin + 1] > 0) | (store.spend[alive, lo:origin + 1] > 0)).any(axis=1)
            out.loc[alive, "in_population"] = recent
        return out


def signature():
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema
    from mlflow.types.schema import Array
    inputs = Schema([ColSpec(DataType.string, "user_id"), ColSpec(DataType.long, "first_week"),
                     ColSpec(Array(DataType.double), "hours"), ColSpec(Array(DataType.double), "spend"),
                     ColSpec(Array(DataType.long), "platform_id", required=False),
                     ColSpec(Array(DataType.long), "country_id", required=False)])
    outputs = Schema([ColSpec(DataType.string, "user_id"), ColSpec(DataType.long, "origin_week"),
                      ColSpec(DataType.boolean, "in_population"), *[ColSpec(DataType.double, t) for t in TARGET_COLS]])
    params = ParamSchema([ParamSpec("origin_week", DataType.long, -1), ParamSpec("calibrated", DataType.boolean, True),
                          ParamSpec("season_week", DataType.long, NO_CALENDAR, (-1,)),
                          ParamSpec("season_idx", DataType.long, NO_CALENDAR, (-1,))])
    return ModelSignature(inputs=inputs, outputs=outputs, params=params)


def _requirements(model_id: str) -> list[str]:
    """Pinned to what is installed where the model is logged, without local build tags such as +cpu."""
    import importlib.metadata as md
    names = ["numpy", "pandas", "scipy", "pyyaml"] + (["lightgbm"] if model_id in TREE_IDS else ["torch"])
    return [f"{n}=={md.version(n).split('+')[0]}" for n in names]


def model_files(artifact_path, model_id: str, fit: str) -> Path:
    """The saved fit behind a model id, relative layout kept: trees/... or seq/<variant>/..."""
    art = Path(artifact_path)
    rel = Path("trees") / f"suite_{fit}.pkl" if model_id in TREE_IDS else Path("seq") / model_id / f"model_{fit}.pkl"
    if not (art / rel).exists():
        raise FileNotFoundError(f"no saved '{fit}' fit for {model_id} under {art} (expected {rel}). Run its notebook first.")
    return rel


def log_fit(artifact_path, model_id: str, fit: str, calendar: pd.DataFrame, origin_week: int, recal=None,
            known_window="current_season", registered_name: str | None = None, calibration: str = "none",
            extra: dict | None = None) -> dict:
    """Log one saved fit as an MLflow model, and register it when ``registered_name`` is given.

    Nothing is trained. ``recal`` is the fitted calibration layer (a ``Recalibrator``) or
    None; ``calendar`` has week_idx, season_week, season_idx. A three-part name
    (``catalog.schema.model``) registers in Unity Catalog. Returns the model URI and, when
    registered, the name and version.
    """
    _need_mlflow()
    import ltv
    rel = model_files(artifact_path, model_id, fit)
    meta = {"model_id": model_id, "fit": fit, "origin_week": int(origin_week), "known_window": known_window,
            "calibration": calibration if recal is not None else "none", "source": "ltv.serve.log_fit", **(extra or {})}
    with tempfile.TemporaryDirectory() as tmp:
        bundle, code = Path(tmp) / "bundle", Path(tmp) / "ltv"
        (bundle / rel).parent.mkdir(parents=True)
        shutil.copyfile(Path(artifact_path) / rel, bundle / rel)
        if recal is not None:
            (bundle / "calibration.json").write_text(json.dumps(_maps_to_json(recal)))
        calendar[["week_idx", "season_week", "season_idx"]].sort_values("week_idx").to_csv(bundle / "calendar.csv", index=False)
        (bundle / "meta.json").write_text(json.dumps(meta))
        shutil.copytree(Path(ltv.__file__).parent, code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if registered_name and registered_name.count(".") == 2:
            mlflow.set_registry_uri("databricks-uc")
        example = pd.DataFrame({"user_id": ["example-1", "example-2"], "first_week": [max(int(origin_week) - 5, 0)] * 2,
                                "hours": [[2.0, 0.0, 3.5, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 4.0]],
                                "spend": [[0.0, 0.0, 4.99, 0.0, 0.0, 0.0], [0.0] * 6]})
        kw = dict(python_model=LTVModel(), artifacts={"bundle": str(bundle)}, code_paths=[str(code)],
                  pip_requirements=_requirements(model_id), signature=signature(), registered_model_name=registered_name,
                  metadata={k: str(v) for k, v in meta.items()},
                  input_example=(example, {"origin_week": int(origin_week), "calibrated": True}))
        named = "name" in inspect.signature(mlflow.pyfunc.log_model).parameters       # MLflow 3; MLflow 2 has artifact_path
        kw["name" if named else "artifact_path"] = f"ltv_{model_id}"
        info = mlflow.pyfunc.log_model(**kw)
    out = {"model_uri": info.model_uri, "model_id": model_id, "fit": fit, "calibration": meta["calibration"],
           "registered_name": registered_name, "version": None}
    if registered_name:
        version = getattr(info, "registered_model_version", None)
        if version is None:
            found = mlflow.MlflowClient().search_model_versions(f"name='{registered_name}'")
            version = max((int(v.version) for v in found), default=None)
        out["version"] = None if version is None else int(version)
    return out


def set_alias(registered_name: str, alias: str, version: int) -> None:
    """Point an alias (for example ``candidate`` or ``champion``) at a version of a registered model."""
    _need_mlflow()
    if registered_name.count(".") == 2:
        mlflow.set_registry_uri("databricks-uc")
    mlflow.MlflowClient().set_registered_model_alias(registered_name, alias, str(version))


def load(model_uri: str):
    """Load a logged or registered model, for example ``models:/catalog.schema.ltv_c1@champion``."""
    _need_mlflow()
    if model_uri.startswith("models:/") and model_uri[len("models:/"):].split("@")[0].split("/")[0].count(".") == 2:
        mlflow.set_registry_uri("databricks-uc")
    return mlflow.pyfunc.load_model(model_uri)


def calendar_params(calendar: pd.DataFrame | None) -> dict:
    if calendar is None:
        return {}
    cal = calendar.sort_values("week_idx")
    return {"season_week": [int(v) for v in cal["season_week"]], "season_idx": [int(v) for v in cal["season_idx"]]}


def predict(model, rows: pd.DataFrame, origin_week: int | None = None, calendar: pd.DataFrame | None = None,
            calibrated: bool = True) -> pd.DataFrame:
    """Score rows shaped like ``user_sequences`` with a loaded model."""
    cols = [c for c in ("user_id", "first_week", "hours", "spend", "platform_id", "country_id") if c in rows.columns]
    params = {"origin_week": -1 if origin_week is None else int(origin_week), "calibrated": bool(calibrated),
              **calendar_params(calendar)}
    return model.predict(rows[cols], params=params)


def score_sequences(spark, sequences_table: str, model_uri: str, origin_week: int | None = None,
                    calendar: pd.DataFrame | None = None, calibrated: bool = True, n_slices: int = 1,
                    where: str | None = None) -> pd.DataFrame:
    """Batch scoring on the driver: read ``sequences_table`` in hashed slices of players and
    score each slice. Memory is one slice at a time. It uses nothing Spark Connect lacks."""
    model, sdf, parts = load(model_uri), spark.table(sequences_table), []
    if where:
        sdf = sdf.where(where)
    for i in range(max(int(n_slices), 1)):
        part = sdf if n_slices <= 1 else sdf.where(f"PMOD(XXHASH64(user_id, 'slice'), {int(n_slices)}) = {i}")
        pdf = part.select("user_id", "first_week", "hours", "spend", "platform_id", "country_id").toPandas()
        if len(pdf):
            parts.append(predict(model, pdf, origin_week, calendar, calibrated))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({c: [] for c in OUTPUT_COLS})
