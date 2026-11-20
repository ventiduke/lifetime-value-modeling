"""Configuration loading.

One YAML file (conf/bakeoff.yml) holds every real name and setting. A second file
(conf/synth.yml) can be layered on top for the synthetic smoke test. Code never hardcodes
a table name: it asks the config.

    cfg = load_config("conf/bakeoff.yml")                      # real data
    cfg = load_config("conf/bakeoff.yml", "conf/synth.yml")    # synthetic smoke test
    cfg.table("labels")       -> "main.ltv_bakeoff.labels"
    cfg["horizon_weeks"]      -> 52
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Every table the pipeline writes. docs/DATA_CONTRACT.md documents each one.
OUTPUT_TABLES = (
    "activity_clean", "calendar_weeks", "dim_platform", "dim_country", "users", "origins",
    "user_origins", "calendar_projected", "panel_weekly", "features_tabular", "labels",
    "user_sequences", "predictions", "eval_metrics", "eval_pairwise", "hmm_user_state",
    "score_user_origins", "score_features", "score_predictions", "cost_timings",
    "example_users", "example_weekly",
)


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _check_table_name(name: str, what: str) -> None:
    parts = name.split(".")
    if not 1 <= len(parts) <= 3 or not all(_IDENT.match(p) for p in parts):
        raise ValueError(f"{what}: {name!r} is not a valid [catalog.][schema.]table name")


class Config(dict):
    """A dict with a few helpers. Nested values stay plain dicts/lists."""

    def table(self, name: str) -> str:
        """Fully qualified name of a table this pipeline writes."""
        if name not in OUTPUT_TABLES:
            raise KeyError(f"{name!r} is not a pipeline table; see ltv.config.OUTPUT_TABLES")
        parts = [self.get("catalog"), self["schema"], name]
        return ".".join(p for p in parts if p)

    def tables(self) -> dict[str, str]:
        return {t: self.table(t) for t in OUTPUT_TABLES}

    @property
    def horizon(self) -> int:
        return int(self["horizon_weeks"])

    def budget_minutes(self) -> float:
        """Total training budget per model family, in minutes, for the active profile."""
        prof = self["budget"]["profile"]
        return float(self["budget"][prof]["family_hours"]) * 60.0

    def artifact_dir(self, *parts: str) -> Path:
        p = Path(self["artifact_path"]).joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def validate(cfg: Config) -> None:
    """Fail early, with a message that says what to fix."""
    for key in ("schema", "artifact_path", "source", "splits", "horizon_weeks"):
        if key not in cfg:
            raise ValueError(f"config is missing required key {key!r}")
    if cfg.get("catalog") is not None and not _IDENT.match(str(cfg["catalog"])):
        raise ValueError(f"catalog {cfg['catalog']!r} is not a valid identifier")
    if not _IDENT.match(str(cfg["schema"])):
        raise ValueError(f"schema {cfg['schema']!r} is not a valid identifier")
    src = cfg["source"]
    _check_table_name(src["activity_table"], "source.activity_table")
    _check_table_name(src["calendar_table"], "source.calendar_table")
    if src.get("exclude_users_table"):
        _check_table_name(src["exclude_users_table"], "source.exclude_users_table")
    for group in ("activity_columns", "calendar_columns"):
        for logical, real in src[group].items():
            if not _IDENT.match(str(real)):
                raise ValueError(f"source.{group}.{logical}: {real!r} is not a valid column name")
    need_a = {"user_id", "week_start", "hours", "spend", "platform", "country"}
    need_c = {"week_start", "season", "season_week"}
    if set(src["activity_columns"]) != need_a:
        raise ValueError(f"source.activity_columns must map exactly {sorted(need_a)}")
    if set(src["calendar_columns"]) != need_c:
        raise ValueError(f"source.calendar_columns must map exactly {sorted(need_c)}")
    if cfg["horizon_weeks"] != max(cfg["cum_horizons"]):
        raise ValueError("the last cum_horizons entry must equal horizon_weeks")
    if list(cfg["cum_horizons"]) != sorted(cfg["cum_horizons"]):
        raise ValueError("cum_horizons must be increasing")
    if sum(cfg["splits"]["user_split_pct"]) != 100:
        raise ValueError("splits.user_split_pct must sum to 100")
    if not 0 < float(cfg["sample"]["fraction"]) <= 1:
        raise ValueError("sample.fraction must be in (0, 1]")
    if cfg["cleaning"]["negative_spend"] not in ("clip", "drop"):
        raise ValueError("cleaning.negative_spend must be 'clip' or 'drop'")
    if cfg["budget"]["profile"] not in ("quick", "full"):
        raise ValueError("budget.profile must be 'quick' or 'full'")


def load_config(*paths: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load one or more YAML files, later files overriding earlier ones."""
    merged: dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})
    if overrides:
        merged = _deep_merge(merged, overrides)
    cfg = Config(merged)
    validate(cfg)
    return cfg
