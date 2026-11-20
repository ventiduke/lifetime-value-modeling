"""The synthetic worlds and the Spark-free simulation driver (``ltv.sim``)."""
from pathlib import Path

import numpy as np
import pytest

from ltv.config import load_config
from ltv.data.sequences import SequenceStore
from ltv.sim import Sim, SimSettings
from ltv.synth import worlds
from ltv.synth.generator import generate

ROOT = Path(__file__).resolve().parents[1]


def test_generate_is_unchanged_by_the_dense_refactor():
    a, b = generate(500, 120, seed=3), generate(500, 120, seed=3)
    assert a.activity.equals(b.activity)
    assert set(a.truth) >= {"states", "hours", "spend", "user_type", "first_week", "user_ids"}
    assert len(a.activity) == int(((a.truth["hours"] > 0) | (a.truth["spend"] > 0)).sum())


@pytest.mark.parametrize("world", worlds.WORLDS)
def test_worlds_write_resumable_shards(world, tmp_path):
    paths = worlds.write_world(world, 2500, tmp_path, n_weeks=200, seed=5, shard_users=1000)
    assert [p.name for p in paths] == ["shard_00000.npz", "shard_00001.npz", "shard_00002.npz"]
    stores = [SequenceStore.load(p) for p in paths]
    assert [s.n_users for s in stores] == [1000, 1000, 500] and stores[0].n_weeks == 200
    ids = np.concatenate([s.user_ids for s in stores])
    assert len(set(ids)) == 2500                                    # ids are unique across shards
    assert all((s.season_week == stores[0].season_week).all() for s in stores)   # one calendar for all
    for s in stores:
        first = (s.hours > 0).argmax(axis=1)
        assert (first == s.first_week).all()                        # the first week is active
        assert (s.platform[s.hours > 0] > 0).all() and (s.platform[(s.hours == 0) & (s.spend == 0)] == 0).all()
    assert set(np.unique(stores[0].split)) == {"train", "val", "test"}
    truth = worlds.load_truth(paths[0])
    assert truth["states"].shape == (1000, 200) and (truth["hours"] == stores[0].hours).all()
    # a second call keeps what is there and gives the same data
    before = paths[1].stat().st_mtime_ns
    worlds.write_world(world, 2500, tmp_path, n_weeks=200, seed=5, shard_users=1000)
    assert paths[1].stat().st_mtime_ns == before
    again = worlds.SIMULATORS[world](1000, 200, worlds.world_calendar(200, 5)["season_week_number"].to_numpy(),
                                     np.random.default_rng([5, 1, 0]))
    assert (again["hours"] == stores[0].hours).all()


def test_rhythm_world_has_personal_lapse_lengths():
    """What makes the world non-Markov: a user's lapses cluster around their own typical length."""
    cal = worlds.world_calendar(320, 7)
    sim = worlds.simulate_rhythm(6000, 320, cal["season_week_number"].to_numpy(), np.random.default_rng(0))
    act = sim["hours"] > 0
    first, second = [], []
    for row in act:
        gaps = np.diff(np.flatnonzero(row)) - 1
        gaps = gaps[gaps >= 4]
        if len(gaps) >= 2:
            first.append(gaps[0]); second.append(gaps[1])
    r = np.corrcoef(np.log(first), np.log(second))[0, 1]
    assert len(first) > 1000 and r > 0.25           # in the Markov world this correlation comes only from types


@pytest.mark.slow
def test_sim_runs_every_stage_and_resumes(tmp_path):
    worlds.write_world("rhythm", 6000, tmp_path, n_weeks=320, seed=3, shard_users=3000)
    cfg = load_config(ROOT / "conf" / "bakeoff.yml", ROOT / "conf" / "synth.yml",
                      overrides={"artifact_path": str(tmp_path)})
    s = SimSettings(eval_shards=2, tab_train_shards=2, tab_train_rows=20_000, tab_rounds=30, hmm_iters=3,
                    hmm_final_iters=1, seq_train_shards=1, seq_steps=60, draws=10, tree_curve_rows=(5000,),
                    hmm_curve_users=(500,), seq_curve_users=(500,))
    sim = Sim(tmp_path, cfg, s)
    res = sim.run_all("rhythm")
    models = set(res["metrics"]["model_id"])
    assert {"R1", "R2", "R3", "A1", "A1tw", "A2", "B1", "B2", "C1"} <= models
    assert np.isfinite(res["scorecard"].to_numpy(float)).all()
    assert {"A1", "B2", "C1"} <= set(res["curves"]["family"])
    stages = [l for l in (tmp_path / "timings.jsonl").read_text().splitlines()]
    # every stage is cached: a second run trains nothing
    Sim(tmp_path, cfg, s).run_baselines(); Sim(tmp_path, cfg, s).run_trees(); Sim(tmp_path, cfg, s).run_hmm()
    assert len((tmp_path / "timings.jsonl").read_text().splitlines()) == len(stages)
