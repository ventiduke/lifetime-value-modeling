"""Candidate C: inputs, heads, the models and their resumable training (no Spark)."""
import copy

import numpy as np
import pytest
import torch

from ltv.data import seq_inputs as si
from ltv.data.np_features import F, tabular_features
from ltv.data.sequences import SequenceStore
from ltv.data.splits import make_schedule
from ltv.eval import metrics as M
from ltv.models import kernels_np as kn
from ltv.models.heads import MixtureHead, OrdinalHead, ZILNHead, quantile_edges
from ltv.models.seq import SeqConfig, SeqModel, label_tensors, predict_seq, to_tensors
from ltv.synth.generator import generate, to_store
from ltv.train.seq_fit import fit_seq, make_batch, prepare_config, validation_batches
from ltv.train.timebox import TimeBox

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def store():
    return to_store(generate(5000, 320, seed=9))


@pytest.fixture(scope="module")
def sched():
    return make_schedule(319)


# -- inputs -----------------------------------------------------------------------

def test_block_grid_is_fixed_so_states_can_be_carried(store):
    rows = np.arange(300)
    b1, l1 = si.block_inputs(store, rows, 200)
    b2, l2 = si.block_inputs(store, rows, 204)
    assert si.n_complete_blocks(204) == si.n_complete_blocks(200) + 1
    old = l1 > 0
    assert (l2[old] == l1[old] + 1).all(), "four weeks later, exactly one more block"
    for i in np.flatnonzero(old)[:50]:
        np.testing.assert_array_equal(b2[i, :l1[i]], b1[i, :l1[i]])       # earlier blocks never change
    _, wl = si.weekly_inputs(store, rows, 200)
    full = store.first_week[rows] <= si.n_complete_blocks(200) * si.BLOCK
    assert wl[full].min() >= si.RECENT and wl.max() <= si.MAX_WEEKLY


def test_inputs_cover_the_whole_history(store):
    """Blocks plus recent weeks account for every week from a user's first block to the origin."""
    origin = 250
    rows = store.rows_for(origin)[:200]
    _, bl = si.block_inputs(store, rows, origin)
    _, wl = si.weekly_inputs(store, rows, origin)
    first_block_start = (store.first_week[rows] // si.BLOCK) * si.BLOCK
    with_blocks = bl > 0
    assert ((bl * si.BLOCK + wl)[with_blocks] == (origin + 1 - first_block_start)[with_blocks]).all()
    assert (wl[~with_blocks] == (origin + 1 - store.first_week[rows])[~with_blocks]).all()


def test_episode_tokens_on_a_hand_built_history():
    W = 80
    h = np.zeros((1, W), np.float32)
    h[0, [0, 1, 2, 30, 31, 60]] = [5, 5, 5, 2, 2, 7]
    z = np.zeros((1, W), np.int16)
    st = SequenceStore(np.array(["a"]), np.array(["train"]), np.array([0], np.int32), h, np.zeros((1, W), np.float32),
                       z, z, np.tile(np.arange(1, 9), 20).astype(np.int16), (np.arange(160) // 8).astype(np.int16))
    tok, n = si.episode_tokens(st, np.array([0]), 70)
    assert n[0] == 3
    np.testing.assert_allclose(np.expm1(tok[0, :, 0]), [3, 2, 1], rtol=1e-5)          # spell lengths
    np.testing.assert_allclose(np.expm1(tok[0, :, 2]), [15, 4, 7], rtol=1e-5)         # hours per spell
    np.testing.assert_allclose(np.expm1(tok[0, :, 4]), [0, 27, 28], rtol=1e-5)        # lapse before each spell
    np.testing.assert_allclose(np.expm1(tok[0, :, 7]), [68, 39, 10], rtol=1e-5)       # weeks since the spell ended


def test_inputs_read_nothing_after_the_origin(store):
    origin = 220
    rows = store.rows_for(origin)[:300]
    st2 = copy.deepcopy(store)
    st2.hours[:, origin + 1:] = 9.0
    st2.spend[:, origin + 1:] = 9.0
    a = si.build_inputs(store, rows, origin, tokens=True)
    b = si.build_inputs(st2, rows, origin, tokens=True)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


# -- heads ------------------------------------------------------------------------

def test_mixture_and_ordinal_quantiles_against_sampling():
    rng = np.random.default_rng(1)
    w, mu, sg = np.array([0.6, 0.4]), np.array([1.0, 3.0]), np.array([0.5, 0.4])
    comp = rng.choice(2, 300_000, p=w)
    pos = rng.lognormal(mu[comp], sg[comp])
    draws = np.where(rng.random(300_000) < 0.5, pos, 0.0)
    prm = {"logw": np.log(w)[None], "mu": mu[None], "sigma": sg[None]}
    q = MixtureHead.quantiles(prm, np.array([0.5]), (0.6, 0.8, 0.95))[0]
    np.testing.assert_allclose(q, np.quantile(draws, [0.6, 0.8, 0.95]), rtol=0.04)
    edges = quantile_edges(pos, 8)
    head = OrdinalHead(4, edges=edges)
    uniform = {"logp": np.log(np.full((1, 8), 1 / 8)), "pos": np.full((1, 8), 0.5)}
    qo = head.quantiles(uniform, np.array([1.0]), (0.25, 0.5, 0.75))[0]
    np.testing.assert_allclose(qo, np.quantile(pos, [0.25, 0.5, 0.75]), rtol=0.05)


def test_head_means_match_their_distributions():
    z = torch.zeros(1, 4)
    zh = ZILNHead(4, center=2.0)
    with torch.no_grad():
        zh.out.weight.zero_()
        zh.out.bias[1] = 0.5413                      # softplus -> sigma = 1.0
    p = zh(z)
    assert float(zh.mean(p).detach()) == pytest.approx(np.exp(2.0 + 0.5), rel=1e-3)
    y = torch.tensor([np.exp(2.0)])
    assert float(zh.nll(p, y).detach()) == pytest.approx(0.0, abs=1e-3)       # at the mode of log y, with sigma = 1


# -- model ------------------------------------------------------------------------

@pytest.mark.parametrize("encoder,head", [("gru2res", "ziln"), ("episodes", "mixture"), ("transformer", "ordinal")])
def test_forward_and_losses_for_every_encoder_and_head(store, sched, encoder, head):
    cfg = prepare_config(SeqConfig(encoder=encoder, head=head, d=16, whale_aux=True), store, sched.tune_train)
    model = SeqModel(cfg)
    rows = store.rows_for(sched.tune_train[-1], "train")[:64]
    inp, y = make_batch(store, rows, sched.tune_train[-1], cfg, "current_season")
    out = model(inp)
    assert out["weekly"].shape == (64, 52)
    L = model.losses(out, y)
    assert set(L) >= {"weekly", "any_active", "any_spend", "hours", "spend", "share_hours", "share_spend", "whale"}
    assert all(torch.isfinite(v) for v in L.values())
    model.total_loss(L).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_block_state_can_be_carried_forward(store, sched):
    """The history encoder's state after n blocks equals its state after n - 1 blocks plus one step:
    the property that lets weekly scoring keep a per-user state instead of rereading history."""
    cfg = prepare_config(SeqConfig(d=16), store, sched.tune_train)
    model = SeqModel(cfg).eval()
    rows = store.rows_for(200)
    rows = rows[store.first_week[rows] < 100][:40]
    b, l = si.block_inputs(store, rows, 204)
    blocks, length = torch.from_numpy(b), torch.from_numpy(l)
    with torch.no_grad():
        full = model.encode_blocks(blocks, length)
        prev = model.encode_blocks(blocks, length - 1)
        last = blocks[torch.arange(len(rows)), length - 1]
        x = model.hist_proj(last)
        _, stepped = model.hist_gru(x.unsqueeze(1), prev)
        np.testing.assert_allclose(stepped.numpy(), full.numpy(), rtol=1e-5, atol=1e-6)
        # and the NumPy kernel takes the same step without PyTorch
        h_np = kn.gru_step(prev[0].numpy().astype(np.float64), x.numpy().astype(np.float64),
                           kn.gru_weights(model.hist_gru))
        np.testing.assert_allclose(h_np, full[0].numpy(), rtol=1e-4, atol=1e-5)


@pytest.fixture(scope="module")
def trained(store, sched):
    cfg = prepare_config(SeqConfig(d=32), store, sched.tune_train)
    val = validation_batches(store, sched.validation, cfg, "current_season")
    model, info = fit_seq(cfg, [store], sched.tune_train, TimeBox(None), val=val, max_steps=240, eval_every=60,
                          patience=10, seed=1, batch=256, threads=2)
    return model, info


def test_training_reduces_the_validation_loss(trained):
    _, info = trained
    hist = [h["total"] for h in info["history"]]
    assert len(hist) == 4 and hist[-1] < hist[0]
    assert info["complete"] and info["best_step"] > 0


def test_predictions_are_coherent_and_sensible(trained, store, sched):
    model, _ = trained
    rows = store.rows_for(sched.validation, "val")
    y = store.labels(rows, sched.validation)
    p = predict_seq(model, store, rows, sched.validation)
    assert M.auc(y["any_active_52"], p.p_reengage) > 0.8
    assert (np.diff(p.cum_hours, axis=1) >= -1e-6).all() and (np.diff(p.cum_spend, axis=1) >= -1e-6).all()
    np.testing.assert_allclose(p.cum_hours[:, -1], p.exp_hours, rtol=1e-4)
    np.testing.assert_allclose(p.weekly_p_active.sum(1), p.exp_active_weeks, rtol=1e-4)
    assert (p.q_hours[:, 0] <= p.q_hours[:, 2] + 1e-6).all()
    assert p.state.shape == (len(rows), 32)


def test_the_model_uses_long_history_to_spot_returners(trained, store, sched):
    model, _ = trained
    o = sched.validation
    rows = store.rows_for(o)
    p = predict_seq(model, store, rows, o).p_reengage
    X = tabular_features(store, rows, o)
    lapsed = X[:, F["wk_since_active"]] >= 13
    many, none = lapsed & (X[:, F["n_returns_26"]] >= 1), lapsed & (X[:, F["n_returns_26"]] == 0)
    assert many.sum() > 50 and none.sum() > 50
    assert p[many].mean() > p[none].mean() + 0.15


def test_training_resumes_on_the_same_path(store, sched, tmp_path):
    cfg = prepare_config(SeqConfig(d=16), store, sched.tune_train)
    val = validation_batches(store, sched.validation, cfg, "current_season", max_users=300)
    kw = dict(val=val, eval_every=10, patience=99, seed=3, batch=64, threads=1)
    straight, _ = fit_seq(cfg, [store], sched.tune_train, TimeBox(None), max_steps=30, **kw)
    ck = tmp_path / "seq.pt"
    _, i1 = fit_seq(cfg, [store], sched.tune_train, TimeBox(None), ck, max_steps=20, **kw)
    resumed, i2 = fit_seq(cfg, [store], sched.tune_train, TimeBox(None), ck, max_steps=30, **kw)
    assert i1["steps"] == 20 and i2["steps"] == 30
    assert [h["step"] for h in i2["history"]] == [10, 20, 30]
    a = dict(straight.state_dict())
    for k, v in resumed.state_dict().items():
        np.testing.assert_allclose(v.numpy(), a[k].numpy(), rtol=2e-4, atol=2e-5, err_msg=k)


def test_final_fit_runs_exactly_the_frozen_number_of_steps(store, sched):
    cfg = prepare_config(SeqConfig(d=16), store, sched.tune_train)
    _, info = fit_seq(cfg, [store], sched.final_train_1, TimeBox(None), fixed_steps=17, seed=3, batch=64)
    assert info["steps"] == 17 and info["complete"] and info["history"] == []


def test_learning_rate_drops_are_recorded_and_replayed(store, sched):
    """With ``lr_drops`` the tune fit cuts the rate when patience runs out instead of stopping,
    and a final fit cuts it at the same steps."""
    cfg = prepare_config(SeqConfig(d=16), store, sched.tune_train)
    val = validation_batches(store, sched.validation, cfg, "current_season", max_users=300)
    kw = dict(seed=3, batch=64, threads=1, lr=2e-3)
    _, plain = fit_seq(cfg, [store], sched.tune_train, TimeBox(None), val=val, max_steps=400, eval_every=5,
                       patience=1, **kw)
    _, tuned = fit_seq(cfg, [store], sched.tune_train, TimeBox(None), val=val, max_steps=400, eval_every=5,
                       patience=1, lr_drops=2, **kw)
    assert plain["lr_steps"] == [] and plain["lr"] == pytest.approx(2e-3)
    assert len(tuned["lr_steps"]) == 2 and tuned["steps"] > plain["steps"]
    assert tuned["lr"] == pytest.approx(2e-3 * 0.3 ** 2)
    last = tuned["lr_steps"][-1]
    _, final = fit_seq(cfg, [store], sched.final_train_1, TimeBox(None), fixed_steps=last,
                       lr_steps=tuned["lr_steps"] + [last + 50], **kw)
    assert final["steps"] == last and final["lr_steps"] == tuned["lr_steps"]
    assert final["lr"] == pytest.approx(2e-3 * 0.3 ** 2)


def test_an_expired_time_box_returns_incomplete(store, sched, tmp_path):
    cfg = prepare_config(SeqConfig(d=16), store, sched.tune_train)
    _, info = fit_seq(cfg, [store], sched.tune_train, TimeBox(0.0), tmp_path / "x.pt", max_steps=50, seed=3)
    assert info["steps"] == 0 and not info["complete"]
