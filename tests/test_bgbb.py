"""R3 (BG/BB): the closed forms are checked against simulation of the model itself."""
import numpy as np
import pandas as pd
import pytest

from ltv.models import bgbb


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(3)
    N, n, m = 150_000, 40, 52
    a, b, g, d = 1.2, 2.5, 0.7, 6.0
    p = rng.beta(a, b, N)
    th = rng.beta(g, d, N)
    # A user alive for periods 1..life-1 dies at the start of period `life`.
    life = rng.geometric(th)
    t = np.arange(1, n + m + 1)[None, :]
    buys = (rng.random((N, n + m)) < p[:, None]) & (t < life[:, None])
    past, fut = buys[:, :n], buys[:, n:]
    x = past.sum(1)
    tx = np.where(x > 0, n - np.argmax(past[:, ::-1], axis=1), 0)
    return dict(params=(a, b, g, d), x=x, tx=tx, n=np.full(N, n), fut=fut, m=m)


def test_expected_transactions_match_simulation(sim):
    e = bgbb.expected_transactions(sim["params"], sim["x"], sim["tx"], sim["n"], sim["m"])
    actual = sim["fut"].sum(1)
    assert e.mean() == pytest.approx(actual.mean(), rel=0.02)
    # and within groups of past frequency, not only on average
    df = pd.DataFrame({"x": np.minimum(sim["x"], 8), "e": e, "a": actual}).groupby("x").mean()
    np.testing.assert_allclose(df["e"], df["a"], rtol=0.08, atol=0.03)


def test_prob_no_transaction_matches_simulation(sim):
    p0 = bgbb.prob_no_transaction(sim["params"], sim["x"], sim["tx"], sim["n"], sim["m"])
    quiet = ~sim["fut"].any(1)
    assert p0.mean() == pytest.approx(quiet.mean(), abs=0.005)
    g = pd.DataFrame({"r": np.minimum(sim["n"] - sim["tx"], 20), "p": p0, "q": quiet}).groupby("r")
    df, size = g.mean(), g.size()
    # By recency group: tight on average (weighted by group size), loose for the small groups.
    assert np.average(np.abs(df["p"] - df["q"]), weights=size) < 0.01
    np.testing.assert_allclose(df["p"], df["q"], atol=0.06)
    assert ((p0 >= 0) & (p0 <= 1)).all()


def test_fit_recovers_the_forecasts(sim):
    """Parameters are weakly identified individually; the forecasts they imply are not."""
    est = bgbb.fit_params(sim["x"], sim["tx"], sim["n"])
    assert bgbb.loglik(est, sim["x"][:20000], sim["tx"][:20000], sim["n"][:20000]) >= \
        bgbb.loglik(sim["params"], sim["x"][:20000], sim["tx"][:20000], sim["n"][:20000]) - 20
    e_est = bgbb.expected_transactions(est, sim["x"], sim["tx"], sim["n"], 52)
    e_true = bgbb.expected_transactions(sim["params"], sim["x"], sim["tx"], sim["n"], 52)
    assert e_est.mean() == pytest.approx(e_true.mean(), rel=0.05)
    assert np.corrcoef(e_est, e_true)[0, 1] > 0.995
