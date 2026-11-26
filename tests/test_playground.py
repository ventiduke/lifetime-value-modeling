"""The local playground (ltv.playground): a designed or simulated player scored by saved fits. No Spark."""
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from ltv.eval import examples as ex
from ltv.playground import core


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """A tiny demo bundle: synthetic players, a few trees, a few GRU steps. About 20 seconds."""
    out = core.train_demo(tmp_path_factory.mktemp("bundle"), users=3000, rows=12_000, rounds=20, seq_steps=40,
                          log=lambda *_: None)
    return core.load_bundle(out)


def test_demo_bundle_is_a_bundle(bundle):
    assert bundle.fit == "final_1" and list(bundle.scorers) == ["A1", "A2", "C1"] and set(bundle.recal) == set(bundle.scorers)
    assert bundle.meta["synthetic"] is True and bundle.origin == bundle.meta["origin_week"]
    assert len(bundle.season_week) >= bundle.origin + 1 + core.HORIZON, "the calendar reaches past the forecast window"
    assert not (bundle.path / "_world").exists(), "the simulated players are not kept"
    i = core.info(bundle)
    assert [m["id"] for m in i["models"]] == ["A1", "A2", "C1"] and i["synthetic"] and len(i["presets"]) >= 5
    assert i["calendar"]["future"] and all(0 < k <= 52 for k in i["calendar"]["future"])
    assert all(len(p["hours"]) <= bundle.max_weeks and max(p["hours"]) > 0 for p in i["presets"])
    json.dumps(i)                                                    # everything the page receives is plain JSON


def test_a_folder_without_models_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="train-demo"):
        core.load_bundle(tmp_path)


def test_clean_history_matches_how_the_data_holds_a_player():
    h, s = core.clean_history([0, 0, -3, 5, float("nan"), 400, 0], [0, 0, 0, 0, 0, 0, 2], 100)
    assert h.tolist() == [5, 0, 168, 0] and s.tolist() == [0, 0, 0, 2], "starts at the first week with a row; hours capped"
    h, s = core.clean_history([0, 0, 0, 1], [7], 100)                # a shorter spend list lines up with the latest weeks
    assert h.tolist() == [1] and s.tolist() == [7]
    h, s = core.clean_history([0, 2, 0], [3, 0, 0], 100)
    assert h.tolist() == [0, 2, 0] and s.tolist() == [3, 0, 0], "a spend-only week starts the history too"
    assert len(core.clean_history(np.ones(500), None, 120)[0]) == 120
    with pytest.raises(ValueError, match="empty"):
        core.clean_history([0, 0], [0, 0], 100)


def test_forecast_gives_totals_and_a_weekly_path_per_model(bundle):
    playing = [6.0] * 80
    gone = [6.0] * 80 + [0.0] * 60
    f, g = core.forecast(bundle, playing), core.forecast(bundle, gone)
    assert f["weeks_of_history"] == 80 and f["weeks_since_active"] == 0 and f["in_population"]
    assert g["weeks_since_active"] == 60 and g["in_population"]
    for m in bundle.scorers:
        t, path = f["models"][m]["totals"], f["models"][m]["path"]
        assert 0 <= t["p_reengage"] <= 1 and 0 <= t["p_spend"] <= 1 and t["exp_hours"] >= 0
        assert len(path["hours"]) == 52 and sum(path["hours"]) == pytest.approx(t["exp_hours"], rel=1e-6, abs=1e-9)
        assert all(0 <= a <= 1 for a in path["active"]) and f["models"][m]["calibrated"]
        assert t["p_reengage"] > g["models"][m]["totals"]["p_reengage"], "playing now beats gone for over a year"
    assert not f["models"]["A1"]["has_shape"] and f["models"]["A2"]["has_shape"] and f["models"]["C1"]["has_shape"]
    assert len(set(np.round(f["models"]["A1"]["path"]["hours"], 9))) == 1, "A1 has totals only: a flat path"
    assert not core.forecast(bundle, [5.0] + [0.0] * 150)["in_population"], "quiet for over 104 weeks"
    raw = core.load_bundle(bundle.path, calibrated=False)
    assert not raw.recal and not core.forecast(raw, playing)["models"]["C1"]["calibrated"]


def test_timeline_ends_now_and_never_reads_ahead(bundle):
    h = [5.0] * 40 + [0.0] * 30 + [4.0] * 20 + [0.0] * 10
    t = core.timeline(bundle, h, weeks=60, step=1)
    assert t["offsets"] == list(range(-60, 1))
    now = core.forecast(bundle, h)
    store, rows = core.make_store(bundle, h), np.array([0])
    for m, score in bundle.scorers.items():
        assert t["models"][m]["p_reengage"][-1] == pytest.approx(now["models"][m]["totals"]["p_reengage"], abs=1e-9)
        # the stacked scoring of the trees equals scoring one week at a time
        k = t["offsets"].index(-17)
        one = ex._calibrate(score(store, rows, bundle.origin - 17), bundle.recal.get(m))
        assert t["models"][m]["exp_hours"][k] == pytest.approx(float(one.exp_hours[0]), rel=1e-9, abs=1e-9)
    changed = h[:-8] + [30.0] * 8                                    # rewrite the last 8 weeks
    u = core.timeline(bundle, changed, weeks=60, step=1)
    for m in bundle.scorers:
        np.testing.assert_allclose(u["models"][m]["p_reengage"][:-8], t["models"][m]["p_reengage"][:-8], rtol=0, atol=1e-12)
        assert u["models"][m]["exp_hours"][-1] != t["models"][m]["exp_hours"][-1]
    short = core.timeline(bundle, [3.0] * 6, weeks=104, step=2)
    assert short["offsets"] == [-4, -2, 0], "a player has no opinion before their first week"


def test_simulated_players_are_reproducible_and_of_the_kind_asked_for(bundle):
    for world in core.WORLDS:
        a, b = core.simulate(bundle, world, "any", seed=5), core.simulate(bundle, world, "any", seed=5)
        assert a == b and len(a["future_hours"]) == 52 and 12 < len(a["hours"]) <= bundle.max_weeks
        assert a["hours"][0] > 0 or a["spend"][0] > 0
        core.forecast(bundle, a["hours"], a["spend"])                # every simulated player can be scored
        lapsed = core.simulate(bundle, world, "lapsed", seed=5)
        assert lapsed["matched"] and not any(x > 0 for x in lapsed["hours"][-13:])
        assert core.simulate(bundle, world, "playing", seed=5)["hours"][-1] > 0 or core.simulate(bundle, world, "playing", seed=5)["hours"][-2] > 0
        assert core.simulate(bundle, world, "returner", seed=5)["returns"] >= 1
    with pytest.raises(ValueError):
        core.simulate(bundle, "nowhere")


def test_server_serves_the_page_and_the_four_endpoints(bundle):
    from http.server import ThreadingHTTPServer

    from ltv.playground.server import make_handler
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(bundle))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    try:
        status, page = call("/")
        assert status == 200 and b"LTV playground" in page and b"/api/forecast" in page
        assert json.loads(call("/api/info")[1])["fit"] == "final_1"
        h = [4.0] * 30
        f = json.loads(call("/api/forecast", {"hours": h, "spend": [0.0] * 30})[1])
        assert set(f["models"]) == {"A1", "A2", "C1"}
        assert json.loads(call("/api/timeline", {"hours": h, "weeks": 10, "step": 5})[1])["offsets"] == [-10, -5, 0]
        assert len(json.loads(call("/api/simulate", {"world": "markov", "seed": 2})[1])["future_hours"]) == 52
        with pytest.raises(urllib.error.HTTPError) as bad:
            call("/api/forecast", {"hours": [0, 0, 0]})
        assert bad.value.code == 400 and "empty" in json.loads(bad.value.read())["error"]
        with pytest.raises(urllib.error.HTTPError) as missing:
            call("/api/nothing", {})
        assert missing.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
