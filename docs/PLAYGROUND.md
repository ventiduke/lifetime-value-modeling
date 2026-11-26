# The playground: one player, every model

A small web page for getting a feel for the models. You draw or simulate one player's
weekly history, and each model's view of the next 52 weeks updates as you edit. It is a
way to look at models, not to evaluate them: that is `50_eval`.

It comes in two forms, with the same page:

| | Where the models run | Which models | Use it for |
| --- | --- | --- | --- |
| **Local** (`python -m ltv.playground`) | Python on your machine, the pipeline's own code | Any bundle, including the real fits | The real models; anything private |
| **Static site** (`docs/playground/`, served by GitHub Pages) | JavaScript in the browser, phone included | The synthetic demo only | A quick look from anywhere |

![The playground with the synthetic demo models: a player who lapsed for six months, came back, and lapsed again](playground.png)

## Run it

Once, from a checkout of this repo (Python 3.10 or newer; no Spark or Java needed):

```bash
pip install -e .
```

Train small demo models on **synthetic** players (about 10 minutes on 4 cores, once):

```bash
python -m ltv.playground --train-demo --no-serve
```

Start it:

```bash
python -m ltv.playground
```

It opens `http://127.0.0.1:8765/` in your browser. Stop it with Ctrl+C. The models live in
`./artifacts/playground`, which Git ignores.

| Option | What it does |
| --- | --- |
| `--bundle DIR` | Use the fits in another folder |
| `--train-demo` | Train the synthetic demo into the bundle folder first (`--world markov` for the other synthetic world, `--users N` for more or fewer players) |
| `--raw` | Show raw model outputs, without the calibration layer |
| `--fit final_2` | Load another fit (default: `final_1`, else `final_2`, else `tune`) |
| `--port N`, `--no-browser` | |

## What you can do

- **Start from a preset**: steady player; lapsed 6 months, came back, lapsed again; serial
  returner; hours fading; new player; gone a year; back this week.
- **Simulate a player** from one of the two synthetic worlds, optionally of a kind (playing
  now, lapsed, has come back from a long lapse). The 52 weeks that followed in the
  simulation are drawn as dashed bars. They are one random draw, not what a model should
  have said.
- **Draw**: drag across the past on the hours or spend panel. *Draw* follows the pointer,
  *Flat* repeats the height where you started, *Erase* clears weeks. Undo with the button
  or Ctrl+Z.
- **Let time pass**: add 1, 4, 13 or 26 weeks with no play, and watch each model's
  P(re-engage) fall.

What it shows:

| Where | What |
| --- | --- |
| Hours and spend panels, right of *now* | Each model's forecast for each of the next 52 weeks |
| Bottom panel, left of *now* | Each model's P(re-engage within 52 weeks) as it stood at the end of each past week: a fresh score of the history up to that week |
| Table | The five 52-week outputs per model, as of now |
| Small triangles | Season launches: real ones in the past, projected ones after now |

A1 predicts 52-week totals only, so its weekly line is flat. A2 is the tree variant with a
shape over time. A1 and A2 share one P(re-engage) model, so those lines coincide.

## The static site (GitHub Pages)

`docs/playground/` is a complete static site: the page, `engine.js` (the models in
JavaScript), `model.json` (the synthetic demo models, about 5 MB, 2.7 MB compressed),
`players.json` (a library of 200 simulated players for the Simulate button) and
`mode.json`. With GitHub Pages serving the `main` branch it is at:

- `https://<user>.github.io/<repo>/docs/playground/` if Pages serves the repository root;
- `https://<user>.github.io/<repo>/playground/` if Pages serves the `/docs` folder.

It works on a phone. The chart comes first on a narrow screen; the controls are under it.
A touch screen starts in **Look** (the page scrolls; tap a week to read it). Choose
**Draw**, **Flat** or **Erase** to edit by dragging. Nothing is sent anywhere: everything
runs in the page.

To rebuild it after changing the page, the engine or the demo models:

```bash
python -m ltv.playground --export-web docs/playground
```

**Only synthetic models can be exported.** A Pages site is public. The export refuses any
bundle whose `examples/meta.json` does not say `"synthetic": true`, and a test checks that
the published `model.json` says so.

`engine.js` is a JavaScript copy of the features (`np_features.py`), the calendar
projection, the tree walk, the GRU's forward pass and the calibration layer. That makes it
a third copy of the features, after the SQL and the NumPy. It is kept honest by
`tests/test_playground.py`, which scores the same players in Python and in JavaScript and
fails when they differ (features exactly; model outputs within 0.02%). The test needs
Node.js and is skipped without it. The static site supports the default models only: A1,
A2 and C1.

## Using the real models

After the Databricks run, the artifact folder is a bundle as it stands. Copy these files
from `<artifact_path>` on the volume to a local folder, keeping the layout:

```
trees/suite_final_1.pkl            the tree models
seq/C1/model_final_1.pkl           the GRU (and any other variant you ran)
examples/calibration.pkl           written by 55_examples: the shared calibration layer
examples/calendar.csv              written by 55_examples: the season calendar
examples/meta.json                 written by 55_examples: which fit, and which week is "now"
```

```bash
python -m ltv.playground --bundle /path/to/that/folder
```

**Keep real models on your machine.** They are trained on player data. Do not commit them
(`artifacts/` is ignored by Git) and do not publish them. The static site is for the
synthetic demo only, and its export refuses anything else.

Only open bundles you made yourself: the files are Python pickles.

## How it works

- "Now" is the bundle's origin week: the week the fit was built to be scored at. Your
  history is placed so that it ends there, which keeps the calendar and the tenure the
  models see inside what they were trained on. The longest history is that many weeks.
- A history starts at its first week with hours or spend, as in the data.
- The trees are scored on features rebuilt from the history by `np_features.py`, the
  implementation the tests keep equal to the SQL. The GRU reads the history directly.
- Scoring a past week reads only the weeks up to it (tested).
- Platform and country are fixed (one platform, one country). The page does not vary them.
- Players quiet for more than 104 weeks are outside the population the models were trained
  on. The page flags them.

Code: `src/ltv/playground/`. `core.py` is the logic and is what `tests/test_playground.py`
exercises. `server.py` is a thin standard-library HTTP layer bound to 127.0.0.1.
`index.html` is the page, with no external resources; it asks `mode.json` whether to call
the server or to run `engine.js` itself. `web.py` exports the static site.

## Limits

- **The demo models are trained on synthetic players.** They show how each kind of model
  responds to a history. They say nothing about real players and are not a comparison
  (`AGENTS.md`, rule 3). The demo fits are also small: 60,000 players, default settings.
- **The simulator is always synthetic**, also with real models loaded. On the static site
  Simulate draws from a fixed library of 200 simulated players instead of simulating anew.
- The page was checked in desktop Chromium and in Chromium's iPhone and Pixel emulation,
  not on a physical phone or in Safari.
- One player at a time, no saving, no authentication. It is a personal tool.
