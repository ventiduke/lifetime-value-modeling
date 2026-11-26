"""A local playground: design or simulate one player's weekly history and see what each model says.

Run it on your own machine (never deployed anywhere):

    python -m ltv.playground --train-demo     # once: trains small models on SYNTHETIC data (minutes)
    python -m ltv.playground                  # opens http://127.0.0.1:8765

or point it at fits from the real run (``--bundle <folder>``, see docs/PLAYGROUND.md).
``core`` holds the logic and is what the tests exercise; ``server`` is a thin HTTP layer.
"""
