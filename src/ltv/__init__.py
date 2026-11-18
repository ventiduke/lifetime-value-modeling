"""LTV model bake-off: trees vs HMM vs sequence models on weekly activity data.

Layout:
    ltv.config      load and validate conf/*.yml
    ltv.data        SQL runner, split schedule, future calendar, sequence store and loaders
    ltv.synth       synthetic data generator (test fixture only)
    ltv.models      baselines, BG/BB, trees, HMM, sequence models, output heads
    ltv.train       time-boxed, resumable fit and tuning loops
    ltv.eval        metrics, segments, coherence, calibration, bootstrap, revisions
    ltv.score       scoring paths (Arrow batch functions, per-user state updates)
    ltv.pipeline    the phase-level entry points the notebooks call

See docs/DESIGN.md for the how and why, and docs/DATA_CONTRACT.md for every table.
"""

__version__ = "0.1.0"
