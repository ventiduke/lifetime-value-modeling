"""python -m ltv.playground [--bundle DIR] [--train-demo] [--export-web DIR] [--port 8765]"""
from __future__ import annotations

import argparse
from pathlib import Path

from ltv.playground import core


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ltv.playground", description=__doc__)
    ap.add_argument("--bundle", default=str(Path.cwd() / "artifacts" / "playground"),
                    help="folder with saved fits (default: ./artifacts/playground)")
    ap.add_argument("--train-demo", action="store_true",
                    help="first train small models on SYNTHETIC players into the bundle folder (several minutes)")
    ap.add_argument("--world", default="rhythm", choices=sorted(core.WORLDS), help="synthetic world for --train-demo")
    ap.add_argument("--users", type=int, default=60_000, help="synthetic players for --train-demo")
    ap.add_argument("--fit", default=None, help="which fit to load (default: final_1, else final_2, else tune)")
    ap.add_argument("--raw", action="store_true", help="show raw model outputs, without the calibration layer")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-serve", action="store_true", help="with --train-demo: train and exit")
    ap.add_argument("--export-web", metavar="DIR", default=None,
                    help="write the static site (models run in the browser) to DIR and exit; synthetic bundles only")
    a = ap.parse_args()
    if a.train_demo:
        core.train_demo(a.bundle, world=a.world, users=a.users, rows=min(400_000, a.users * 7))
        if a.no_serve:
            return
    if a.export_web:
        from ltv.playground.web import export_site
        out = export_site(core.load_bundle(a.bundle, fit=a.fit, calibrated=not a.raw), a.export_web)
        print(f"static site written to {out}: " + ", ".join(sorted(p.name for p in out.iterdir())))
        return
    from ltv.playground.server import serve
    serve(core.load_bundle(a.bundle, fit=a.fit, calibrated=not a.raw), a.port, not a.no_browser)


if __name__ == "__main__":
    main()
