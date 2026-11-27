"""The repository itself: what a fresh clone holds."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHES = ("__pycache__", ".pytest_cache", ".egg-info", ".pyc")


def _git(*args: str) -> list[str]:
    run = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    if run.returncode != 0:
        pytest.skip("not a git checkout")
    return run.stdout.splitlines()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_fresh_clone_holds_everything_the_code_needs():
    """sql/build was once ignored by a stray .gitignore rule and never committed: every clone was
    missing the SQL that builds the modeling tables, and `pip install` failed. Nothing under the
    code, SQL, config, notebook, test or doc folders may be ignored, and the files on disk are tracked."""
    folders = ["sql", "src", "conf", "notebooks", "tests", "docs", "jobs", "genie"]
    ignored = [line[3:] for line in _git("status", "--ignored", "--porcelain", "--", *folders)
               if line.startswith("!!") and not any(c in line for c in CACHES)]
    assert not ignored, f"ignored by .gitignore but needed: {ignored}"
    tracked = set(_git("ls-files", "--", "sql", "src"))
    on_disk = {p.relative_to(ROOT).as_posix() for d in ("sql", "src") for p in (ROOT / d).rglob("*")
               if p.is_file() and not any(c in p.as_posix() for c in CACHES)}
    assert on_disk <= tracked, f"on disk but not committed: {sorted(on_disk - tracked)}"
    assert len([f for f in tracked if f.startswith("sql/build/") and f.endswith(".sql")]) == 10
