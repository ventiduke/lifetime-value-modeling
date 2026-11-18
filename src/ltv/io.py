"""File writes that are safe on Unity Catalog volumes.

Volumes do not support random-access or append writes. Formats that seek while writing
(zip-based ones: ``np.savez``, ``torch.save``) or files opened in append mode can fail when
written there directly. The safe pattern is to produce the file on local disk and copy it
over in one sequential write. Every checkpoint, shard and log in this package goes through
``write_via_local``. Reading from a volume needs no special care.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable


def write_via_local(path: str | Path, write_fn: Callable[[str], None]) -> None:
    """Call ``write_fn(local_path)`` to produce a file, then copy it to ``path``.

    The copy goes to ``<path>.tmp`` first and is renamed, so a killed command does not leave
    a half-written file under the real name. Where rename is not supported the file is
    copied directly instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=path.suffix or ".bin")
    os.close(fd)
    try:
        write_fn(tmp)
        staged = str(path) + ".tmp"
        shutil.copyfile(tmp, staged)
        try:
            os.replace(staged, path)
        except OSError:
            shutil.copyfile(tmp, path)
            try:
                os.remove(staged)
            except OSError:
                pass
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def write_bytes(path: str | Path, data: bytes) -> None:
    def w(p: str) -> None:
        with open(p, "wb") as fh:
            fh.write(data)
    write_via_local(path, w)


def write_text(path: str | Path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))
