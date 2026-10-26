"""The explorer notebook opens a run without leaving error boxes on its cells.

`marimo export html` does not reproduce this: a cell stopped with `mo.stop`
leaves its descendants marked only in a live session, which is what `marimo run`
serves and what a user sees. So this drives a real app session over the socket.
"""
import json
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
from qpocket.io import layout
from qpocket.viz import explorer_path

NOTEBOOK = explorer_path()
websockets = pytest.importorskip("websockets", reason="needs the websockets client")


def _old_style_run(root: Path) -> Path:
    """A run as they were written before the shape and feature matrices existed."""
    run = root / "legacy" / "1ABC"
    meta = run / "metadata"
    meta.mkdir(parents=True)
    (meta / "run.json").write_text(json.dumps({"layout_version": layout.LAYOUT_VERSION}))
    ids = ["1ABC", "2DEF", "3GHI"]
    frame = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]], index=ids, columns=ids)
    # Only the spectral matrix, which is all those runs carried.
    frame.to_csv(meta / "spectral_distance_matrix.csv")
    pd.DataFrame({"pdb_id": ids, "cluster_id": [0, 0, 1],
                  "is_representative": [True, False, True]}).to_csv(
        meta / "clustering_metadata.csv", index=False)
    pd.DataFrame({"pdb_id": ids, "protein_state": ["BOUND_STATE"] * 3,
                  "rmsd_to_ref": [0.0, 0.3, 0.5]}).to_csv(meta / "master.csv", index=False)
    # A directory counts as a run only when it also holds pockets/.
    for pid in ids:
        pocket = run / "pockets" / pid
        pocket.mkdir(parents=True)
        pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0],
                      "apbs": [0.1, 0.2]}).to_csv(pocket / "featurized.csv", index=False)
        pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}).to_csv(
            pocket / "binding_site.csv", index=False)
    return run


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cell_errors(port: int, seconds: float = 90.0):
    """Every error the kernel reports while the app runs its cells."""
    import asyncio

    async def listen():
        found, finished = [], set()
        url = f"ws://127.0.0.1:{port}/ws?session_id={uuid.uuid4()}"
        async with websockets.connect(url, max_size=None) as ws:
            try:
                async with asyncio.timeout(seconds):
                    async for raw in ws:
                        message = json.loads(raw)
                        data = message.get("data", {})
                        if message.get("op") != "cell-op":
                            continue
                        output = data.get("output") or {}
                        if output.get("channel") == "marimo-error":
                            # `mo.stop` is how a marimo cell says "nothing to
                            # show"; it marks descendants, and that is expected.
                            found.extend(item for item in (output.get("data") or [])
                                         if item.get("type") != "ancestor-stopped")
                        if data.get("status") == "idle":
                            finished.add(data.get("cell_id"))
            except (asyncio.TimeoutError, TimeoutError):
                pass
        return found, finished

    return asyncio.run(listen())


@pytest.mark.golden
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_a_run_without_the_newer_matrices_leaves_no_error_boxes(tmp_path):
    """A small run used to crash inside UMAP's spectral start, deep in scipy."""
    _old_style_run(tmp_path)
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "marimo", "run", str(NOTEBOOK), "--headless",
         "--no-token", "-p", str(port), "--", "--runs", str(tmp_path)],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(1.0)
        else:
            pytest.fail("marimo did not start")

        errors, finished = _cell_errors(port)
    finally:
        server.terminate()
        server.wait(timeout=60)

    assert finished, "no cell ran at all"
    assert not errors, "cells left showing errors: " + "; ".join(
        str(e.get("msg") or e.get("type"))[:120] for e in errors[:5])
