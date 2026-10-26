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

import pytest

REPO = Path(__file__).resolve().parents[1]
from qpocket.viz import explorer_path

NOTEBOOK = explorer_path()
websockets = pytest.importorskip("websockets", reason="needs the websockets client")


def _site_run(tmp_path, monkeypatch, shape_clustered: bool) -> Path:
    """The site-steps fixture run: clouds only, or with site_01 shape-clustered.

    The shape-clustered run also carries a link cutoff beyond the regroup control's range.
    """
    from qpocket.arms import sites
    from qpocket.config import evolve, load_run_config, save_run_config
    from qpocket.io import layout
    from tests.test_site_steps import _finished_run

    root = _finished_run(tmp_path, monkeypatch)
    if shape_clustered:
        sites.shape_cluster_site(root, "site_01")
        save_run_config(evolve(load_run_config(root), "sites.link_cutoff", 25.0), layout.workspace(root))
    return root


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cell_errors(port: int, seconds: float = 90.0):
    """Every error the kernel reports while the app runs its cells, and every output it sends."""
    import asyncio

    async def listen():
        found, finished, outputs = [], set(), []
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
                        outputs.append(json.dumps(output, ensure_ascii=False))
                        if output.get("channel") == "marimo-error":
                            # `mo.stop` is how a marimo cell says "nothing to
                            # show"; it marks descendants, and that is expected.
                            found.extend(item for item in (output.get("data") or [])
                                         if item.get("type") != "ancestor-stopped")
                        if data.get("status") == "idle":
                            finished.add(data.get("cell_id"))
            except (asyncio.TimeoutError, TimeoutError):
                pass
        return found, finished, "\n".join(outputs)

    return asyncio.run(listen())


@pytest.mark.golden
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
@pytest.mark.parametrize("shape_clustered", [False, True], ids=["clouds-only", "shape-clustered"])
def test_a_site_run_leaves_no_error_boxes(tmp_path, monkeypatch, shape_clustered):
    """A three-structure site: the landscape's UMAP once crashed on runs this small, deep in scipy."""
    root = _site_run(tmp_path, monkeypatch, shape_clustered)
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "marimo", "run", str(NOTEBOOK), "--headless",
         "--no-token", "-p", str(port), "--", "--runs", str(root)],
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

        errors, finished, shown = _cell_errors(port)
    finally:
        server.terminate()
        server.wait(timeout=60)

    assert finished, "no cell ran at all"
    assert not errors, "cells left showing errors: " + "; ".join(
        str(e.get("msg") or e.get("type"))[:120] for e in errors[:5])
    for text in ("Binding sites", "Steps for site_01", "site_01 · "):
        assert text in shown, f"{text!r} never rendered"
    assert ("No distance matrix for this site yet" in shown) is not shape_clustered
    assert "Two-stage clustering (0/3 featurised)" in shown
