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
    """The site-steps fixture run: clouds only, or with the near site shape-clustered.

    The shape-clustered run also carries a link cutoff beyond the regroup control's range.
    """
    from qpocket.arms import sites
    from qpocket.config import evolve, load_run_config, save_run_config
    from qpocket.io import layout
    from tests.test_site_steps import _finished_run

    root = _finished_run(tmp_path, monkeypatch)
    if shape_clustered:
        sites.shape_cluster_site(root, "S1-Y1")
        save_run_config(evolve(load_run_config(root), "sites.box_padding", 25.0), layout.workspace(root))
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
    errors, finished, shown = _render(root)
    assert finished, "no cell ran at all"
    assert not errors, "cells left showing errors: " + "; ".join(
        str(e.get("msg") or e.get("type"))[:120] for e in errors[:5])
    for text in ("Binding sites", "Steps for S1-Y1", "S1-Y1 · ", "Ligands shown",
                 "Exploration", "Screening", "Merge…", "Split…", "Overrides", "Site overview", "qp-on-Site"):
        assert text in shown, f"{text!r} never rendered"
    assert ("No distance matrix for this site yet" in shown) is not shape_clustered
    if shape_clustered:  # the clustering views need the site's distance matrices
        for text in ("percentile", "Pockets shown", "shared volume", "Finalise site", "Set representative",
                     "Residue contacts by cluster", "Finalise this site on the Exploration page.", "Not finalised yet"):
            assert text in shown, f"{text!r} never rendered"
    assert "Two-stage clustering (0/3 featurised)" in shown
    assert "silhouette" not in shown.lower()
    assert "exclusiv" not in shown.lower()
    assert "Can two states be targeted" not in shown  # the section heading, which never says the word


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_a_run_with_ligand_linked_sites_asks_for_a_regroup(tmp_path, monkeypatch):
    """A run made before cavity-defined sites has the old site table; the notebook must not break on it."""
    import pandas as pd

    root = _site_run(tmp_path, monkeypatch, False)
    old = pd.read_csv(root / "sites" / "sites.csv")
    pd.DataFrame({"site_id": old["site_id"], "n_ligands": old["n_ligands"], "n_holo_structures": old["n_holo"],
                  "centre_x": old["centre_x"], "centre_y": old["centre_y"], "centre_z": old["centre_z"],
                  "extent": 3.0, "resnames": old["resnames"], "contact_residues": ""}).to_csv(
        root / "sites" / "sites.csv", index=False)
    errors, finished, shown = _render(root)
    assert finished and not errors, "; ".join(str(e.get("msg") or e.get("type"))[:120] for e in errors[:5])
    assert "made before cavity-defined sites" in shown


def _render(root):
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

        return _cell_errors(port)
    finally:
        server.terminate()
        server.wait(timeout=60)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_a_two_stage_site_renders_the_chemistry_views(tmp_path, monkeypatch):
    """Featurised and two-stage clustered: the mirrored chemistry tree, chemistry rows and cluster map."""
    from qpocket.arms import sites
    from tests.test_site_steps import _finished_run

    from qpocket.pipelines import structure

    def featurize(csv_path, pdb_path, features, charges, options=None, charge_params=None):
        import numpy as np
        import pandas as pd

        table = pd.read_csv(csv_path)
        rng = np.random.default_rng(len(table))
        table.assign(apbs=rng.normal(size=len(table)), hbond=rng.normal(size=len(table))).to_csv(
            csv_path.with_name("featurized.csv"), index=False)

    monkeypatch.setattr(structure, "featurize", featurize)
    root = _finished_run(tmp_path, monkeypatch)
    sites.featurize_site(root, "S1-Y1", workers=1)
    sites.two_stage_site(root, "S1-Y1")
    errors, finished, shown = _render(root)
    assert finished, "no cell ran at all"
    assert not errors, "cells left showing errors: " + "; ".join(
        str(e.get("msg") or e.get("type"))[:160] for e in errors[:5])
    for text in ("Chemistry cut (percentile)", "Cluster distances", "Pockets shown", "Residue contacts by cluster"):
        assert text in shown, f"{text!r} never rendered"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_tabs_guard_without_a_site(tmp_path, monkeypatch):
    """A run with no sites: the Site page still renders; Exploration and Screening say what to do first."""
    import pandas as pd

    root = _site_run(tmp_path, monkeypatch, False)
    table = pd.read_csv(root / "sites" / "sites.csv")
    table.iloc[:0].to_csv(root / "sites" / "sites.csv", index=False)
    errors, finished, shown = _render(root)
    assert finished and not errors, "; ".join(str(e.get("msg") or e.get("type"))[:120] for e in errors[:5])
    assert "Pick a site on the Site page." in shown and "qp-on-Exploration qp-on-Screening" in shown
    assert "No sites to draw." in shown


def _element_id(html: str, kind: str, label: str):
    """The object id of the last UI element of this kind (button, multiselect, …) whose label holds ``label``."""
    import html as _html
    import re

    found = None
    for _ in range(4):  # callouts and stacks carry their children as escaped HTML in an attribute
        for match in re.finditer(r"object-id=['\"]([^'\"]+)['\"]", html):
            element = html[match.end():match.end() + 1500].split("marimo-ui-element", 1)[0]
            if f"<marimo-{kind}" in element and label in _html.unescape(element):
                found = match.group(1)
        html = _html.unescape(html).replace("\\u003c", "<").replace("\\u003e", ">").replace('\\"', '"')
    return found


def _session(root, actions, settle: float = 25.0):
    """Serve the explorer, let it run, then perform ``actions`` (callables given the outputs so far).

    Each action returns ``(object_ids, values)`` to send; after each, outputs are collected for ``settle`` s.
    Returns the outputs after each step, as lists of (cell_id, output html) pairs.
    """
    import asyncio
    import urllib.request

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "marimo", "run", str(NOTEBOOK), "--headless",
         "--no-token", "-p", str(port), "--", "--runs", str(root)],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    session_id = str(uuid.uuid4())

    async def drive():
        import re

        page = await asyncio.to_thread(lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode())
        token = re.search(r"""(?:serverToken|server_token)["']?\s*:\s*["']([^"']+)""", page)
        assert token, "no server token in the page"
        steps = []
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws?session_id={session_id}", max_size=None) as ws:
            async def collect(seconds):
                seen = []
                try:
                    async with asyncio.timeout(seconds):
                        async for raw in ws:
                            message = json.loads(raw)
                            if message.get("op") != "cell-op":
                                continue
                            output = (message.get("data") or {}).get("output") or {}
                            data = output.get("data")
                            if isinstance(data, str):
                                seen.append((message["data"].get("cell_id"), data))
                            elif output.get("channel") == "marimo-error":
                                seen.extend(("ERROR", json.dumps(item)) for item in (data or [])
                                            if item.get("type") != "ancestor-stopped")
                except (asyncio.TimeoutError, TimeoutError):
                    pass
                return seen

            steps.append(await collect(90.0))
            for action in actions:
                ids, values = action([html for step in steps for _, html in step])
                body = json.dumps({"objectIds": ids, "values": values}).encode()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/kernel/set_ui_element_value", data=body, method="POST",
                    headers={"Content-Type": "application/json", "Marimo-Session-Id": session_id,
                             "Marimo-Server-Token": token.group(1)})
                await asyncio.to_thread(urllib.request.urlopen, request)
                steps.append(await collect(settle))
        return steps

    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(1.0)
        else:
            pytest.fail("marimo did not start")
        return asyncio.run(drive())
    finally:
        server.terminate()
        server.wait(timeout=60)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_a_site_edit_waits_for_confirm_and_keeps_its_outcome(tmp_path, monkeypatch):
    """Drop… only previews (the site table does not re-read); Confirm applies, and its message stays."""
    import pandas as pd

    root = _site_run(tmp_path, monkeypatch, False)
    far = "S2-Y1"

    def pick(outputs):
        html = "\n".join(outputs)
        sites = _element_id(html, "multiselect", "Sites")
        assert sites, "no site multiselect"
        return [sites], [[far]]

    def drop(outputs):
        button = _element_id("\n".join(outputs), "button", "Drop…")
        assert button, "no Drop… button"
        return [button], [1]

    def confirm(outputs):
        button = _element_id("\n".join(outputs), "button", "Confirm")
        assert button, "no Confirm button after Drop…"
        return [button], [1]

    first, _, dropped, confirmed = _session(root, [pick, drop, confirm])
    assert any(f"Pending: drop {far}" in html for _, html in dropped), "Drop… showed no pending change"
    assert not any("Binding sites" in html for _, html in dropped), "previewing re-read the site table"
    assert pd.read_csv(root / "sites" / "sites.csv")["site_id"].tolist() == ["S1-Y1"]
    last = {}
    for cell, html in first + dropped + confirmed:
        last[cell] = html
    assert any(f"Done: drop {far}" in html for html in last.values()), "the outcome was wiped or never shown"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_finalise_from_the_page_writes_final_and_clears_the_guard(tmp_path, monkeypatch):
    from qpocket.arms.sites import final

    root = _site_run(tmp_path, monkeypatch, True)

    def finalise(outputs):
        button = _element_id("\n".join(outputs), "button", "Finalise site")
        assert button, "no Finalise site button"
        return [button], [1]

    first, clicked = _session(root, [finalise])
    assert final.final_status(root, "S1-Y1") == "current"
    shown = "\n".join(html for _, html in clicked)
    assert "Finalised into" in shown and "Screening S1-Y1 from its final metadata" in shown


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_a_finalised_site_renders_the_screening_page_and_saves_its_box(tmp_path, monkeypatch):
    """Screening from a finalised site: box, docking counts, run options, analysis; Save box writes box.json."""
    import json as _json

    from qpocket.arms.sites import final

    root = _site_run(tmp_path, monkeypatch, True)
    final.finalise_site(root, "S1-Y1")

    def save(outputs):
        button = _element_id("\n".join(outputs), "button", "Save box")
        assert button, "no Save box button"
        return [button], [1]

    first, saved = _session(root, [save])
    errors = [html for cell, html in first + saved if cell == "ERROR"]
    assert not errors, "cells left showing errors: " + "; ".join(e[:200] for e in errors[:5])
    shown = "\n".join(html for _, html in first)
    for text in ("Active/decoy benchmark — not implemented yet", "Save box", "dockings", "Run here",
                 "SLURM (writes scripts only)", "VM over SSH", "Analyse", "Poses"):
        assert text in shown, f"{text!r} never rendered"
    box = _json.loads((root / "sites" / "S1-Y1" / "screening" / "box.json").read_text())
    assert box["mode"] in ("interacting_residues", "representative_pockets") and box["site_id"] == "S1-Y1"
