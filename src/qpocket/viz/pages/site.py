"""The Site page: a 3D overview of the run's sites, the override table, and what a change would delete."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from biopandas.pdb import PandasPdb

from qpocket.config import load_run_config
from qpocket.io import layout
from qpocket.viz.style import GREY, HUES, apply

#: Index pairs of a box's eight corners (bit 0 = x, 1 = y, 2 = z) joined by an edge.
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def _box_line(row) -> Dict[str, List[Optional[float]]]:
    lo = (row.box_lo_x, row.box_lo_y, row.box_lo_z)
    hi = (row.box_hi_x, row.box_hi_y, row.box_hi_z)
    corners = [tuple(hi[d] if (c >> d) & 1 else lo[d] for d in range(3)) for c in range(8)]
    xyz: Dict[str, List[Optional[float]]] = {"x": [], "y": [], "z": []}
    for a, b in _EDGES:  # None breaks the line between edges
        for axis, d in zip("xyz", range(3)):
            xyz[axis] += [corners[a][d], corners[b][d], None]
    return xyz


def _reference(run_dir: Path) -> tuple:
    """(pdb_id, cleaned file, whether it is the run's reference) for the structure the overview draws."""
    from qpocket.pipelines.sites import _reference_id

    ws = layout.workspace(run_dir)
    master = pd.read_csv(ws.metadata / "master.csv", dtype=str, keep_default_na=False)
    records = [SimpleNamespace(pdb_id=p) for p in master["pdb_id"]]
    ref = _reference_id(load_run_config(run_dir), ws, records)
    row = master[master["pdb_id"] == ref] if ref else master.iloc[:0]
    found = not row.empty
    row = row if found else master.iloc[:1]
    return (row["pdb_id"].iloc[0], row["cleaned_file"].iloc[0], found) if len(row) else (None, "", False)


def _ca_traces(cleaned_file: str) -> List[go.Scatter3d]:
    if not cleaned_file or not Path(cleaned_file).exists():
        return []
    atoms = PandasPdb().read_pdb(cleaned_file).df["ATOM"]
    ca = atoms[atoms["atom_name"] == "CA"]
    return [go.Scatter3d(x=c["x_coord"], y=c["y_coord"], z=c["z_coord"], mode="lines", name="protein",
                         legendgroup="protein", showlegend=i == 0, hoverinfo="skip",
                         line=dict(color="#c9ced4", width=2))
            for i, (_, c) in enumerate(ca.groupby("chain_id", sort=True))]


def overview_figure(run_dir: Path, sites: pd.DataFrame, selected: Optional[str] = None,
                    max_ligands: int = 150) -> go.Figure:
    """Every site box on one structure's backbone, with a sample of the run's ligands.

    Args:
        run_dir: The run directory.
        sites: The site table (``sites/sites.csv``), in rank order.
        selected: Site drawn with a thick outline.
        max_ligands: Ligands drawn at most, evenly sampled with a fixed seed.

    Returns:
        A plotly 3D figure: protein Cα trace, one wireframe per site, ligand heavy atoms coloured by site.
    """
    from qpocket.pipelines.sites import _heavy_coords

    run_dir = Path(run_dir)
    ref_id, cleaned, is_ref = _reference(run_dir)
    fig = go.Figure(_ca_traces(cleaned))
    colour = {}
    for rank, row in enumerate(sites.itertuples()):
        colour[row.site_id] = HUES[rank % len(HUES)]
        thick = row.site_id == selected
        fig.add_trace(go.Scatter3d(**_box_line(row), mode="lines", name=row.site_id,
                                   line=dict(color=colour[row.site_id], width=4 if thick else 1.5),
                                   hovertext=f"{row.site_id} · {row.n_structures} structures", hoverinfo="text"))

    ligs_csv = layout.workspace(run_dir).structures / "ligands.csv"
    ligs = pd.read_csv(ligs_csv, dtype=str, keep_default_na=False) if ligs_csv.is_file() else pd.DataFrame()
    n_all = len(ligs)
    if n_all > max_ligands:
        ligs = ligs.iloc[np.sort(np.random.default_rng(0).choice(n_all, size=max_ligands, replace=False))]
    points, colours, hover = [], [], []
    for row in ligs.itertuples():
        try:
            xyz = _heavy_coords(row.file)
        except Exception:  # a moved or unreadable ligand file costs that ligand its dots, not the figure
            continue
        site = next(iter(filter(None, row.site_ids.split(";"))), "")
        points.append(xyz)
        colours += [colour.get(site, GREY)] * len(xyz)
        hover += [f"{row.pdb_id} {row.resname} {row.resseq} · {site or 'no site'}"] * len(xyz)
    if points:
        xyz = np.vstack(points)
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name="ligands",
                                   marker=dict(size=2, color=colours), hovertext=hover, hoverinfo="text"))
    frame = f"reference {ref_id}" if is_ref else f"{ref_id} (no reference found; first structure)"
    fig = apply(fig)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=36, b=0),
                      title=dict(text=f"{len(sites)} sites on {frame} · {len(ligs)} of {n_all} ligands",
                                 font=dict(size=11)),
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                                 aspectmode="data"))
    return fig


#: The picked site's sphere; every other site's.
CURRENT, OTHER = "#d62828", GREY


def site_scene(sites: pd.DataFrame, current: Optional[str], frame_pdb: Optional[Path],
               ligand_files: List[str], offline: bool = True) -> str:
    """Every site's centre as a sphere over the frame protein: the picked one red, the rest grey, names on hover.

    Args:
        sites: sites.csv (site_id, centre_x/y/z, n_structures, n_holo, n_ligands).
        current: The picked site.
        frame_pdb: Aligned structure drawn as a cartoon; none when None.
        ligand_files: The picked site's ligands to draw as sticks.
        offline: Inline the vendored 3Dmol build.

    Returns:
        A full HTML page, for ``mo.iframe``.
    """
    from qpocket.viz.pages.screening import CARTOON, scene_html
    from qpocket.viz.pdb_text import ligand_pdb

    models = []
    if frame_pdb is not None and Path(frame_pdb).exists():
        models.append({"pdb": Path(frame_pdb).read_text(encoding="utf-8"), "style": CARTOON, "role": "protein"})
    for path in ligand_files:
        text = ligand_pdb(path) if Path(path).exists() else ""
        if text:
            models.append({"pdb": text, "style": {"stick": {"colorscheme": "greenCarbon", "radius": 0.15}},
                           "role": "ligand"})
    spheres = [{"center": [float(r.centre_x), float(r.centre_y), float(r.centre_z)],
                "radius": 1.6 if r.site_id == current else 1.2,
                "color": CURRENT if r.site_id == current else OTHER,
                "hover": f"{r.site_id} · {int(r.n_structures)} structures · {int(r.n_holo)} holo · "
                         f"{int(r.n_ligands)} ligands"} for r in sites.itertuples()]
    return scene_html(models, [], spheres, height=440, offline=offline,
                      message=f"{current} red, other sites grey · hover a sphere for its site")


def override_table(run_dir: Path) -> pd.DataFrame:
    """The run's manual site edits, in the order they apply.

    Args:
        run_dir: The run directory.

    Returns:
        Columns ``#``, ``operation``, ``sites`` (names when the edit was made), ``status``.
    """
    from qpocket.arms.sites import list_overrides

    rows = [{"#": i, "operation": o["op"], "sites": o.get("label", ""), "status": o["status"]}
            for i, o in enumerate(list_overrides(run_dir))]
    return pd.DataFrame(rows, columns=["#", "operation", "sites", "status"])


def loss_markdown(losses: Dict[str, List[str]]) -> str:
    """What a change would delete, one bullet per site that holds results.

    Args:
        losses: Site id to the results it would lose (from ``preview_changes``).

    Returns:
        Markdown; "Nothing is lost." when no site holds anything.
    """
    lines = [f"- **{sid}**: {', '.join(items)}" for sid, items in losses.items() if items]
    return "\n".join(lines) if lines else "Nothing is lost."
