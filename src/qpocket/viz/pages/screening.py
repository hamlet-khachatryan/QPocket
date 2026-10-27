"""The Screening page: the box and pose scenes (one 3Dmol viewer each), the ligand picker, the success matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem

from qpocket.arms.screen.campaign import engine_for
from qpocket.core.crossdock import pose_rmsd
from qpocket.viz.assets import threedmol_script
from qpocket.viz.pdb_text import ligand_pdb, pseudo_atom_pdb, residue_model
from qpocket.viz.style import GREY, HUES, apply

#: Pose colours: within the success threshold, within twice it, beyond.
GOOD, CLOSE, FAR = "#2a9d8f", "#e9a03b", "#c4566a"
CARTOON = {"cartoon": {"color": "#c9ced4", "opacity": 0.75}}
TIGHT, PADDED = "#222222", "#1f5f8b"

_SCRIPT = """
<script>
(function () {
  const payload = JSON.parse(document.getElementById("qp-scene").textContent);
  const el = document.getElementById("qp-viewer");
  if (typeof $3Dmol === "undefined") { el.textContent = "3Dmol.js did not load."; return; }
  const viewer = $3Dmol.createViewer(el, {backgroundColor: "white"});
  for (const m of payload.models) { const model = viewer.addModel(m.pdb, "pdb"); model.setStyle({}, m.style); }
  for (const b of payload.boxes) {
    const c = [0, 1, 2].map(i => (b.min[i] + b.max[i]) / 2), d = [0, 1, 2].map(i => b.max[i] - b.min[i]);
    viewer.addBox({center: {x: c[0], y: c[1], z: c[2]}, dimensions: {w: d[0], h: d[1], d: d[2]},
                   color: b.color, wireframe: true, opacity: b.dashed ? 0.55 : 1.0});
  }
  for (const s of payload.spheres) {
    const p = {x: s.center[0], y: s.center[1], z: s.center[2]};
    viewer.addSphere({center: p, radius: s.radius, color: s.color});
    if (s.label) viewer.addLabel(s.label, {position: p, fontSize: 11, fontColor: "#222",
                                           backgroundColor: "white", backgroundOpacity: 0.6});
  }
  viewer.zoomTo(); viewer.render();
})();
</script>
"""


def scene_html(models: List[dict], boxes: List[dict], spheres: List[dict], height: int = 520,
               offline: bool = True, message: str = "") -> str:
    """One 3Dmol viewer of PDB models, wireframe boxes and labelled spheres.

    Args:
        models: ``{"pdb": text, "style": 3Dmol style}``, optionally with a ``role``.
        boxes: ``{"min": [x, y, z], "max": [x, y, z], "color", "dashed"}``.
        spheres: ``{"center": [x, y, z], "radius", "color", "label"}``.
        height: Viewer height in pixels.
        offline: Inline the vendored 3Dmol build.
        message: Text shown above the viewer.

    Returns:
        A full HTML page, for ``mo.iframe``.
    """
    payload = json.dumps({"models": models, "boxes": boxes, "spheres": spheres}).replace("</", "<\\/")
    note = f'<div style="font:12px Helvetica,Arial;color:#555;margin:4px 0">{message}</div>' if message else ""
    return (f'<!doctype html><html><body style="margin:0">{note}'
            f'<div id="qp-viewer" style="width:100%;height:{int(height)}px;position:relative"></div>'
            f'<script type="application/json" id="qp-scene">{payload}</script>'
            f'{threedmol_script(offline=offline)}{_SCRIPT}</body></html>')


def _box(box: Dict[str, float], color: str, dashed: bool) -> dict:
    return {"min": [float(box[f"min_{a}"]) for a in "xyz"], "max": [float(box[f"max_{a}"]) for a in "xyz"],
            "color": color, "dashed": dashed}


def _frame_models(preview, residues: Sequence[str]) -> List[dict]:
    models = [{"pdb": Path(preview.frame_pdb).read_text(encoding="utf-8"), "style": CARTOON, "role": "protein"}]
    highlighted = residue_model(str(preview.frame_pdb), list(residues)) if residues else ""
    if highlighted:
        models.append({"pdb": highlighted, "style": {"stick": {"colorscheme": "orangeCarbon", "radius": 0.18}},
                       "role": "residues"})
    return models


def default_ligands(ligands: pd.DataFrame, clusters: pd.DataFrame, limit: int = 40) -> List[str]:
    """The ligands the box viewer opens with: representatives' first, then one per cluster, then the most frequent.

    Args:
        ligands: final/ligands.csv (cluster, pdb_id, ligand_key, smiles).
        clusters: final/clusters.csv (pdb_id, cluster, is_representative).
        limit: At most this many.

    Returns:
        ligand_keys, in that order, without repeats.
    """
    reps = clusters.loc[clusters["is_representative"].astype(str).str.lower() == "true", "pdb_id"].astype(str)
    picked: List[str] = []

    def take(keys):
        for key in keys:
            if key not in picked and len(picked) < limit:
                picked.append(key)

    by_pdb = ligands.assign(pdb_id=ligands["pdb_id"].astype(str))
    take(k for rep in reps for k in by_pdb.loc[by_pdb["pdb_id"] == rep, "ligand_key"])
    covered = set(by_pdb.loc[by_pdb["ligand_key"].isin(picked), "cluster"])
    take(group["ligand_key"].iloc[0] for cluster, group in by_pdb.groupby("cluster", sort=False) if cluster not in covered)
    counts = by_pdb["smiles"].map(by_pdb["smiles"].value_counts())
    take(by_pdb.assign(_n=counts).sort_values(["_n", "ligand_key"], ascending=[False, True])["ligand_key"])
    return picked


def ligand_box_scene(preview, ligand_files: Sequence[str], offline: bool = True) -> str:
    """The interacting-residues box: cartoon, the contacted residues as sticks, the picked ligands, both boxes."""
    models = _frame_models(preview, preview.residues)
    for path in ligand_files:
        text = ligand_pdb(path) if Path(path).exists() else ""
        if text:
            models.append({"pdb": text, "style": {"stick": {"colorscheme": "greenCarbon", "radius": 0.15}},
                           "role": "ligand"})
    return scene_html(models, [_box(preview.tight, TIGHT, False), _box(preview.padded, PADDED, True)], [],
                      offline=offline, message=f"{len(preview.residues)} contacted residues · "
                                               f"{len(ligand_files)} ligands · padding {preview.padding:g} Å")


def cloud_box_scene(preview, offline: bool = True) -> str:
    """The cloud box: cartoon, representatives' clouds coloured by cluster, extremum points, both boxes."""
    models = _frame_models(preview, [])
    clusters = list(dict.fromkeys(preview.cluster_of.get(r, "") for r in preview.clouds))
    for rank, cluster in enumerate(clusters):
        pts = np.vstack([c for r, c in preview.clouds.items() if preview.cluster_of.get(r, "") == cluster])
        colour = HUES[rank % len(HUES)] if cluster else GREY
        models.append({"pdb": pseudo_atom_pdb(pts), "style": {"sphere": {"radius": 0.22, "color": colour}},
                       "role": f"cloud {cluster}"})
    spheres = [{"center": [float(v) for v in p], "radius": 0.7, "color": "#111111", "label": label}
               for p, label in zip(preview.extremes, preview.extreme_labels)]
    return scene_html(models, [_box(preview.tight, TIGHT, False), _box(preview.padded, PADDED, True)], spheres,
                      offline=offline, message=f"{len(preview.clouds)} representatives' clouds · "
                                               f"padding {preview.padding:g} Å")


def success_matrix(pairs: pd.DataFrame) -> pd.DataFrame:
    """Best-pose RMSD (Å) per receptor × ligand.

    Args:
        pairs: crossdock_pairs.csv (representative, active_id, best_topk_rmsd).

    Returns:
        Representatives as rows, actives as columns; NaN where nothing was docked or scored.
    """
    return pairs.pivot_table(index="representative", columns="active_id", values="best_topk_rmsd", aggfunc="min")


def matrix_figure(matrix: pd.DataFrame, threshold: float) -> go.Figure:
    """The success matrix: green below the threshold, red from twice it."""
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(dtype=float), x=list(matrix.columns), y=list(matrix.index),
                               zmin=0, zmax=2 * float(threshold), colorscale="RdYlGn_r",
                               colorbar=dict(title=dict(text="best RMSD (Å)", font=dict(size=9)), thickness=8)))
    fig = apply(fig)
    fig.update_layout(height=max(260, 14 * len(matrix) + 140), xaxis=dict(title="ligand", tickangle=-60),
                      yaxis=dict(title="receptor", autorange="reversed"),
                      title=dict(text=f"Cross-docking success · threshold {threshold:g} Å", font=dict(size=11)))
    return fig


def _colour(rmsd: float, threshold: float) -> str:
    if rmsd is None or not np.isfinite(rmsd):
        return GREY
    return GOOD if rmsd <= threshold else CLOSE if rmsd <= 2 * threshold else FAR


def pose_scene(folder: Path, job: pd.Series, top_n: int, threshold: float, residues: Sequence[str],
               offline: bool = True) -> Tuple[str, pd.DataFrame]:
    """One docking: receptor cartoon, crystal pose, the top-N docked poses coloured by RMSD, residues highlighted.

    Args:
        folder: The cross-docking task folder.
        job: Its docking_jobs.csv row.
        top_n: Poses shown, best-scored first.
        threshold: Success threshold (Å) for the colours.
        residues: Residues to highlight on the receptor.
        offline: Inline the vendored 3Dmol build.

    Returns:
        The scene and a table of rank, score, rmsd for the poses shown.
    """
    folder = Path(folder)
    out = Path(job["output_pdbqt"])
    out = out if out.is_absolute() else folder / out
    empty = pd.DataFrame(columns=["rank", "score", "rmsd"])
    if not out.exists():
        return scene_html([], [], [], height=120, offline=offline,
                          message=f"No poses yet for {job['active_id']} in {job['representative']}."), empty
    ligands = pd.read_csv(folder / "ligands.csv", keep_default_na=False).set_index("active_id")
    row = ligands.loc[job["active_id"]] if job["active_id"] in ligands.index else None
    crystal_files = [f for f in str(row["crystal_files"]).split(";") if f] if row is not None else []
    crystals = [m for m in (Chem.MolFromPDBFile(f, sanitize=False, removeHs=True) for f in crystal_files)
                if m is not None]
    smiles = row["canonical"] if row is not None else None
    receptor = folder / "receptors" / f"{job['representative']}.pdb"
    models = []
    if receptor.exists():
        models.append({"pdb": receptor.read_text(encoding="utf-8"), "style": CARTOON, "role": "protein"})
        highlighted = residue_model(str(receptor), list(residues)) if residues else ""
        if highlighted:
            models.append({"pdb": highlighted, "style": {"stick": {"colorscheme": "orangeCarbon", "radius": 0.18}},
                           "role": "residues"})
    if crystal_files and Path(crystal_files[0]).exists():
        models.append({"pdb": ligand_pdb(crystal_files[0]), "style": {"stick": {"color": "#6a4c93", "radius": 0.2}},
                       "role": "crystal"})
    rows = []
    for pose in sorted(engine_for(job, folder).read_poses(out), key=lambda p: p.rank)[:int(top_n)]:
        values = [v for v in (pose_rmsd(pose.mol, c, smiles) for c in crystals) if v is not None] if pose.mol else []
        rmsd = min(values) if values else float("nan")
        rows.append({"rank": pose.rank, "score": pose.score, "rmsd": rmsd})
        if pose.mol is not None:
            models.append({"pdb": Chem.MolToPDBBlock(Chem.RemoveHs(pose.mol, sanitize=False)),
                           "style": {"stick": {"color": _colour(rmsd, threshold), "radius": 0.15}}, "role": "pose"})
    message = (f"{job['active_id']} in {job['representative']} · crystal pose purple · poses green ≤ {threshold:g} Å,"
               f" amber ≤ {2 * threshold:g} Å, red beyond")
    return scene_html(models, [], [], offline=offline, message=message), pd.DataFrame(rows, columns=empty.columns)
