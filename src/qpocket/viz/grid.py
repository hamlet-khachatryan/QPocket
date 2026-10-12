"""The multi-panel 3Dmol pocket grid: which pockets, what the spheres show, the page."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class PanelValues(NamedTuple):
    """What every panel's spheres encode, and on what scale.

    Attributes:
        values: pdb_id to (coordinates, values).
        label: What the values are.
        ramp: Colour ramp.
        domain: Value range of the ramp.
        stops: Gradient stops for the ramp.
    """

    values: Dict[str, Tuple[np.ndarray, np.ndarray]]
    label: str
    ramp: Any
    domain: Any
    stops: Any


def representative_rows(active_info: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """One row per cluster: its marked representative, else the member closest to its exemplar.

    Args:
        active_info: Run table with pdb_id, cluster_id, is_representative and maybe dist_to_exemplar.

    Returns:
        (rows, clusters that had no marked representative).
    """
    _picked, _fallbacks = [], []
    for _cluster, _members in active_info.groupby("cluster_id", sort=True):
        _reps = _members[_members["is_representative"].fillna(False)]
        if len(_reps):
            _picked.append(_reps.iloc[0])
            continue
        _fallbacks.append(int(_cluster))
        if "dist_to_exemplar" in _members.columns and _members["dist_to_exemplar"].notna().any():
            _picked.append(_members.nsmallest(1, "dist_to_exemplar").iloc[0])
        else:
            _picked.append(_members.iloc[0])
    _rows = pd.DataFrame(_picked)
    return _rows, _fallbacks


def panel_values(pockets: Sequence, mode: str, feat: str, rep_row: Optional[pd.Series],
                 grid_spacing: Optional[float], color_scheme: str, clip_pct: float) -> PanelValues:
    """Per-panel sphere values for one colour mode.

    Args:
        pockets: (pdb_id, metadata row, point table) per panel.
        mode: "feature value", "consensus occupancy" or "difference from representative".
        feat: Feature column.
        rep_row: Reference row for the difference mode.
        grid_spacing: Shared lattice spacing, or None.
        color_scheme: Colour scheme name.
        clip_pct: Percentile clip for the colour domain.

    Returns:
        The values, their label, ramp, domain and gradient stops.
    """
    from qpocket import viz as qviz

    panel_values = {}
    if mode == "consensus occupancy":
        _occ = qviz.occupancy([p[["x", "y", "z"]].to_numpy(dtype=float) for _, _, p in pockets], grid_spacing)
        for pid, _, pocket in pockets:
            _own = qviz.voxel_index(pocket[["x", "y", "z"]].to_numpy(dtype=float), grid_spacing)
            _own_keys = {tuple(_r) for _r in _own}
            _keep = np.array(
                [tuple(_r) in _own_keys for _r in qviz.voxel_index(_occ.coords, grid_spacing)],
                dtype=bool,
            )
            panel_values[pid] = (_occ.coords[_keep], _occ.fraction[_keep])
        value_label = f"in how many of {_occ.n_pockets} pockets"
        ramp = qviz.RampSpec("sequential", "Viridis", ("this pocket only", "every pocket"))
        domain = qviz.ColourDomain(0.0, 1.0, None)
    elif mode == "difference from representative":
        _ref_id = str(rep_row["pdb_id"])
        _ref = next(p for pid, _, p in pockets if pid == _ref_id)
        _ref_xyz = _ref[["x", "y", "z"]].to_numpy(dtype=float)
        _deltas = []
        for pid, _, pocket in pockets:
            _diff = qviz.field_difference(
                _ref_xyz, _ref[feat].to_numpy(dtype=float),
                pocket[["x", "y", "z"]].to_numpy(dtype=float), pocket[feat].to_numpy(dtype=float),
                grid_spacing,
            )
            panel_values[pid] = (_diff.coords, _diff.delta)
            _deltas.append(_diff.delta)
        value_label = f"{feat} minus {_ref_id}"
        ramp = qviz.RampSpec("diverging", qviz.feature_ramp(feat, color_scheme).scale,
                             (f"lower than {_ref_id}", f"higher than {_ref_id}"))
        domain = qviz.colour_domain(np.concatenate(_deltas) if _deltas else [0.0],
                                    float(clip_pct), "diverging")
    else:
        for pid, _, pocket in pockets:
            panel_values[pid] = (
                pocket[["x", "y", "z"]].to_numpy(dtype=float),
                pocket[feat].to_numpy(dtype=float) if feat in pocket.columns else np.zeros(len(pocket)),
            )
        value_label = feat
        ramp = qviz.feature_ramp(feat, color_scheme)
        domain = qviz.colour_domain(
            np.concatenate([v for _, v in panel_values.values()]), float(clip_pct), ramp.kind
        )
    stops = qviz.ramp_stops(ramp.scale, domain.zero_frac)
    return PanelValues(panel_values, value_label, ramp, domain, stops)


def build_panels(pockets: Sequence, values: Dict, domain, *, across_clusters: bool, sizes: Dict,
                 show_pocket: bool, picks: Sequence[str], run_interactions, protein_style: str,
                 show_all_sidechains: bool, context_radius: float, contact_r: float,
                 show_ligand: bool) -> List[dict]:
    """The molecular models and caption of every panel.

    Args:
        pockets: (pdb_id, metadata row, point table) per panel.
        values: pdb_id to (coordinates, values).
        domain: Colour domain of the values.
        across_clusters: Whether each panel stands for a cluster.
        sizes: Cluster sizes.
        show_pocket: Draw the point cloud.
        picks: Residues to highlight.
        run_interactions: The run's interaction tables, or None.
        protein_style: "none", "cartoon only" or with side chains.
        show_all_sidechains: Draw every interacting residue in every panel.
        context_radius: Protein context around the cloud, Å.
        contact_r: Side chains drawn this close to the cloud, Å.
        show_ligand: Draw the ligand.

    Returns:
        One dict per panel, as the page's script reads it.
    """
    from qpocket import viz as qviz

    panels = []
    for pid, row, pocket in pockets:
        _xyz, _vals = values[pid]

        _cloud = ""
        if show_pocket and len(_xyz):
            _cloud = qviz.pseudo_atom_pdb(_xyz, b=qviz.scale_to_percent(_vals, domain))

        _models = qviz.ProteinModels("", "", 0, 0, 0)
        _highlights = {}
        _aligned = row.get("aligned_file")
        _has_structure = isinstance(_aligned, str) and Path(_aligned).exists()

        if _has_structure and picks:
            # Picked residues are drawn whatever the context radius is, each class
            # in its own model so it can carry its own colour scheme.
            _groups = (qviz.interactions.classify_residues(run_interactions, picks)
                       if run_interactions is not None
                       else {"observed": tuple(picks)})
            _highlights = qviz.residue_models(_aligned, _groups)

        if protein_style != "none" and _has_structure:
            # Every residue that interacts anywhere in the run, in every panel: an
            # absence then means the conformation moved it, not that a radius
            # filtered it out.
            _always = (sorted(run_interactions.engaged)
                       if run_interactions is not None and show_all_sidechains else ())
            _models = qviz.pocket_context_models(
                _aligned,
                pocket[["x", "y", "z"]].to_numpy(dtype=float),
                float(context_radius),
                contact_r,
                always=_always,
            )
            if protein_style == "cartoon only":
                _models = _models._replace(sidechains="")

        _lig = ""
        _lp = row.get("ligand_file")
        if show_ligand and isinstance(_lp, str) and _lp and Path(_lp).exists():
            _lig = qviz.ligand_pdb(_lp)

        _title = []
        if across_clusters:
            _title.append(f"cluster {int(row['cluster_id'])} (n={sizes.get(row['cluster_id'], 0)})")
        _title.append(pid)
        if bool(row.get("is_representative")) and not across_clusters:
            _title.append("rep")
        if len(_vals):
            _title.append(f"mean {float(np.mean(_vals)):.2f}")
        else:
            _title.append("no shared sites")

        panels.append(
            {
                "id": pid,
                "title": " · ".join(_title),
                "cloud": _cloud,
                "cartoon": _models.cartoon,
                "side": _models.sidechains,
                "highlights": _highlights,
                "n_forced": int(_models.n_forced),
                "ligand": _lig,
                "vals": [round(float(v), 3) for v in _vals],
                "n_points": int(len(_xyz)),
                "n_ctx": int(_models.n_context),
            }
        )
    return panels


def grid_html(panels: List[dict], *, bbox_pdb: str, n_cols: int, panel_px: int, sphere_radius: float,
              stops, value_label: str, ramp, domain, hover: bool, cartoon_opacity: float,
              ligand_color: str, observed_colour: str, potential_colour: str,
              offline: bool) -> Tuple[str, int]:
    """The page: one canvas split into synchronised 3Dmol viewports, plus the colour bar.

    Args:
        panels: From build_panels.
        bbox_pdb: Invisible framing box shared by every panel.
        n_cols: Panels per row.
        panel_px: Panel height in pixels.
        sphere_radius: Point sphere radius.
        stops: Gradient stops.
        value_label: What the spheres show.
        ramp: Colour ramp.
        domain: Value range.
        hover: Show values on hover.
        cartoon_opacity: Cartoon opacity.
        ligand_color: Ligand colour scheme.
        observed_colour: Carbon colour of observed residues.
        potential_colour: Carbon colour of potential residues.
        offline: Inline the vendored 3Dmol build.

    Returns:
        (HTML, grid height in pixels).
    """
    from qpocket import viz as qviz

    ncol = max(1, min(int(n_cols), len(panels)))
    nrow = int(np.ceil(len(panels) / ncol))
    grid_h = int(panel_px) * nrow

    payload = {
        "rows": nrow,
        "cols": ncol,
        "panels": panels,
        "bbox": bbox_pdb,
        "radius": float(sphere_radius),
        "colorscheme": qviz.gradient_spec(stops),
        "feature": value_label,
        "hover": bool(hover),
        # One opacity for the whole cartoon model: 3Dmol splits cartoon geometry
        # wherever adjacent residues disagree on opacity, so keep it uniform.
        "cartoonOpacity": float(cartoon_opacity),
        "ligandScheme": ligand_color,
        "highlightSchemes": {
            "observed": qviz.interactions.carbon_scheme(observed_colour),
            "potential": qviz.interactions.carbon_scheme(potential_colour),
        },
    }

    # ---- One canvas, one WebGL context, one viewer per cell -----------------
    # createViewerGrid splits a single canvas into viewports, so the panel count
    # is not capped by the browser's WebGL context limit, and control_all makes a
    # drag in any panel drive every panel.
    _caps = "".join(
        f'<div class="cap" style="left:{(i % ncol) / ncol * 100:.4f}%;'
        f'top:{(i // ncol) / nrow * 100:.4f}%;width:{100 / ncol:.4f}%">{p["title"]}</div>'
        for i, p in enumerate(panels)
    )
    _legend_css = ",".join(qviz.legend_gradient(stops))
    # Mark where zero falls, but only when it is actually inside the domain.
    _zero_tick = ""
    if domain.zero_frac is not None and 0.02 < domain.zero_frac < 0.98:
        _zero_tick = (
            f'<div id="zero" style="left:{domain.zero_frac * 100:.2f}%"></div>'
            f'<div id="zerolab" style="left:{domain.zero_frac * 100:.2f}%">0</div>'
        )
    _html = f"""
    <style>
      body {{ margin:0; font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#0b0b0b; background:#fff; }}
      #wrap {{ position:relative; width:100%; height:{grid_h}px; }}
      #grid {{ width:100%; height:{grid_h}px; }}
      .cap {{ position:absolute; pointer-events:none; text-align:center; padding-top:4px;
             font-size:12px; color:#52514e; z-index:2; }}
      #bar {{ display:flex; align-items:center; gap:8px; padding:8px 4px 2px; color:#52514e; }}
      #rampbox {{ position:relative; flex:0 0 300px; }}
      #ramp {{ height:12px; border:1px solid #d8d7d2; border-radius:2px;
              background:linear-gradient(to right,{_legend_css}); }}
      #zero {{ position:absolute; top:-3px; width:1px; height:18px; background:#1a1a19; }}
      #zerolab {{ position:absolute; top:15px; transform:translateX(-50%); font-size:10px; color:#52514e; }}
      .end {{ font-size:11px; }}
      #err {{ display:none; padding:12px; color:#8a1f11; }}
    </style>
    <div id="wrap"><div id="grid"></div>{_caps}</div>
    <div id="bar">
      <span><b>{value_label}</b></span>
      <span class="end">{ramp.ends[0]}</span><span>{domain.lo:.2f}</span>
      <div id="rampbox"><div id="ramp"></div>{_zero_tick}</div>
      <span>{domain.hi:.2f}</span><span class="end">{ramp.ends[1]}</span>
      <span style="margin-left:10px">{ramp.scale} · drag any panel, all follow</span>
    </div>
    <div id="err">3Dmol.js did not load. Tick “offline viewer” to inline the vendored build.</div>
    {qviz.threedmol_script(offline=bool(offline))}
    <script>
    const D = {json.dumps(payload)};
    function build() {{
      if (typeof $3Dmol === "undefined") {{ document.getElementById("err").style.display = "block"; return; }}
      const viewers = $3Dmol.createViewerGrid(
        document.getElementById("grid"),
        {{ rows: D.rows, cols: D.cols, control_all: true }},
        {{ backgroundColor: "white", antialias: true }}
      );
      const all = [];
      D.panels.forEach((p, i) => {{
        const v = viewers[Math.floor(i / D.cols)][i % D.cols];
        all.push(v);
        // Invisible box first: every panel frames the identical volume.
        const box = v.addModel(D.bbox, "pdb", {{ assignBonds: false }});
        if (p.cartoon) {{
          const m = v.addModel(p.cartoon, "pdb");
          v.setStyle({{ model: m }}, {{ cartoon: {{ color: "#b9bdc4", thickness: 0.35, opacity: D.cartoonOpacity }} }});
        }}
        if (p.side) {{
          const m = v.addModel(p.side, "pdb");
          v.setStyle({{ model: m }}, {{ stick: {{ radius: 0.11 }} }});
        }}
        Object.entries(p.highlights || {{}}).forEach(([cls, pdb]) => {{
          const m = v.addModel(pdb, "pdb");
          const scheme = D.highlightSchemes[cls];
          v.setStyle({{ model: m }}, {{ stick: {{ radius: 0.22, colorscheme: scheme }} }});
          v.addStyle({{ model: m }}, {{ sphere: {{ radius: 0.28, colorscheme: scheme }} }});
        }});
        if (p.ligand) {{
          const m = v.addModel(p.ligand, "pdb");
          v.setStyle({{ model: m }}, {{
            stick: {{ radius: 0.17, colorscheme: D.ligandScheme }},
            sphere: {{ radius: 0.30, colorscheme: D.ligandScheme }},
          }});
        }}
        if (p.cloud) {{
          const m = v.addModel(p.cloud, "pdb", {{ assignBonds: false }});
          v.setStyle({{ model: m }}, {{ sphere: {{ radius: D.radius, colorscheme: D.colorscheme }} }});
          if (D.hover) v.setHoverable({{ model: m }}, true,
            (atom, viewer) => {{
              if (atom.label) return;
              const val = p.vals[atom.serial - 1];
              atom.label = viewer.addLabel(
                p.id + "  " + D.feature + " " + (val === undefined ? "?" : val),
                {{ position: atom, backgroundColor: "#1a1a19", backgroundOpacity: 0.85,
                   fontColor: "white", fontSize: 11 }}
              );
              viewer.render();
            }},
            (atom, viewer) => {{
              if (atom.label) {{ viewer.removeLabel(atom.label); delete atom.label; viewer.render(); }}
            }}
          );
        }}
        v.zoomTo({{ model: box }});
      }});
      all.forEach(v => v.render());
      window.addEventListener("resize", () => all.forEach(v => {{ v.resize(); v.render(); }}));
    }}
    if (document.readyState === "loading") {{
      document.addEventListener("DOMContentLoaded", build);
    }} else {{
      build();
    }}
    </script>
    """
    return _html, grid_h


def grid_notes(panels: List[dict], *, across_clusters: bool, fallbacks: Sequence[int],
               missing: Sequence[str], protein_style: str, context_radius: float, contact_r: float,
               run_interactions, mode: str, value_label: str, picks: Sequence[str]) -> List[str]:
    """The caption under the grid.

    Args:
        panels: From build_panels.
        across_clusters: Whether each panel stands for a cluster.
        fallbacks: Clusters shown by their member closest to the exemplar.
        missing: Structures without featurized.csv.
        protein_style: Protein drawing style.
        context_radius: Protein context radius, Å.
        contact_r: Side-chain radius, Å.
        run_interactions: The run's interaction tables, or None.
        mode: Colour mode.
        value_label: What the spheres show.
        picks: Highlighted residues.

    Returns:
        Caption fragments, in order.
    """
    from qpocket import viz as qviz

    _notes = []
    if across_clusters:
        _notes.append(f"one representative from each of {len(panels)} clusters")
    if fallbacks:
        _notes.append(
            f"no marked representative for cluster(s) {', '.join(str(c) for c in fallbacks)};"
            " used the member closest to the exemplar"
        )
    if missing:
        _notes.append(f"No `featurized.csv` for: {', '.join(missing)}")
    if protein_style != "none":
        _note = (f"protein: residues within {context_radius} Å of the cloud "
                 f"(median {int(np.median([p['n_ctx'] for p in panels]))} per panel); "
                 f"side chains within {contact_r:.0f} Å")
        _forced = [p["n_forced"] for p in panels]
        if any(_forced):
            _note += (f", plus every interacting residue "
                      f"({min(_forced)}–{max(_forced)} of "
                      f"{len(run_interactions.engaged)} found per panel)")
        _notes.append(_note)
    _notes.append(f"{sum(p['n_points'] for p in panels)} spheres across {len(panels)} panels")
    if mode != "feature value":
        _notes.append(f"spheres show {value_label}")
    if picks:
        _classes = (", ".join(
            f"{len(v)} {k}" for k, v in qviz.interactions.classify_residues(
                run_interactions, picks).items())
            if run_interactions is not None else str(len(picks)))
        _notes.append(f"highlighted: {_classes}")
    return _notes
