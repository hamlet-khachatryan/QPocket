"""The Exploration page: which residues each cluster contacts and lines, and picking a cluster's representative."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

from qpocket.viz.style import apply

PROFILE_COLUMNS = ["cluster", "residue", "contacted_pct", "lining_pct", "n_members", "n_holo"]


def residue_profile(run, members: Dict[str, List[str]]) -> pd.DataFrame:
    """Per cluster, how often each residue is contacted by a ligand and how often it lines the pocket.

    Args:
        run: The site's RunInteractions, or None when it has no interaction archives.
        members: Cluster name to its pdb_ids, in the order clusters are shown.

    Returns:
        :data:`PROFILE_COLUMNS`; contacted_pct over the cluster's holo members, lining_pct over all of them;
        residues at 0 for both are left out.
    """
    if run is None:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    rows = []
    for cluster, ids in members.items():
        ids = [str(i) for i in ids]
        holo = [i for i in ids if run.observed.get(i)]
        contacted = {i: {e.split("|", 1)[0] for e in run.observed.get(i, ())} for i in holo}
        lining = {i: set(run.potential.get(i, ())) for i in ids}
        for residue in run.residues:
            c = sum(residue in contacted[i] for i in holo)
            lined = sum(residue in lining[i] for i in ids)
            if c or lined:
                rows.append({"cluster": cluster, "residue": residue,
                             "contacted_pct": 100.0 * c / len(holo) if holo else 0.0,
                             "lining_pct": 100.0 * lined / len(ids) if ids else 0.0,
                             "n_members": len(ids), "n_holo": len(holo)})
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


def residue_heatmap(profile: pd.DataFrame, top: int = 40) -> go.Figure:
    """Clusters × residues, coloured by how often the cluster's ligands contact each residue.

    Args:
        profile: From :func:`residue_profile`.
        top: Residues shown, the most contacted on average.

    Returns:
        A plotly heatmap on 0–100 %, hover giving contacted and lining shares.
    """
    if profile.empty or not (profile["contacted_pct"] > 0).any():
        fig = apply(go.Figure())
        fig.update_layout(title=dict(text="No observed contacts at this site", font=dict(size=11)), height=160)
        return fig
    clusters = list(dict.fromkeys(profile["cluster"]))
    ranked = profile.groupby("residue")["contacted_pct"].sum().div(len(clusters)).sort_values(ascending=False)
    residues = list(ranked.index[:top])
    contacted = profile.pivot(index="cluster", columns="residue", values="contacted_pct").reindex(
        index=clusters, columns=residues).fillna(0.0)
    lining = profile.pivot(index="cluster", columns="residue", values="lining_pct").reindex(
        index=clusters, columns=residues).fillna(0.0)
    hover = [[f"{c} · {r}<br>contacted {contacted.loc[c, r]:.0f}% · lining {lining.loc[c, r]:.0f}%"
              for r in residues] for c in clusters]
    fig = go.Figure(go.Heatmap(z=contacted.to_numpy(), x=residues, y=clusters, zmin=0, zmax=100,
                               colorscale="Blues", hovertext=hover, hoverinfo="text",
                               colorbar=dict(title=dict(text="contacted (%)", font=dict(size=9)), thickness=8)))
    fig = apply(fig)
    fig.update_layout(height=max(220, 22 * len(clusters) + 120), xaxis=dict(tickangle=-60, title="residue"),
                      yaxis=dict(autorange="reversed", title="cluster"),
                      title=dict(text=f"Residue contacts by cluster · top {len(residues)} residues",
                                 font=dict(size=11)))
    return fig


def representative_options(info: pd.DataFrame, cluster_id: int) -> Dict[str, str]:
    """A cluster's members as picker options, its current representative first.

    Args:
        info: pdb_id, cluster_id, is_representative, dist_to_exemplar, protein_state.
        cluster_id: The cluster.

    Returns:
        Label ``pdb_id · holo|apo · d=…`` to pdb_id.
    """
    rows = info[(info["cluster_id"] == cluster_id) & (info["cluster_id"] != -1)].copy()  # -1: not clustered yet
    if rows.empty:
        return {}
    if "dist_to_exemplar" not in rows:
        rows["dist_to_exemplar"] = float("nan")
    rows["_rep"] = rows["is_representative"].astype(bool)
    rows = rows.sort_values(["_rep", "dist_to_exemplar", "pdb_id"], ascending=[False, True, True])
    return {f"{r.pdb_id} · {'holo' if r.protein_state == 'BOUND_STATE' else 'apo'} · d={r.dist_to_exemplar:.2g}":
            str(r.pdb_id) for r in rows.itertuples()}


def cluster_members(info: pd.DataFrame, names: Dict[int, str]) -> Dict[str, List[str]]:
    """Cluster name to its pdb_ids, largest cluster first (the order the profile and heatmap use).

    Args:
        info: pdb_id and cluster_id of the applied clustering.
        names: cluster_id to display name (``cluster_labels_of``).

    Returns:
        Ordered name → members.
    """
    sizes = info.loc[info["cluster_id"] != -1, "cluster_id"].value_counts()  # -1: not clustered yet
    order = sorted(sizes.index, key=lambda c: (-sizes[c], c))
    return {names.get(int(c), str(c)): info.loc[info["cluster_id"] == c, "pdb_id"].astype(str).tolist()
            for c in order}

