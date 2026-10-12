"""The ligand sheet: distinct crystallised molecules of a run, drawn and tabulated."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd


def in_scope(ligand_rows: pd.DataFrame, active_info: pd.DataFrame,
             cluster: str) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    """Ligand rows of this run's structures, optionally of one cluster.

    Args:
        ligand_rows: pdb_id, code, canonical per crystallised pose.
        active_info: Run table with pdb_id and cluster_id.
        cluster: A cluster id, or "all clusters".

    Returns:
        (rows, rows naming a structure outside the run, pdb_id to cluster).
    """
    cluster_of = {str(row.pdb_id).upper(): int(row.cluster_id)
                  for row in active_info.itertuples() if pd.notna(row.cluster_id)}
    rows = ligand_rows[ligand_rows["pdb_id"].isin(cluster_of)]
    outside = len(ligand_rows) - len(rows)
    if cluster != "all clusters":
        rows = rows[rows["pdb_id"].map(lambda p: str(cluster_of[p]) == str(cluster))]
    return rows, outside, cluster_of


def distinct_molecules(rows: pd.DataFrame, cluster_of: Dict[str, int]) -> List[dict]:
    """One entry per distinct molecule, most widely crystallised first.

    Args:
        rows: From in_scope.
        cluster_of: pdb_id to cluster.

    Returns:
        Dicts with canonical, codes, pdb_ids, clusters and poses.
    """
    unique = []
    for canonical, group in rows.groupby("canonical", sort=False):
        pdb_ids = list(dict.fromkeys(group["pdb_id"]))
        unique.append({"canonical": canonical, "codes": list(dict.fromkeys(group["code"])),
                       "pdb_ids": pdb_ids, "clusters": sorted({cluster_of[p] for p in pdb_ids}),
                       "poses": len(group)})
    # Most widely crystallised first: those are the chemotypes the site really binds.
    unique.sort(key=lambda u: (-len(u["pdb_ids"]), u["codes"][0]))
    return unique


def depictions(shown: Sequence[dict], *, n_cols: int, all_clusters: bool) -> Tuple[str, list]:
    """The molecules as one SVG grid.

    Args:
        shown: Entries to draw.
        n_cols: Columns.
        all_clusters: Caption each molecule with its clusters.

    Returns:
        (SVG text, the RDKit molecules in drawing order).
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    rdDepictor.SetPreferCoordGen(True)  # cleaner ring layouts than the default
    mols, legends = [], []
    for u in shown:
        mol = Chem.MolFromSmiles(u["canonical"])
        rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
        more = len(u["pdb_ids"]) - 3
        # ASCII only, and pipes rather than the middle dot used elsewhere: RDKit
        # draws legends in its own font, which has no glyph for "·".
        caption = ["/".join(u["codes"]),
                   ", ".join(u["pdb_ids"][:3]) + (f" +{more}" if more > 0 else "")]
        if all_clusters:
            caption.append(f"cluster {', '.join(str(c) for c in u['clusters'])}")
        legends.append(" | ".join(caption))

    ncol = max(1, min(int(n_cols), len(mols)))
    nrow = -(-len(mols) // ncol)
    w, h = 300, 265
    drawer = rdMolDraw2D.MolDraw2DSVG(w * ncol, h * nrow, w, h)
    opts = drawer.drawOptions()
    opts.legendFontSize = 15
    # Per-panel scaling: one large molecule would otherwise shrink every other depiction.
    opts.drawMolsSameScale = False
    drawer.DrawMolecules(mols, legends=legends)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return (svg[svg.index("<svg"):] if "<svg" in svg else svg), mols


def properties(shown: Sequence[dict], mols: Sequence) -> pd.DataFrame:
    """Formula and drug-likeness properties of each drawn molecule.

    Args:
        shown: Entries drawn.
        mols: Their RDKit molecules.

    Returns:
        One row per molecule.
    """
    from rdkit.Chem import Descriptors
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    return pd.DataFrame([
        {"ligand": "/".join(u["codes"]), "structures": len(u["pdb_ids"]), "poses": u["poses"],
         "clusters": ", ".join(str(c) for c in u["clusters"]), "formula": CalcMolFormula(mol),
         "mw": round(Descriptors.MolWt(mol), 1), "clogp": round(Descriptors.MolLogP(mol), 2),
         "hbd": Descriptors.NumHDonors(mol), "hba": Descriptors.NumHAcceptors(mol),
         "tpsa": round(Descriptors.TPSA(mol), 1), "rot_bonds": Descriptors.NumRotatableBonds(mol),
         "pdb_ids": ", ".join(u["pdb_ids"]), "smiles": u["canonical"]}
        for u, mol in zip(shown, mols)
    ])


def notes(shown: Sequence[dict], unique: Sequence[dict], rows: pd.DataFrame, outside: int,
          n_unparsed: int) -> List[str]:
    """The caption under the sheet.

    Args:
        shown: Entries drawn.
        unique: Every distinct molecule.
        rows: Poses in scope.
        outside: Rows naming a structure outside the run.
        n_unparsed: Rows RDKit could not parse.

    Returns:
        Caption fragments.
    """
    out = [f"{len(shown)} of {len(unique)} distinct molecules drawn" if len(shown) < len(unique)
           else f"{len(unique)} distinct molecules",
           f"{len(rows)} crystallised poses across {rows['pdb_id'].nunique()} structures"]
    if outside:
        out.append(f"{outside} rows name a PDB entry outside this run, dropped")
    if n_unparsed:
        out.append(f"{n_unparsed} rows RDKit could not parse, dropped")
    return out
