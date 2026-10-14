"""Cross-docking maths: in-place pose RMSD, the job grid, success rates (pure)."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem

from qpocket.core.screening import JOB_COLUMNS


def _any_bonds(smarts: str) -> str:
    """The same SMARTS with every bond outside atom brackets made "any bond"."""
    out, depth = [], 0
    for ch in smarts:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif depth == 0 and ch in "-=#:/\\":
            ch = "~"
        out.append(ch)
    return "".join(out)


def _graph_query(mol):
    """Heavy-atom graph of a molecule as a query: elements and connectivity, no bond orders."""
    return Chem.MolFromSmarts(_any_bonds(Chem.MolToSmarts(Chem.RemoveHs(mol, sanitize=False))))


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _best(docked, crystal, query) -> Optional[float]:
    ref = crystal.GetSubstructMatch(query)
    if not ref:
        return None
    c_xyz = crystal.GetConformer().GetPositions()[list(ref)]
    d_all = docked.GetConformer().GetPositions()
    matches = docked.GetSubstructMatches(query, uniquify=False, maxMatches=10000)
    return min((_rmsd(d_all[list(m)], c_xyz) for m in matches), default=None)


def pose_rmsd(docked, crystal, smiles: Optional[str] = None) -> Optional[float]:
    """Heavy-atom RMSD between two poses in the same frame, without superposing them.

    Args:
        docked: Docked pose.
        crystal: Crystal pose.
        smiles: The molecule; its graph is matched when given, else the crystal's own graph.

    Returns:
        The smallest RMSD over equivalent atom mappings, in Å, or None when the poses cannot be matched.
    """
    if docked is None or crystal is None:
        return None
    # Matched on the bond-order-free heavy-atom graph: symmetric groups (carboxylates,
    # phenyl flips) take their best mapping, and PDBQT/PDB bond-order disagreements do not matter.
    docked = Chem.RemoveHs(docked, sanitize=False)
    crystal = Chem.RemoveHs(crystal, sanitize=False)
    for source in ([Chem.MolFromSmiles(smiles)] if smiles else []) + [crystal]:
        if source is None:
            continue
        value = _best(docked, crystal, _graph_query(source))
        if value is not None:
            return value
    from rdkit.Chem import rdFMCS  # partially resolved ligands: fall back to the common substructure

    mcs = rdFMCS.FindMCS([docked, crystal], atomCompare=rdFMCS.AtomCompare.CompareElements,
                         bondCompare=rdFMCS.BondCompare.CompareAny, timeout=10)
    if mcs.numAtoms == 0:
        return None
    return _best(docked, crystal, Chem.MolFromSmarts(_any_bonds(mcs.smartsString)))


def crossdock_jobs(receptors: Dict[str, str], ligands: Dict[str, str], box: Dict[str, float], engine: str,
                   output_ext: str, exhaustiveness: int, num_modes: int, energy_range: float) -> pd.DataFrame:
    """Every ligand docked into every receptor, as a task table.

    Args:
        receptors: Representative id to receptor path (relative to the task folder).
        ligands: Active id to ligand path (relative).
        box: center_x/y/z and size_x/y/z.
        engine: DOCKING plug-in name.
        output_ext: Its output suffix.
        exhaustiveness: Search effort.
        num_modes: Poses kept.
        energy_range: Score window of kept poses, kcal/mol.

    Returns:
        One row per docking, in ligand-major order.
    """
    rows = []
    for active_id, ligand in ligands.items():
        for rep_id, receptor in receptors.items():
            job_id = f"job_{len(rows):05d}"
            rows.append({
                "job_id": job_id, "protein_pdbqt": receptor, "ligand_pdbqt": ligand,
                "output_pdbqt": f"poses/{job_id}{output_ext}",
                **{k: float(box[k]) for k in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")},
                "exhaustiveness": int(exhaustiveness), "num_modes": int(num_modes),
                "energy_range": float(energy_range), "molecule_type": "active", "pdb_id": rep_id,
                "ligand_id": active_id, "engine": engine, "representative": rep_id, "active_id": active_id,
            })
    return pd.DataFrame(rows, columns=list(JOB_COLUMNS) + ["representative", "active_id"])


def pair_table(poses: pd.DataFrame, threshold: float, top_k: int) -> pd.DataFrame:
    """One row per docking: its best-scored pose and its best pose among the top k.

    Args:
        poses: job_id, representative, active_id, rank, score, rmsd.
        threshold: Success when RMSD < threshold, Å.
        top_k: Poses considered for the top-k criterion.

    Returns:
        representative, active_id, n_poses, top1_score, top1_rmsd, best_topk_rmsd, top1_success, topk_success per job_id.
    """
    rows: List[dict] = []
    for job_id, group in poses.sort_values(["job_id", "rank"]).groupby("job_id", sort=False):
        top = group.iloc[0]
        best_k = group.head(int(top_k))["rmsd"].min()
        rows.append({"job_id": job_id, "representative": top["representative"], "active_id": top["active_id"],
                     "n_poses": len(group), "top1_score": top["score"], "top1_rmsd": top["rmsd"],
                     "best_topk_rmsd": best_k,
                     "top1_success": bool(pd.notna(top["rmsd"]) and top["rmsd"] < threshold),
                     "topk_success": bool(pd.notna(best_k) and best_k < threshold)})
    return pd.DataFrame(rows, columns=["job_id", "representative", "active_id", "n_poses", "top1_score",
                                       "top1_rmsd", "best_topk_rmsd", "top1_success", "topk_success"])


def success_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Success rates per representative, per active and overall.

    Args:
        pairs: From pair_table.

    Returns:
        level, name, n_pairs, top1_success_rate, topk_success_rate (percent).
    """
    def rates(level, name, group):
        return {"level": level, "name": str(name), "n_pairs": len(group),
                "top1_success_rate": 100.0 * group["top1_success"].mean() if len(group) else 0.0,
                "topk_success_rate": 100.0 * group["topk_success"].mean() if len(group) else 0.0}

    rows = [rates("representative", n, g) for n, g in pairs.groupby("representative", sort=True)]
    rows += [rates("active", n, g) for n, g in pairs.groupby("active_id", sort=True)]
    rows.append(rates("overall", "all", pairs))
    return pd.DataFrame(rows, columns=["level", "name", "n_pairs", "top1_success_rate", "topk_success_rate"])
