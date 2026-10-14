"""The actives to cross-dock: one per distinct molecule, with every crystal copy."""
from pathlib import Path
from typing import Dict

import pandas as pd
from eliot import log_message

from qpocket.io.screening import prepare_ligand_pdbqt
from qpocket.screen_config import ScreenConfig


def _canonical(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def distinct_actives(actives_smi: Path, master_csv: Path) -> pd.DataFrame:
    """One row per distinct active, with the crystal pose files it came from.

    Args:
        actives_smi: screening/actives.smi (header SMILES, PDB_ID, Ligand_Name).
        master_csv: metadata/master.csv, for each structure's ligand_file.

    Returns:
        active_id, smiles, canonical and crystal_files (";"-joined), in first-seen order.
    """
    rows = pd.read_csv(actives_smi, sep="\t")
    ligand_file = pd.read_csv(master_csv).set_index("pdb_id")["ligand_file"].to_dict()
    rows["canonical"] = rows["SMILES"].astype(str).map(_canonical)
    dropped = int(rows["canonical"].isna().sum())
    if dropped:
        log_message(message_type="warning", stage="distinct_actives", unparsed=dropped)
    out, used = [], {}
    for canonical, group in rows.dropna(subset=["canonical"]).groupby("canonical", sort=False):
        code = str(group["Ligand_Name"].iloc[0])
        used[code] = used.get(code, 0) + 1
        files = [str(ligand_file[p]) for p in group["PDB_ID"]
                 if isinstance(ligand_file.get(p), str) and ligand_file.get(p)]
        out.append({"active_id": code if used[code] == 1 else f"{code}_{used[code]}",
                    "smiles": str(group["SMILES"].iloc[0]), "canonical": canonical,
                    "crystal_files": ";".join(files)})
    return pd.DataFrame(out, columns=["active_id", "smiles", "canonical", "crystal_files"])


def prepare_ligands(actives: pd.DataFrame, out_dir: Path, cfg: ScreenConfig) -> Dict[str, Path]:
    """Ligand PDBQT files for docking.

    Args:
        actives: From distinct_actives.
        out_dir: ligands/ of the task folder.
        cfg: Screening configuration (protonation pH, conformers).

    Returns:
        Active id to PDBQT path, for the ones prepared.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for row in actives.itertuples():
        path = out_dir / f"{row.active_id}.pdbqt"
        if path.exists() or prepare_ligand_pdbqt(row.smiles, path, cfg.preparation, ligand_id=row.active_id):
            done[row.active_id] = path
    log_message(message_type="ligands_prepared", total=len(actives), prepared=len(done))
    return done
