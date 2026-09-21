"""Decoy generation (LUDe-style) for virtual screening."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from eliot import log_message
from molvs import Standardizer
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from qpocket.core.screening import calc_descriptors, standardize_molecule
from qpocket.screen_config import DecoyConfig


def generate_decoys_lude(
    actives_path: Path,
    chembl_dir: Path,
    output_dir: Path,
    config: DecoyConfig,
) -> None:
    """Property-matched decoy generation following LUDe methodology.

    Reads actives from *actives_path* (tab-separated: SMILES, PDB_ID, Ligand_Name),
    searches *chembl_dir* ``.txt`` files for property-matched, dissimilar
    molecules, and writes ``decoys.smi`` to *output_dir*.

    The output format is ``SMILES\\tActive_ID_decoy_N`` — consistent with
    ``actives.smi``.

    Args:
        actives_path: Path to ``actives.smi`` (tab-separated).
        chembl_dir: Directory containing ChEMBL ``.txt`` / ``.csv`` files.
        output_dir: Directory in which to write ``decoys.smi``.
        config: :class:`~qpocket.screen_config.DecoyConfig` instance.
    """
    log_message(message_type="decoy_gen_start", actives=str(actives_path))

    if not actives_path.exists():
        log_message(message_type="error", error="Actives file missing", path=str(actives_path))
        return

    # actives.smi format: SMILES\tPDB_ID\tLigand_Name  (no header)
    df_actives = pd.read_csv(actives_path, sep="\t", header=None, names=["SMILES", "PDB_ID", "Ligand_Name"])
    actives_smiles = df_actives["SMILES"].tolist()
    active_ids = df_actives["PDB_ID"].tolist()

    standardizer = Standardizer()
    chembl_files = list(chembl_dir.glob("*.csv")) + list(chembl_dir.glob("*.txt"))

    if not chembl_files:
        log_message(message_type="warning", msg="No ChEMBL files found in chembl_dir", dir=str(chembl_dir))
        return

    decoy_results = []
    prop_cols = ["MW", "LogP", "rb", "Hba", "Hbd", "charge"]

    # Tolerance limits matching DecoyConfig fields
    limits = {
        "MW": config.mw_diff,
        "LogP": config.logp_diff,
        "rb": config.rb_diff,
        "Hba": config.hba_diff,
        "Hbd": config.hbd_diff,
        "charge": config.charge_diff,
    }

    for i, smi_active in enumerate(actives_smiles):
        active_id = active_ids[i] if i < len(active_ids) else f"Query_{i}"

        mol_active = standardize_molecule(smi_active, standardizer)
        if not mol_active:
            continue

        props_active, _ = calc_descriptors(mol_active)
        fp_active = AllChem.GetMorganFingerprintAsBitVect(mol_active, 2, nBits=1024)

        found_decoys: list = []
        random.seed(i)
        available_dbs = chembl_files.copy()

        while len(found_decoys) < config.n_decoys_per_active and available_dbs:
            db_idx = random.randrange(len(available_dbs))
            db_path = available_dbs.pop(db_idx)

            try:
                df_db = pd.read_csv(db_path, sep="\t")

                # ChEMBL export uses "SMILE" (not "SMILES")
                smiles_col = "SMILE" if "SMILE" in df_db.columns else "SMILES"
                if smiles_col not in df_db.columns:
                    log_message(
                        message_type="warning",
                        msg=f"No SMILES column in {db_path.name}, skipping",
                    )
                    continue

                # Keep original props; compute diffs on a copy to avoid overwriting
                prop_present = [c for c in prop_cols if c in df_db.columns]
                if not prop_present:
                    continue

                df_diffs = df_db[prop_present].sub(props_active[prop_present], axis=1).abs()

                mask = pd.Series([True] * len(df_db), index=df_db.index)
                for col in prop_present:
                    if col in limits:
                        mask &= df_diffs[col] <= limits[col]

                candidates = df_db[mask].copy()
                if len(candidates) < 50:
                    continue

                for _, row in candidates.iterrows():
                    decoy_smi = row[smiles_col]
                    if not isinstance(decoy_smi, str):
                        continue
                    mol_decoy = Chem.MolFromSmiles(decoy_smi)
                    if not mol_decoy:
                        continue

                    fp_decoy = AllChem.GetMorganFingerprintAsBitVect(mol_decoy, 2, nBits=1024)
                    sim = DataStructs.TanimotoSimilarity(fp_active, fp_decoy)

                    if sim <= config.max_similarity:
                        decoy_label = f"{active_id}_decoy_{len(found_decoys) + 1}"
                        found_decoys.append((decoy_smi, decoy_label))
                        if len(found_decoys) >= config.n_decoys_per_active:
                            break

            except Exception as e:
                log_message(message_type="db_error", file=str(db_path), error=str(e))

        decoy_results.extend(found_decoys)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "decoys.smi"

    if decoy_results:
        with open(out_file, "w") as f:
            for smi, label in decoy_results:
                f.write(f"{smi}\t{label}\n")

        log_message(message_type="decoys_saved", path=str(out_file), count=len(decoy_results))
    else:
        log_message(message_type="warning", msg="No decoys generated")


# Known issue (spec §6): unused and out of date; prepare() is the entry point.
def run_screening_prep(
    base_dir: str,
    chembl_path: str,
    box_config: dict,
    decoy_config: dict,
) -> None:
    """Orchestrates Vina Box Extraction + Decoy Generation."""
    from qpocket.io.screening import extract_vina_box

    BASE = Path(base_dir)
    DIRS = {
        "metadata": BASE / "metadata",
        "ligands": BASE / "metadata" / "ligands",
        "cavities": BASE / "metadata" / "cavities",
        "screening": BASE / "screening",
    }

    chembl_dir = Path(chembl_path)

    # 1. Vina Box
    extract_vina_box(box_config, DIRS)

    # 2. Decoys
    actives_path = DIRS["screening"] / "actives.smi"
    if actives_path.exists():
        cfg = DecoyConfig(
            n_decoys_per_active=decoy_config.get("max_decoys", 50),
            max_similarity=decoy_config.get("max_similarity", 0.2),
        )
        generate_decoys_lude(actives_path, chembl_dir, DIRS["screening"], cfg)
    else:
        log_message(message_type="warning", msg="Actives file missing, skipping decoy generation")
