"""Prepare the protonated complexes the known-answer tests read."""
import sys
import urllib.request
from pathlib import Path

from qpocket.core.hydroxyl import optimise_hydroxyls
from qpocket.io.interactions import (load_molecule, protonated_ligand,
                                     write_complex, write_molecule)
from qpocket.io.tools import protonate_for_interactions

OUT = Path(__file__).resolve().parents[1] / "tests/fixtures/complexes"
# (PDB ID, ligand component ID, chain)
CASES = [("3PTB", "BEN", "A"), ("1STP", "BTN", "A"), ("7KG6", "WDG", "A")]


def prepare(pdb_id: str, ligand_id: str, chain: str) -> Path:
    """Download one entry and write its protonated complex.

    Args:
        pdb_id: PDB entry ID.
        ligand_id: Chemical-component ID of the ligand.
        chain: Chain to keep.

    Returns:
        Path of the written complex.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"{pdb_id}.pdb"
    if not raw.exists():
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", raw)

    protein, ligand = [], []
    for line in raw.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[21] != chain:
            continue
        if line[16] not in (" ", "A"):           # keep one alternate location
            continue
        cleaned = line[:16] + " " + line[17:]
        res_name = line[17:20].strip()
        if line.startswith("ATOM") and res_name != "HOH":
            protein.append(cleaned)
        elif res_name == ligand_id:
            ligand.append(cleaned)

    protein_pdb = OUT / f"{pdb_id}_protein.pdb"
    protein_pdb.write_text("\n".join(protein) + "\nEND\n", encoding="utf-8")
    pqr = protonate_for_interactions(protein_pdb, OUT / f"{pdb_id}.pqr")

    ligand_pdb = OUT / f"{pdb_id}_ligand.pdb"
    ligand_pdb.write_text("\n".join(ligand) + "\nEND\n", encoding="utf-8")
    ligand_mol = protonated_ligand(ligand_pdb)
    complex_pdb = write_complex(OUT / f"{pdb_id}_complex.pdb", pqr, ligand_mol, ligand_id)
    # pdb2pqr placed the hydroxyl hydrogens before the ligand existed, so their
    # rotamers are re-chosen now that it is present.
    return write_molecule(complex_pdb, optimise_hydroxyls(load_molecule(complex_pdb)))


def main() -> None:
    """Prepare every fixture."""
    for pdb_id, ligand_id, chain in CASES:
        print("wrote", prepare(pdb_id, ligand_id, chain).name)


if __name__ == "__main__":
    sys.exit(main())
