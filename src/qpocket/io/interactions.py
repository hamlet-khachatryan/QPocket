"""Read a complex, perceive it, and detect its interactions."""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from eliot import log_message
from rdkit import Chem

from qpocket.core import detect
from qpocket.core.perception import component_ids, parse_pdb_atoms, perceive
from qpocket.core.protonation import assign_protonation
from qpocket.io.components import MissingComponentError, component_bonds


def pdb_line(record: str, serial: int, name: str, res_name: str, chain: str,
             res_seq: int, x: float, y: float, z: float, element: str) -> str:
    """One ATOM or HETATM line in the exact fixed columns readers expect.

    Args:
        record: ATOM or HETATM.
        serial: Atom serial number.
        name: Atom name.
        res_name: Residue name.
        chain: Chain identifier.
        res_seq: Residue sequence number.
        x: Cartesian x.
        y: Cartesian y.
        z: Cartesian z.
        element: Element symbol.

    Returns:
        A formatted PDB line, without a trailing newline.
    """
    name = name.strip()
    # Names shorter than four characters start in column 14, per the PDB convention.
    field = f"{name:<4s}" if len(name) == 4 else f" {name:<3s}"
    return (f"{record:<6s}{serial:5d} {field}{'':1s}{res_name:>3s} {chain:1s}{res_seq:4d}"
            f"{'':4s}{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}{'':10s}{element:>2s}")


def load_molecule(pdb_path: Path, no_implicit: bool = True) -> Chem.Mol:
    """Perceive a structure using chemical-component definitions.

    Args:
        pdb_path: Path to a PDB file.
        no_implicit: Forbid implicit hydrogens; pass False before adding them.

    Returns:
        The perceived molecule.

    Raises:
        MissingComponentError: If a component definition cannot be obtained.
    """
    atoms = parse_pdb_atoms(Path(pdb_path).read_text(encoding="utf-8"))
    templates = {comp: component_bonds(comp) for comp in component_ids(atoms)}
    return perceive(atoms, templates, no_implicit=no_implicit)


def protonated_ligand(pdb_path: Path, ph: float = 7.4) -> Chem.Mol:
    """Perceive a ligand, charge it for the given pH, and add its hydrogens.

    Args:
        pdb_path: Path to a PDB file holding one ligand.
        ph: pH at which to assign the protonation state.

    Returns:
        The ligand with formal charges, explicit hydrogens and coordinates.
    """
    # Perceived without NoImplicit, because RDKit will not add hydrogens to a
    # molecule that forbids implicit ones; the flag goes on afterwards.
    charged = assign_protonation(load_molecule(pdb_path, no_implicit=False), ph)
    mol = Chem.AddHs(charged, addCoords=True)
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    Chem.SanitizeMol(mol)
    return mol


def split_complex(mol: Chem.Mol, ligand_id: str) -> Tuple[Chem.Mol, Chem.Mol]:
    """Separate a perceived complex into its ligand and its protein.

    Args:
        mol: The perceived complex.
        ligand_id: Residue name of the ligand.

    Returns:
        The ligand molecule and the protein molecule.

    Raises:
        ValueError: If the ligand is not present.
    """
    wanted = ligand_id.strip().upper()
    ligand_indices = {atom.GetIdx() for atom in mol.GetAtoms()
                      if atom.GetPDBResidueInfo().GetResidueName().strip().upper() == wanted}
    if not ligand_indices:
        raise ValueError(f"ligand {ligand_id!r} is not present in the structure")

    def subset(keep):
        editable = Chem.RWMol(mol)
        for index in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in keep),
                            reverse=True):
            editable.RemoveAtom(index)
        out = editable.GetMol()
        Chem.SanitizeMol(out)
        return out

    everything = {atom.GetIdx() for atom in mol.GetAtoms()}
    return subset(ligand_indices), subset(everything - ligand_indices)


def interactions_for(pdb_path: Path, ligand_id: Optional[str],
                     enabled: Optional[Sequence[str]] = None,
                     parameters: Optional[Mapping[str, Mapping]] = None) -> List[str]:
    """Interactions between a structure's ligand and its protein.

    Args:
        pdb_path: Path to the protonated complex.
        ligand_id: Chemical-component ID of the ligand, or None for an apo structure.
        enabled: ProLIF interaction names to request.
        parameters: Per-interaction parameter overrides.

    Returns:
        Interaction strings, or one of no_ligand_found, file_not_found, analysis_error.
    """
    plf = detect.load_prolif()

    if not ligand_id or not str(ligand_id).strip():
        return ["no_ligand_found"]
    path = Path(pdb_path)
    if not path.exists():
        return ["file_not_found"]
    try:
        ligand_mol, protein_mol = split_complex(load_molecule(path), str(ligand_id))
    except ValueError:
        return ["no_ligand_found"]
    except MissingComponentError as error:
        log_message(message_type="error", stage="perception", pdb_path=str(path),
                    component=error.comp_id, error=str(error))
        return ["analysis_error"]
    except Exception as error:
        log_message(message_type="error", stage="perception", pdb_path=str(path),
                    error=str(error))
        return ["analysis_error"]

    try:
        found = detect.interactions(plf.Molecule.from_rdkit(ligand_mol),
                                    plf.Molecule.from_rdkit(protein_mol),
                                    enabled=enabled, parameters=parameters)
    except Exception as error:
        log_message(message_type="error", stage="interaction_detection",
                    pdb_path=str(path), error=str(error))
        return ["analysis_error"]
    return found or ["no_ligand_found"]


def write_molecule(path: Path, mol: Chem.Mol) -> Path:
    """Write a perceived molecule back to PDB, keeping its residue identity.

    Args:
        path: Destination PDB path.
        mol: A perceived molecule with one conformer.

    Returns:
        The path written.
    """
    conformer = mol.GetConformer()
    lines: List[str] = []
    hetatm_serials = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        position = conformer.GetAtomPosition(atom.GetIdx())
        serial = atom.GetIdx() + 1
        record = "HETATM" if info.GetIsHeteroAtom() else "ATOM"
        lines.append(pdb_line(record, serial, info.GetName(), info.GetResidueName(),
                              info.GetChainId() or "A", info.GetResidueNumber(),
                              position.x, position.y, position.z, atom.GetSymbol()))
        if record == "HETATM":
            hetatm_serials[atom.GetIdx()] = serial
    # Connectivity for HETATM records, which readers do not infer from templates.
    for index, serial in hetatm_serials.items():
        partners = [neighbour.GetIdx() + 1
                    for neighbour in mol.GetAtomWithIdx(index).GetNeighbors()]
        for start in range(0, len(partners), 4):
            lines.append("CONECT" + f"{serial:5d}"
                         + "".join(f"{p:5d}" for p in partners[start:start + 4]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return Path(path)


def write_complex(path: Path, protein_pqr: Path, ligand_mol: Optional[Chem.Mol],
                  ligand_id: Optional[str]) -> Path:
    """Write the docking-ready structure: protonated protein plus protonated ligand.

    Args:
        path: Destination PDB path.
        protein_pqr: PQR written by pdb2pqr.
        ligand_mol: Protonated ligand, or None for an apo structure.
        ligand_id: Chemical-component ID of the ligand, or None.

    Returns:
        The path written.
    """
    lines: List[str] = []
    serial = 1
    for line in Path(protein_pqr).read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        parts = line.split()
        name, res_name = parts[2], parts[3]
        res_seq = "".join(c for c in parts[4] if c.isdigit() or c == "-")
        if not res_seq:              # a chain column is present, shifting the fields
            res_seq = "".join(c for c in parts[5] if c.isdigit() or c == "-")
        x, y, z = parts[-5], parts[-4], parts[-3]
        element = name.lstrip("0123456789")[:1]
        lines.append(pdb_line("ATOM", serial, name, res_name, "A", int(res_seq),
                              float(x), float(y), float(z), element))
        serial += 1

    if ligand_mol is not None and ligand_id:
        conformer = ligand_mol.GetConformer()
        serial_of = {}
        for atom in ligand_mol.GetAtoms():
            position = conformer.GetAtomPosition(atom.GetIdx())
            info = atom.GetPDBResidueInfo()
            name = info.GetName().strip() if info else f"{atom.GetSymbol()}{atom.GetIdx()}"
            lines.append(pdb_line("HETATM", serial, name, ligand_id.upper(), "A", 900,
                                  position.x, position.y, position.z, atom.GetSymbol()))
            serial_of[atom.GetIdx()] = serial
            serial += 1
        # A reader that takes HETATM connectivity from CONECT sees no donors without this.
        for atom in ligand_mol.GetAtoms():
            partners = [serial_of[n.GetIdx()] for n in atom.GetNeighbors()]
            for start in range(0, len(partners), 4):
                lines.append("CONECT" + f"{serial_of[atom.GetIdx()]:5d}"
                             + "".join(f"{p:5d}" for p in partners[start:start + 4]))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return Path(path)
