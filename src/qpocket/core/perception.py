"""Build RDKit molecules from PDB atoms plus chemical-component definitions."""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Mapping, NamedTuple, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from qpocket.constants import canonical_residue

ORDER = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
         "TRIP": Chem.BondType.TRIPLE}

#: Two-letter elements that would be misread from the first character of a name.
TWO_LETTER = {"CL", "BR", "ZN", "MG", "MN", "FE", "SE", "NA", "CU", "NI", "CO"}

#: Maximum bond order each element may carry, used when attaching hydrogens.
CAPACITY = {"C": 4, "N": 4, "O": 2, "S": 2, "SE": 2, "P": 5, "B": 3}

H_BOND_MAX = 1.35
PEPTIDE_MAX = 2.0
DISULFIDE_MAX = 2.5


class AtomRecord(NamedTuple):
    """One atom read from a PDB file.

    Attributes:
        name: Atom name, stripped.
        res_name: Residue name, upper case.
        chain_id: Chain identifier.
        res_seq: Residue sequence number.
        icode: Insertion code.
        element: Element symbol, capitalised.
        is_hetatm: Whether the record was a HETATM.
        xyz: Cartesian coordinates.
    """

    name: str
    res_name: str
    chain_id: str
    res_seq: int
    icode: str
    element: str
    is_hetatm: bool
    xyz: Tuple[float, float, float]


def element_from(name: str, column: str) -> str:
    """Element of an atom, from the element column or else the atom name.

    Args:
        name: Atom name.
        column: Contents of the PDB element column.

    Returns:
        Capitalised element symbol.
    """
    if column.strip():
        return column.strip().capitalize()
    stripped = name.strip().lstrip("0123456789")
    if stripped[:2].upper() in TWO_LETTER:
        return stripped[:2].capitalize()
    return stripped[:1].capitalize()


def parse_pdb_atoms(text: str) -> List[AtomRecord]:
    """Read ATOM and HETATM records from PDB text.

    Args:
        text: Contents of a PDB file.

    Returns:
        One record per atom, in file order.
    """
    atoms: List[AtomRecord] = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16]
        element = element_from(name, line[76:78] if len(line) >= 78 else "")
        if not element:
            continue
        atoms.append(AtomRecord(
            name=name.strip(),
            res_name=line[17:20].strip().upper(),
            chain_id=line[21],
            res_seq=int(line[22:26]),
            icode=line[26],
            element=element,
            is_hetatm=line.startswith("HETATM"),
            xyz=(float(line[30:38]), float(line[38:46]), float(line[46:54])),
        ))
    return atoms


def component_ids(atoms: Sequence[AtomRecord]) -> Set[str]:
    """Chemical components present, as canonical names.

    Args:
        atoms: Parsed atoms.

    Returns:
        Component IDs whose definitions are needed.
    """
    return {canonical_residue(a.res_name) for a in atoms}


def _add_template_bonds(mol, residues, templates) -> None:
    """Add heavy-atom bonds from the component definitions."""
    for (_chain, _num, _icode, res_name), names in residues.items():
        for pair, order in templates.get(canonical_residue(res_name), {}).items():
            first, second = tuple(pair)
            if first not in names or second not in names:
                continue
            if mol.GetBondBetweenAtoms(names[first], names[second]) is not None:
                continue
            mol.AddBond(names[first], names[second], ORDER[order])


def _attach_hydrogens(mol, xyz) -> None:
    """Bond each hydrogen to its nearest heavy atom that still has free valence."""
    heavy = np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"])
    if not len(heavy):
        return
    heavy_xyz = xyz[heavy]
    used = {int(i): sum(b.GetBondTypeAsDouble()
                        for b in mol.GetAtomWithIdx(int(i)).GetBonds()) for i in heavy}
    pending = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H" or atom.GetDegree():
            continue
        distances = np.linalg.norm(heavy_xyz - xyz[atom.GetIdx()], axis=1)
        pending.append((float(distances.min()), atom.GetIdx(), distances))
    # nearest first, so a contested hydrogen goes to the atom it belongs to
    for _nearest, index, distances in sorted(pending):
        for rank in np.argsort(distances):
            if distances[rank] > H_BOND_MAX:
                break
            target = int(heavy[int(rank)])
            limit = CAPACITY.get(mol.GetAtomWithIdx(target).GetSymbol().upper(), 4)
            if used[target] < limit:
                mol.AddBond(index, target, Chem.BondType.SINGLE)
                used[target] += 1
                break


def _link_residues(mol, residues, xyz) -> None:
    """Add peptide bonds and disulfides, the only bonds geometry decides."""
    ordered = sorted(residues, key=lambda key: (key[0], key[1], key[2]))
    for previous, following in zip(ordered, ordered[1:]):
        carbon = residues[previous].get("C")
        nitrogen = residues[following].get("N")
        if previous[0] != following[0] or carbon is None or nitrogen is None:
            continue
        if mol.GetBondBetweenAtoms(carbon, nitrogen) is None and \
                np.linalg.norm(xyz[carbon] - xyz[nitrogen]) < PEPTIDE_MAX:
            mol.AddBond(carbon, nitrogen, Chem.BondType.SINGLE)
    sulfurs = [names["SG"] for names in residues.values() if "SG" in names]
    for i, first in enumerate(sulfurs):
        for second in sulfurs[i + 1:]:
            if mol.GetBondBetweenAtoms(first, second) is None and \
                    np.linalg.norm(xyz[first] - xyz[second]) < DISULFIDE_MAX:
                mol.AddBond(first, second, Chem.BondType.SINGLE)


def _assign_charges(mol) -> None:
    """Express the protonation states pdb2pqr produced as formal charges."""
    for atom in mol.GetAtoms():
        total = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        symbol = atom.GetSymbol()
        if symbol == "N" and total > 3:
            atom.SetFormalCharge(int(total - 3))
        elif symbol == "O" and total == 1 and atom.GetDegree() == 1:
            atom.SetFormalCharge(-1)
        elif symbol in ("Zn", "Mg", "Ca", "Mn", "Fe", "Cu", "Ni") and not atom.GetDegree():
            atom.SetFormalCharge(2)
        elif symbol in ("Na", "K") and not atom.GetDegree():
            atom.SetFormalCharge(1)


def perceive(atoms: Sequence[AtomRecord],
             templates: Mapping[str, Dict[FrozenSet[str], str]]) -> Chem.Mol:
    """Build a sanitised molecule from atoms and their component definitions.

    Args:
        atoms: Parsed PDB atoms.
        templates: Bond tables keyed by canonical component ID.

    Returns:
        Molecule with one conformer, explicit hydrogens and no implicit ones.

    Raises:
        ValueError: If the molecule cannot be sanitised.
    """
    mol = Chem.RWMol()
    residues: Dict[Tuple[str, int, str, str], Dict[str, int]] = {}
    for record in atoms:
        atom = Chem.Atom(record.element)
        info = Chem.AtomPDBResidueInfo()
        info.SetName(f"{record.name:<4s}" if len(record.name) == 4
                     else f" {record.name:<3s}")
        info.SetResidueName(record.res_name)
        info.SetChainId(record.chain_id)
        info.SetResidueNumber(record.res_seq)
        info.SetInsertionCode(record.icode)
        info.SetIsHeteroAtom(record.is_hetatm)
        atom.SetMonomerInfo(info)
        index = mol.AddAtom(atom)
        key = (record.chain_id, record.res_seq, record.icode, record.res_name)
        residues.setdefault(key, {})[record.name] = index

    xyz = np.array([record.xyz for record in atoms], dtype=float)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(xyz):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conformer)

    _add_template_bonds(mol, residues, templates)
    _attach_hydrogens(mol, xyz)
    _link_residues(mol, residues, xyz)
    _assign_charges(mol)
    # Hydrogens are all explicit; RDKit must not add implicit ones on top.
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)

    out = mol.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception as error:
        raise ValueError(f"could not sanitise the perceived molecule: {error}") from error
    return out
