"""Reduce a complex to the protein copy its ligand binds, and define that ligand for QPocket.

Generic codes such as LIG carry no bonds; the SDF supplies them through QPOCKET_CCD_CACHE.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

ResKey = Tuple[str, int, str, str]  # chain, resseq, icode, resname

BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CODE_PREFIXES = "QRSTUVWXYZ"
HEADER_PATTERNS = (r"^HEADER", r"^CRYST1", r"^REMARK   2 RESOLUTION",
                   r"^REMARK   3\s+RESOLUTION RANGE HIGH")


class Atom(NamedTuple):
    """One ATOM/HETATM record, fields at their fixed PDB columns."""

    record: str
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    xyz: np.ndarray
    occupancy: float
    element: str
    line: str


class Anchor(NamedTuple):
    """The ligand residue and conformer whose atoms are the SDF's atoms."""

    key: ResKey
    altloc: str
    name_to_index: Dict[str, int]


class Prepared(NamedTuple):
    """One complex reduced to its ligand's protein copy."""

    lines: List[str]
    kept_chain: str
    moved: List[str]
    renumbered: List[str]
    dropped: List[str]


def _element(line: str) -> str:
    column = line[76:78].strip() if len(line) >= 78 else ""
    if column:
        return column.capitalize()
    return next((c for c in line[12:16] if c.isalpha()), "").upper()


def read_atoms(text: str) -> List[Atom]:
    """ATOM and HETATM records of PDB text, in file order."""
    atoms = []
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append(Atom(
            record=line[:6].strip(), serial=int(line[6:11]), name=line[12:16].strip(),
            altloc=line[16].strip(), resname=line[17:20].strip(), chain=line[21],
            resseq=int(line[22:26]), icode=line[26].strip(),
            xyz=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            occupancy=float(line[54:60]) if line[54:60].strip() else 1.0,
            element=_element(line), line=line))
    return atoms


def residues(atoms: Iterable[Atom]) -> Dict[ResKey, List[Atom]]:
    """Atoms grouped by residue, in first-seen order."""
    grouped: Dict[ResKey, List[Atom]] = {}
    for atom in atoms:
        grouped.setdefault((atom.chain, atom.resseq, atom.icode, atom.resname), []).append(atom)
    return grouped


def conformers(atoms: List[Atom]) -> Dict[str, List[Atom]]:
    """A residue's atoms per altloc letter; blank-altloc atoms belong to every conformer."""
    letters = sorted({a.altloc for a in atoms if a.altloc})
    if not letters:
        return {"": list(atoms)}
    return {letter: [a for a in atoms if a.altloc in ("", letter)] for letter in letters}


def _heavy(atoms: Iterable[Atom]) -> List[Atom]:
    return [a for a in atoms if a.element not in ("H", "D")]


def match_anchor(atoms: List[Atom], mol: Chem.Mol, resname: Optional[str] = None,
                 tol: float = 0.05) -> Optional[Anchor]:
    """The ligand residue conformer whose heavy atoms are the molecule's, atom for atom.

    Args:
        atoms: The complex's atoms.
        mol: The ligand, with coordinates in the complex's frame.
        resname: Only residues of this name are tried; None tries every non-polymer residue.
        tol: Largest allowed distance (Å) between paired atoms.

    Returns:
        The matching residue, conformer and atom-name pairing, or None.
    """
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    positions = mol.GetConformer().GetPositions()[heavy_idx]
    symbols = [mol.GetAtomWithIdx(i).GetSymbol().capitalize() for i in heavy_idx]
    for key, res_atoms in residues(atoms).items():
        if resname is not None and key[3] != resname:
            continue
        if any(a.record == "ATOM" for a in res_atoms):
            continue
        for letter, conf_atoms in conformers(res_atoms).items():
            heavy = _heavy(conf_atoms)
            if len(heavy) != len(heavy_idx):
                continue
            xyz = np.array([a.xyz for a in heavy])
            dist = np.linalg.norm(xyz[:, None] - positions[None], axis=2)
            rows, cols = linear_sum_assignment(dist)
            if dist[rows, cols].max() > tol:
                continue
            if any(heavy[r].element != symbols[c] for r, c in zip(rows, cols)):
                continue
            return Anchor(key, letter, {heavy[r].name: heavy_idx[c] for r, c in zip(rows, cols)})
    return None


def _bond_order(bond: Chem.Bond) -> str:
    return {Chem.BondType.SINGLE: "SING", Chem.BondType.DOUBLE: "DOUB",
            Chem.BondType.TRIPLE: "TRIP"}[bond.GetBondType()]


def component_cif(code: str, mol: Chem.Mol, anchor: Anchor) -> str:
    """A CCD-format definition of the ligand, named by the anchor's PDB atom names.

    Args:
        code: The residue code the definition is for.
        mol: The ligand.
        anchor: Its match in the complex, which supplies the atom names.

    Returns:
        mmCIF text holding _chem_comp, _chem_comp_atom and _chem_comp_bond.
    """
    kekule = Chem.Mol(mol)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    name_of = {index: name for name, index in anchor.name_to_index.items()}
    lines = [f"data_{code}", "#", f"_chem_comp.id {code}", f"_chem_comp.name {code}",
             "_chem_comp.type NON-POLYMER", "#", "loop_", "_chem_comp_atom.comp_id",
             "_chem_comp_atom.atom_id", "_chem_comp_atom.type_symbol", "_chem_comp_atom.charge"]
    for index in sorted(name_of):
        atom = mol.GetAtomWithIdx(index)
        lines.append(f'{code} "{name_of[index]}" {atom.GetSymbol().upper()} {atom.GetFormalCharge()}')
    lines += ["#", "loop_", "_chem_comp_bond.comp_id", "_chem_comp_bond.atom_id_1",
              "_chem_comp_bond.atom_id_2", "_chem_comp_bond.value_order",
              "_chem_comp_bond.pdbx_aromatic_flag", "_chem_comp_bond.pdbx_stereo_config",
              "_chem_comp_bond.pdbx_ordinal"]
    ordinal = 0
    for bond in kekule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in name_of or b not in name_of:
            continue
        ordinal += 1
        aromatic = "Y" if mol.GetBondWithIdx(bond.GetIdx()).GetIsAromatic() else "N"
        lines.append(f'{code} "{name_of[a]}" "{name_of[b]}" {_bond_order(bond)} {aromatic} N {ordinal}')
    return "\n".join(lines + ["#", ""])


def named_bonds(mol: Chem.Mol, anchor: Anchor) -> frozenset:
    """The ligand's bonds as (atom-name pair, order), so two namings can be compared."""
    name_of = {index: name for name, index in anchor.name_to_index.items()}
    kekule = Chem.Mol(mol)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    return frozenset((frozenset((name_of[b.GetBeginAtomIdx()], name_of[b.GetEndAtomIdx()])), _bond_order(b))
                     for b in kekule.GetBonds()
                     if b.GetBeginAtomIdx() in name_of and b.GetEndAtomIdx() in name_of)


def assign_codes(smiles: Iterable[str]) -> Dict[str, str]:
    """One three-character residue code per distinct SMILES, in sorted order.

    Args:
        smiles: Canonical SMILES, repeats allowed.

    Returns:
        SMILES to code: Q00, Q01, …, QZZ, then R00, …

    Raises:
        ValueError: If there are more compounds than codes.
    """
    unique = sorted(set(smiles))
    if len(unique) > len(CODE_PREFIXES) * 36 * 36:
        raise ValueError(f"{len(unique)} compounds exceed the available residue codes")
    return {s: CODE_PREFIXES[i // 1296] + BASE36[i // 36 % 36] + BASE36[i % 36]
            for i, s in enumerate(unique)}


def resolution(text: str) -> Optional[float]:
    """Resolution (Å) from REMARK 2, else from REMARK 3's high-resolution limit."""
    for pattern in (r"^REMARK   2 RESOLUTION\.\s+([\d.]+)",
                    r"^REMARK   3\s+RESOLUTION RANGE HIGH \(ANGSTROMS\)\s*:\s*([\d.]+)"):
        found = re.search(pattern, text, re.MULTILINE)
        if found:
            return float(found.group(1))
    return None


def header(text: str) -> List[str]:
    """The lines worth keeping above the atoms: HEADER, CRYST1 and the resolution."""
    return [line for line in text.splitlines() if any(re.match(p, line) for p in HEADER_PATTERNS)]


def label(key: ResKey) -> str:
    """``chain:RESnum`` for reports."""
    chain, resseq, icode, resname = key
    return f"{chain}:{resname}{resseq}{icode}"


def _rewritten(atom: Atom, serial: int, chain: str, resseq: int, resname: str,
               altloc: str, occupancy: float) -> str:
    line = atom.line.ljust(80)
    return (f"{line[:6]}{serial:5d}{line[11:16]}{altloc or ' '}{resname:>3}{line[20]}{chain}"
            f"{resseq:4d}{line[26:54]}{occupancy:6.2f}{line[60:]}").rstrip()


def _distance(a: Iterable[Atom], b: np.ndarray) -> float:
    xyz = np.array([x.xyz for x in _heavy(a)])
    return float(np.min(np.linalg.norm(xyz[:, None] - b[None], axis=2)))


def _best_conformer(atoms: List[Atom]) -> List[Atom]:
    options = conformers(atoms)
    return max(options.values(), key=lambda c: (np.mean([a.occupancy for a in c]), -len(options)))


def prepare_complex(atoms: List[Atom], anchor: Anchor, code: str, head: List[str],
                    out_chain: Optional[str] = None) -> Prepared:
    """The protein chain the anchor binds, with every HETATM residue nearest it, as PDB lines.

    Args:
        atoms: The complex's atoms.
        anchor: The ligand's match to the SDF.
        code: Residue code that replaces the ligand's name.
        head: Lines written before the atoms (HEADER, CRYST1, resolution remarks).
        out_chain: Chain ID the output is written under; the kept chain's own when None.

    Returns:
        The lines, the kept chain (its ID in the input), and what was moved, renumbered or dropped.
    """
    # Other copies of the ligand keep their best conformer and the code when their atom
    # names match the anchor's; otherwise they are dropped.
    grouped = residues(atoms)
    protein = {c: np.array([a.xyz for a in _heavy(atoms) if a.chain == c and a.record == "ATOM"])
               for c in dict.fromkeys(a.chain for a in atoms if a.record == "ATOM")}
    anchor_atoms = conformers(grouped[anchor.key])[anchor.altloc]
    kept = min(protein, key=lambda c: _distance(anchor_atoms, protein[c]))
    anchor_names = {(a.name, a.element) for a in _heavy(anchor_atoms)}
    used = {(k[1], k[2]) for k in grouped if k[0] == kept}
    top = max(k[1] for k in grouped if k[0] == kept)
    moved: List[str] = []
    renumbered: List[str] = []
    dropped: List[str] = []
    out: List[Tuple[Atom, str, int, str, str, float]] = []   # atom, chain, resseq, resname, altloc, occ

    for key, res_atoms in grouped.items():
        chain, resseq, icode, resname = key
        if any(a.record == "ATOM" for a in res_atoms):
            if chain == kept:
                out += [(a, kept, resseq, resname, a.altloc, a.occupancy) for a in res_atoms]
            continue
        if key == anchor.key:
            chosen, name, flatten = anchor_atoms, code, True
        else:
            nearest = min(protein, key=lambda c: _distance(res_atoms, protein[c]))
            if nearest != kept:
                continue
            if resname == anchor.key[3]:
                chosen = _best_conformer(res_atoms)
                if {(a.name, a.element) for a in _heavy(chosen)} != anchor_names:
                    dropped.append(label(key))
                    continue
                name, flatten = code, True
            else:
                chosen, name, flatten = res_atoms, resname, False
        number = resseq
        if chain != kept:
            moved.append(label(key))
            if (resseq, icode) in used:
                while (top + 1, icode) in used:
                    top += 1
                number = top = top + 1
                renumbered.append(f"{label(key)}->{number}")
            used.add((number, icode))
            top = max(top, number)
        out += [(a, kept, number, name, "" if flatten else a.altloc, 1.0 if flatten else a.occupancy)
                for a in chosen]

    if out_chain is not None:
        out = [(a, out_chain, n, r, alt, occ) for a, _, n, r, alt, occ in out]
    lines = list(head)
    lines += [_rewritten(a, i, c, n, r, alt, occ) for i, (a, c, n, r, alt, occ) in enumerate(out, start=1)]
    return Prepared(lines + ["TER", "END"], kept, moved, renumbered, dropped)
