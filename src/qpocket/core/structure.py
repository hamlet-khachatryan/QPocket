"""Structure operations, in Biopython, standing in for PyMOL."""

# The pipeline used PyMOL for four things -- loading, superposition, cleaning and
# saving -- and PyMOL is the one dependency with no pip wheel for older Linux
# (RHEL 8, glibc 2.28), which is most cluster nodes. Everything it did is done here
# with Biopython, which the pipeline already depends on, and numpy.
#
# Two behaviours are reproduced deliberately, because results depend on them:
#
# **Superposition** follows ``cmd.align``: a BLOSUM62 sequence alignment decides
# which residues correspond, every atom with a matching name in a matched pair
# is fitted, and the fit is refined for up to five cycles, each dropping pairs
# further than twice the current RMSD. The whole mobile structure then moves, not
# only the fitted chain, so auxiliary chains land in the reference frame too.
#
# **CONECT records** follow what PyMOL wrote, because PLIP -- which reads the
# aligned files -- uses them to perceive ligand bonds. Dropping them changed PLIP's
# hydrogen-bond calls on 5 of 16 PARG structures. PyMOL's rule, recovered from the
# files it wrote: bonds within a residue by covalent radii (+0.45 A), never between
# alternate locations; peptide C-N links between residues; and the covalent links
# annotated in the input (``_struct_conn`` in mmCIF, ``LINK`` or inter-residue
# ``CONECT`` in PDB). Only pairs involving a HETATM are written, as PyMOL did.

from __future__ import annotations

from typing import Dict, Iterable, List, NamedTuple, Sequence, Set, Tuple

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from scipy.spatial import cKDTree

from qpocket.constants import STANDARD_AA, STANDARD_AA_ATOMS, THREE_TO_ONE_MAPPING

#: Residue names PyMOL's ``solvent`` selection removes.
WATER = frozenset({"HOH", "WAT", "H2O", "DOD", "D2O", "SOL", "TIP", "TIP3", "TIP4", "SPC"})

#: Covalent radii (A) used for bond perception, matching PyMOL's to within the
#: tolerance below on every CONECT record in the PARG run.
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07, "F": 0.57,
    "CL": 1.02, "BR": 1.20, "I": 1.39, "B": 0.84, "SE": 1.20, "FE": 1.32,
    "ZN": 1.22, "MG": 1.41, "CA": 1.76, "NA": 1.66, "K": 2.03, "CU": 1.32,
    "MN": 1.39, "CO": 1.26, "NI": 1.24,
}
DEFAULT_RADIUS = 0.77
BOND_TOLERANCE = 0.45
PEPTIDE_MAX = 1.8

#: cmd.align defaults.
ALIGN_CYCLES = 5
ALIGN_CUTOFF = 2.0

#: Identifies one atom across files: chain, residue number, insertion code,
#: residue name, atom name.
AtomKey = Tuple[str, int, str, str, str]


class Fit(NamedTuple):
    """Result of a superposition, in the terms ``cmd.align`` reports.

    Attributes:
        rmsd: RMSD over the pairs kept after refinement.
        n_atoms: Atom pairs kept after refinement.
        rmsd_before: RMSD over all matched pairs, before refinement.
        n_atoms_before: Matched atom pairs before refinement.
        n_residues: Residue pairs the sequence alignment matched.
        cycles: Refinement cycles run.
        rotation: 3x3 matrix, applied as ``coord @ rotation + translation`` (Biopython's ``transform`` convention).
        translation: Length-3 vector.
    """

    rmsd: float
    n_atoms: int
    rmsd_before: float
    n_atoms_before: int
    n_residues: int
    cycles: int
    rotation: np.ndarray
    translation: np.ndarray


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------




def without_water(structure: Structure) -> Tuple[Structure, int]:
    """Copy of a structure with solvent residues removed.

    Args:
        structure: Input structure (not modified).

    Returns:
        (new structure, residues removed).
    """
    out = structure.copy()
    removed = 0
    for chain in out[0]:
        for residue in [r for r in chain if r.get_resname().strip() in WATER]:
            chain.detach_child(residue.id)
            removed += 1
    return out, removed




def atom_key(chain: str, resseq, icode: str, resname: str, atom: str) -> AtomKey:
    """Hashable identity of an atom: chain, residue number, insertion code, residue and atom name.

    Args:
        chain: Chain ID.
        resseq: Residue number.
        icode: Insertion code.
        resname: Residue name.
        atom: Atom name.

    Returns:
        The key.
    """
    icode = "" if icode in (None, "?", ".", " ") else str(icode).strip()
    return (str(chain).strip(), int(str(resseq).strip()), icode,
            str(resname).strip(), str(atom).strip())


# --------------------------------------------------------------------------
# Residues and sequences
# --------------------------------------------------------------------------


def is_polymer(residue: Residue) -> bool:
    """An amino acid in the chain: it has the backbone N, CA and C."""
    return all(name in residue for name in ("N", "CA", "C"))


#: Longest C–N distance (Å) still read as a peptide bond.
PEPTIDE_BOND = 2.0


def _bonded(c_side: Residue, n_side: Residue) -> bool:
    return "C" in c_side and "N" in n_side and float(np.linalg.norm(c_side["C"].coord - n_side["N"].coord)) <= PEPTIDE_BOND


def modified_residues(chain: Chain) -> Dict[tuple, str]:
    """Non-standard amino acids peptide-bonded into the chain (PTR, SEP, TPO, MSE …), with their parent residue.

    Args:
        chain: Biopython chain.

    Returns:
        Residue id to the parent's three-letter name (PTR → TYR); a free amino-acid-like HET group is not one.
    """
    from Bio.Data.PDBData import protein_letters_1to3, protein_letters_3to1_extended

    residues = list(chain)
    found: Dict[tuple, str] = {}
    for k, residue in enumerate(residues):
        name = residue.get_resname().strip().upper()
        letter = protein_letters_3to1_extended.get(name)
        if name in STANDARD_AA or not letter or letter == "X" or not is_polymer(residue):
            continue
        before = k > 0 and _bonded(residues[k - 1], residue)
        after = k + 1 < len(residues) and _bonded(residue, residues[k + 1])
        if before or after:
            found[residue.id] = protein_letters_1to3[letter].upper()
    return found


def standardised(structure: Structure) -> Tuple[Structure, int]:
    """Copy with every modified residue in a chain turned into its parent: renamed, extra atoms dropped.

    Args:
        structure: Input structure (not modified).

    Returns:
        (new structure, residues converted); MSE keeps its selenium as the methionine's SD.
    """
    out = structure.copy()
    converted = 0
    for model in out:
        for chain in model:
            for rid, parent in modified_residues(chain).items():
                residue = chain[rid]
                if parent == "MET" and "SE" in residue and "SD" not in residue:
                    residue["SE"].id, residue["SE"].name, residue["SE"].fullname = "SD", "SD", " SD "
                    residue["SE"].element = "S"
                    residue.child_dict["SD"] = residue.child_dict.pop("SE")
                for atom in [a for a in residue if a.get_id() not in STANDARD_AA_ATOMS[parent]]:
                    residue.detach_child(atom.get_id())
                residue.resname = parent
                residue.id = (" ", rid[1], rid[2])
                converted += 1
    return out, converted


def polymer_residues(chain: Chain) -> List[Residue]:
    """Amino-acid residues of a chain, in order.

    Args:
        chain: Biopython chain.

    Returns:
        Residues with a backbone.
    """
    return [r for r in chain if is_polymer(r)]


def one_letter(residue: Residue) -> str:
    """One-letter code of a residue, X when unknown.

    Args:
        residue: Biopython residue.

    Returns:
        Single letter.
    """
    return THREE_TO_ONE_MAPPING.get(residue.get_resname().strip(), "X")


def chain_sequence(chain: Chain) -> str:
    """One-letter sequence of a chain's amino acids, as ``cmd.get_fastastr``."""
    return "".join(one_letter(r) for r in polymer_residues(chain))


def ca_coords(chain: Chain) -> np.ndarray:
    """CA coordinates of a chain's amino acids.

    Args:
        chain: Biopython chain.

    Returns:
        (N, 3) array.
    """
    return np.array([r["CA"].coord for r in chain if "CA" in r], dtype=float).reshape(-1, 3)


# --------------------------------------------------------------------------
# Superposition
# --------------------------------------------------------------------------


def _aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    # Overhangs are not penalised: a construct with an extra tag or a missing
    # terminus should still align on the part both structures have.
    aligner.end_gap_score = 0.0
    return aligner


def residue_pairs(mobile: Sequence[Residue],
                  reference: Sequence[Residue]) -> List[Tuple[Residue, Residue]]:
    """Residues that correspond, by a BLOSUM62 alignment of the two sequences."""
    seq_m = "".join(one_letter(r) for r in mobile)
    seq_r = "".join(one_letter(r) for r in reference)
    if not seq_m or not seq_r:
        return []
    alignment = next(iter(_aligner().align(seq_r, seq_m)))
    pairs = []
    for (r0, r1), (m0, m1) in zip(*alignment.aligned):
        for offset in range(r1 - r0):
            pairs.append((mobile[m0 + offset], reference[r0 + offset]))
    return pairs


def _atom_pairs(pairs: Sequence[Tuple[Residue, Residue]]) -> Tuple[np.ndarray, np.ndarray]:
    """Atoms present under the same name in both residues of each pair."""
    mob, ref = [], []
    for res_m, res_r in pairs:
        for atom in res_m:
            if atom.get_id() not in res_r:
                continue
            mob.append(atom.coord)
            ref.append(res_r[atom.get_id()].coord)
    return np.array(mob, dtype=float).reshape(-1, 3), np.array(ref, dtype=float).reshape(-1, 3)


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation and translation taking ``mobile`` onto ``reference``."""
    mc, rc = mobile.mean(axis=0), reference.mean(axis=0)
    h = (mobile - mc).T @ (reference - rc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, d])
    rotation = u @ correction @ vt
    translation = rc - mc @ rotation
    return rotation, translation


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))) if len(a) else float("nan")


def superpose(mobile: Sequence[Residue], reference: Sequence[Residue],
              cycles: int = ALIGN_CYCLES, cutoff: float = ALIGN_CUTOFF) -> Fit:
    """Fit ``mobile`` residues onto ``reference`` residues, as ``cmd.align``.

    Raises:
        ValueError: If fewer than three atom pairs can be matched.
    """
    pairs = residue_pairs(list(mobile), list(reference))
    mob, ref = _atom_pairs(pairs)
    if len(mob) < 3:
        raise ValueError(f"only {len(mob)} matching atoms; cannot superpose")

    rotation, translation = kabsch(mob, ref)
    rmsd_before = _rmsd(mob @ rotation + translation, ref)
    keep = np.ones(len(mob), dtype=bool)
    rmsd = rmsd_before
    done = 0

    for done in range(1, cycles + 1):
        dist = np.linalg.norm(mob @ rotation + translation - ref, axis=1)
        new_keep = keep & (dist <= cutoff * rmsd)
        if new_keep.sum() == keep.sum() or new_keep.sum() < 3:
            done -= 1
            break
        keep = new_keep
        rotation, translation = kabsch(mob[keep], ref[keep])
        rmsd = _rmsd(mob[keep] @ rotation + translation, ref[keep])

    return Fit(rmsd=rmsd, n_atoms=int(keep.sum()), rmsd_before=rmsd_before,
               n_atoms_before=len(mob), n_residues=len(pairs), cycles=done,
               rotation=rotation, translation=translation)


def moved(entity, fit: Fit):
    """Copy of an entity with every atom moved by a fit.

    Args:
        entity: Structure, model or chain (not modified).
        fit: Superposition from superpose.

    Returns:
        The moved copy.
    """
    out = entity.copy()
    out.transform(fit.rotation, fit.translation)
    return out


# --------------------------------------------------------------------------
# Editing
# --------------------------------------------------------------------------


def keep_chains(structure: Structure, chain_ids: Iterable[str],
                protein_only: bool = False) -> Structure:
    """A new structure holding only ``chain_ids``, in that order."""
    wanted = list(dict.fromkeys(chain_ids))
    model = structure[0]
    out_structure = Structure(structure.id)
    out_model = Model(0)
    out_structure.add(out_model)
    for chain_id in wanted:
        if chain_id not in model:
            continue
        copy = model[chain_id].copy()
        if protein_only:
            for residue in [r for r in copy if not is_polymer(r)]:
                copy.detach_child(residue.id)
        out_model.add(copy)
    return out_structure


def without_residues(structure: Structure, keep_names: Iterable[str]) -> Tuple[Structure, int]:
    """Copy of a structure keeping only residues named in keep_names.

    Args:
        structure: Input structure (not modified).
        keep_names: Residue names to keep.

    Returns:
        (new structure, residues removed).
    """
    keep = {n.strip().upper() for n in keep_names}
    out = structure.copy()
    removed = 0
    for chain in out[0]:
        for residue in [r for r in chain if r.get_resname().strip().upper() not in keep]:
            chain.detach_child(residue.id)
            removed += 1
    return out, removed


def cleaned(structure: Structure, allowed_cofactors: Iterable[str] = ()) -> Tuple[Structure, int]:
    """Copy keeping amino acids (modified ones turned into their parents) and allowed cofactors.

    Args:
        structure: Input structure (not modified).
        allowed_cofactors: Extra residue names to keep.

    Returns:
        (new structure, residues removed).
    """
    return without_residues(standardised(structure)[0], set(STANDARD_AA) | set(allowed_cofactors))


def with_chain(structure: Structure, chain: Chain) -> Structure:
    """Copy of a structure with one chain replaced, keeping chain order.

    Args:
        structure: Input structure (not modified).
        chain: Replacement; its id names the chain to replace.

    Returns:
        The new structure.
    """
    out = structure.copy()
    model = out[0]
    position = [c.id for c in model].index(chain.id)
    model.detach_child(chain.id)
    model.insert(position, chain.copy())
    return out


def count_atoms(structure: Structure) -> int:
    """Number of atoms in a structure.

    Args:
        structure: Any structure.

    Returns:
        Atom count.
    """
    return sum(1 for _ in structure.get_atoms())


def renumbered(chain: Chain, old_to_new: Dict[int, int]) -> Tuple[Chain, int]:
    """Copy of a chain renumbered as ``cmd.alter`` + ``cmd.sort`` would.

    Args:
        chain: Input chain (not modified).
        old_to_new: Old residue number to new.

    Returns:
        (new chain, residues left unrenumbered because of a clash).
    """
    chain = chain.copy()
    residues = list(chain)
    for residue in residues:
        chain.detach_child(residue.id)

    taken: Set[Tuple[str, int, str]] = set()
    clashes = 0
    renumbered = []
    for residue in residues:
        het, number, icode = residue.id
        target = (het, old_to_new.get(number, number), icode)
        if target in taken:
            # Fall back to the old number, then to a free insertion code on it.
            # Never drop the residue: losing atoms is worse than an odd label.
            clashes += 1
            target = residue.id
            if target in taken:
                for code in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    if (het, number, code) not in taken:
                        target = (het, number, code)
                        break
        taken.add(target)
        residue.id = target
        renumbered.append(residue)

    renumbered.sort(key=lambda r: (r.id[1], r.id[2], r.id[0]))
    for residue in renumbered:
        chain.add(residue)
    return chain, clashes


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def conect_lines(pdb_lines: Sequence[str], links: Set[Tuple[AtomKey, AtomKey]]) -> List[str]:
    """CONECT records for the atoms in ``pdb_lines``, following PyMOL's rule."""
    rows = []
    for line in pdb_lines:
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        rows.append(dict(
            serial=int(line[6:11]), het=line[:6] == "HETATM", alt=line[16],
            name=line[12:16].strip(), residue=(line[21], line[22:27]),
            element=(line[76:78].strip() or line[12:14].strip()).upper(),
            key=atom_key(line[21], line[22:26], line[26], line[17:20], line[12:16]),
            xyz=(float(line[30:38]), float(line[38:46]), float(line[46:54])),
        ))
    if not rows:
        return []

    xyz = np.array([r["xyz"] for r in rows])
    bonds: Dict[int, Set[int]] = {}

    def bond(a, b):
        bonds.setdefault(a["serial"], set()).add(b["serial"])
        bonds.setdefault(b["serial"], set()).add(a["serial"])

    for i, j in cKDTree(xyz).query_pairs(3.2):
        a, b = rows[i], rows[j]
        if not (a["het"] or b["het"]):
            continue
        if a["alt"] != " " and b["alt"] != " " and a["alt"] != b["alt"]:
            continue
        d = float(np.linalg.norm(xyz[i] - xyz[j]))
        if a["residue"] == b["residue"]:
            limit = (COVALENT_RADII.get(a["element"], DEFAULT_RADIUS)
                     + COVALENT_RADII.get(b["element"], DEFAULT_RADIUS) + BOND_TOLERANCE)
            if d <= limit:
                bond(a, b)
        elif {a["name"], b["name"]} == {"C", "N"} and d <= PEPTIDE_MAX:
            bond(a, b)

    if links:
        by_key: Dict[AtomKey, List[dict]] = {}
        for row in rows:
            by_key.setdefault(row["key"], []).append(row)
        for ka, kb in links:
            for a in by_key.get(ka, []):
                for b in by_key.get(kb, []):
                    if a["het"] or b["het"]:
                        bond(a, b)

    out = []
    for serial in sorted(bonds):
        partners = sorted(bonds[serial])
        for start in range(0, len(partners), 4):
            out.append("CONECT" + f"{serial:>5}"
                       + "".join(f"{p:>5}" for p in partners[start:start + 4]) + "\n")
    return out


