"""Structure operations, in Biopython, standing in for PyMOL.

The pipeline used PyMOL for four things -- loading, superposition, cleaning and
saving -- and PyMOL is the one dependency with no pip wheel for older Linux
(RHEL 8, glibc 2.28), which is most cluster nodes. Everything it did is done here
with Biopython, which the pipeline already depends on, and numpy.

Two behaviours are reproduced deliberately, because results depend on them:

**Superposition** follows ``cmd.align``: a BLOSUM62 sequence alignment decides
which residues correspond, every atom with a matching name in a matched pair
is fitted, and the fit is refined for up to five cycles, each dropping pairs
further than twice the current RMSD. The whole mobile structure then moves, not
only the fitted chain, so auxiliary chains land in the reference frame too.

**CONECT records** follow what PyMOL wrote, because PLIP -- which reads the
aligned files -- uses them to perceive ligand bonds. Dropping them changed PLIP's
hydrogen-bond calls on 5 of 16 PARG structures. PyMOL's rule, recovered from the
files it wrote: bonds within a residue by covalent radii (+0.45 A), never between
alternate locations; peptide C-N links between residues; and the covalent links
annotated in the input (``_struct_conn`` in mmCIF, ``LINK`` or inter-residue
``CONECT`` in PDB). Only pairs involving a HETATM are written, as PyMOL did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
from Bio.PDB.Chain import Chain
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from scipy.spatial import cKDTree

from qpocket_pipeline.global_vars import STANDARD_AA, THREE_TO_ONE_MAPPING

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
        rotation: 3x3 matrix, applied as ``coord @ rotation + translation``
            (Biopython's ``transform`` convention).
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


def load(path: Path) -> Structure:
    """Read a PDB or mmCIF file, keeping only its first model.

    PyMOL aligned and saved state 1 only; later NMR models were never used.
    """
    path = Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    for model in list(structure)[1:]:
        structure.detach_child(model.id)
    return structure


def remove_water(structure: Structure) -> int:
    """Delete solvent residues in place. Returns how many were removed."""
    removed = 0
    for chain in structure[0]:
        for residue in [r for r in chain if r.get_resname().strip() in WATER]:
            chain.detach_child(residue.id)
            removed += 1
    return removed


def read_links(path: Path) -> Set[Tuple[AtomKey, AtomKey]]:
    """Covalent links annotated in a structure file.

    mmCIF: ``_struct_conn`` rows of type ``covale`` or ``disulf``. PDB: ``LINK``
    records, plus any ``CONECT`` pair joining two different residues (which is
    how a file this module wrote carries its links forward to the next stage).
    """
    path = Path(path)
    links: Set[Tuple[AtomKey, AtomKey]] = set()

    if path.suffix.lower() == ".cif":
        try:
            data = MMCIF2Dict(str(path))
        except Exception:
            return links
        types = data.get("_struct_conn.conn_type_id")
        if not types:
            return links
        if isinstance(types, str):
            types = [types]

        def column(name: str) -> List[str]:
            value = data.get(f"_struct_conn.{name}", ["?"] * len(types))
            return [value] if isinstance(value, str) else value

        cols = {k: column(k) for k in (
            "ptnr1_auth_asym_id", "ptnr1_auth_seq_id", "pdbx_ptnr1_PDB_ins_code",
            "ptnr1_label_comp_id", "ptnr1_label_atom_id",
            "ptnr2_auth_asym_id", "ptnr2_auth_seq_id", "pdbx_ptnr2_PDB_ins_code",
            "ptnr2_label_comp_id", "ptnr2_label_atom_id",
        )}
        for i, kind in enumerate(types):
            if kind not in ("covale", "disulf"):
                continue
            try:
                a = _key(cols["ptnr1_auth_asym_id"][i], cols["ptnr1_auth_seq_id"][i],
                         cols["pdbx_ptnr1_PDB_ins_code"][i], cols["ptnr1_label_comp_id"][i],
                         cols["ptnr1_label_atom_id"][i])
                b = _key(cols["ptnr2_auth_asym_id"][i], cols["ptnr2_auth_seq_id"][i],
                         cols["pdbx_ptnr2_PDB_ins_code"][i], cols["ptnr2_label_comp_id"][i],
                         cols["ptnr2_label_atom_id"][i])
            except (ValueError, IndexError):
                continue
            links.add(tuple(sorted((a, b))))
        return links

    serial_to_key: Dict[int, AtomKey] = {}
    conect: List[Tuple[int, int]] = []
    for line in path.read_text().splitlines():
        record = line[:6]
        if record == "LINK  ":
            try:
                a = _key(line[21], line[22:26], line[26], line[17:20], line[12:16])
                b = _key(line[51], line[52:56], line[56], line[47:50], line[42:46])
            except ValueError:
                continue
            links.add(tuple(sorted((a, b))))
        elif record in ("ATOM  ", "HETATM"):
            try:
                serial_to_key[int(line[6:11])] = _key(
                    line[21], line[22:26], line[26], line[17:20], line[12:16])
            except ValueError:
                continue
        elif record == "CONECT":
            try:
                a = int(line[6:11])
            except ValueError:
                continue
            for start in range(11, 31, 5):
                token = line[start:start + 5].strip()
                if token.isdigit():
                    conect.append((a, int(token)))

    for a, b in conect:
        ka, kb = serial_to_key.get(a), serial_to_key.get(b)
        if ka and kb and ka[:4] != kb[:4]:  # different residues
            links.add(tuple(sorted((ka, kb))))
    return links


def _key(chain: str, resseq, icode: str, resname: str, atom: str) -> AtomKey:
    icode = "" if icode in (None, "?", ".", " ") else str(icode).strip()
    return (str(chain).strip(), int(str(resseq).strip()), icode,
            str(resname).strip(), str(atom).strip())


# --------------------------------------------------------------------------
# Residues and sequences
# --------------------------------------------------------------------------


def is_polymer(residue: Residue) -> bool:
    """An amino acid in the chain: it has the backbone N, CA and C.

    Recognises modified residues (MSE, CME, CSO...) as polymer, which a residue
    name list would miss, and never mistakes a ligand for one.
    """
    return all(name in residue for name in ("N", "CA", "C"))


def polymer_residues(chain: Chain) -> List[Residue]:
    return [r for r in chain if is_polymer(r)]


def one_letter(residue: Residue) -> str:
    return THREE_TO_ONE_MAPPING.get(residue.get_resname().strip(), "X")


def chain_sequence(chain: Chain) -> str:
    """One-letter sequence of a chain's amino acids, as ``cmd.get_fastastr``."""
    return "".join(one_letter(r) for r in polymer_residues(chain))


def ca_coords(chain: Chain) -> np.ndarray:
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
    """Atoms present under the same name in both residues of each pair.

    Hydrogens included, as ``cmd.align`` includes them: on entries deposited with
    riding hydrogens they are half the atoms, and leaving them out moved the RMSD
    by ~0.01 A against PyMOL's.
    """
    mob, ref = [], []
    for res_m, res_r in pairs:
        for atom in res_m:
            if atom.get_id() not in res_r:
                continue
            mob.append(atom.coord)
            ref.append(res_r[atom.get_id()].coord)
    return np.array(mob, dtype=float).reshape(-1, 3), np.array(ref, dtype=float).reshape(-1, 3)


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotation and translation taking ``mobile`` onto ``reference``.

    Returned in Biopython's convention: ``mobile @ rotation + translation``.
    """
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

    Nothing is moved; apply ``fit.rotation`` / ``fit.translation`` with
    :func:`apply` to whatever should follow the fit.

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


def apply(entity, fit: Fit) -> None:
    """Move every atom of ``entity`` by a fit, in place."""
    entity.transform(fit.rotation, fit.translation)


# --------------------------------------------------------------------------
# Editing
# --------------------------------------------------------------------------


def keep_chains(structure: Structure, chain_ids: Iterable[str],
                protein_only: bool = False) -> Structure:
    """A new structure holding only ``chain_ids``, in that order.

    ``protein_only`` keeps just amino-acid residues, as ``polymer.protein``.
    """
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


def remove_residues(structure: Structure, keep_names: Iterable[str]) -> int:
    """Delete every residue whose name is not in ``keep_names``. Returns the count."""
    keep = {n.strip().upper() for n in keep_names}
    removed = 0
    for chain in structure[0]:
        for residue in [r for r in chain if r.get_resname().strip().upper() not in keep]:
            chain.detach_child(residue.id)
            removed += 1
    return removed


def clean(structure: Structure, allowed_cofactors: Iterable[str] = ()) -> int:
    """Keep standard amino acids and allowed cofactors; returns residues removed."""
    return remove_residues(structure, set(STANDARD_AA) | set(allowed_cofactors))


def count_atoms(structure: Structure) -> int:
    return sum(1 for _ in structure.get_atoms())


def renumber(chain: Chain, old_to_new: Dict[int, int]) -> int:
    """Renumber residues of a chain in place, as ``cmd.alter`` + ``cmd.sort``.

    Every residue whose number is a key -- whatever its insertion code or HETATM
    flag, as PyMOL's ``resi`` selection matched -- takes the new number. Where two
    residues would end up with the same identifier, which Biopython cannot hold,
    the later one keeps its old number, or failing that its old number
    with a free insertion code, and is counted in the return value. No residue
    is ever dropped.

    Returns:
        Residues that could not be renumbered because of a clash.
    """
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
    return clashes


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def _conect_lines(pdb_lines: Sequence[str], links: Set[Tuple[AtomKey, AtomKey]]) -> List[str]:
    """CONECT records for the atoms in ``pdb_lines``, following PyMOL's rule."""
    rows = []
    for line in pdb_lines:
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        rows.append(dict(
            serial=int(line[6:11]), het=line[:6] == "HETATM", alt=line[16],
            name=line[12:16].strip(), residue=(line[21], line[22:27]),
            element=(line[76:78].strip() or line[12:14].strip()).upper(),
            key=_key(line[21], line[22:26], line[26], line[17:20], line[12:16]),
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


def save(structure: Structure, path: Path,
         links: Optional[Set[Tuple[AtomKey, AtomKey]]] = None) -> None:
    """Write a PDB file with PyMOL-equivalent CONECT records.

    Raises:
        ValueError: If a chain identifier is longer than PDB format allows.
    """
    for chain in structure[0]:
        if len(chain.id) > 1:
            raise ValueError(
                f"chain {chain.id!r} does not fit the one-character PDB chain column"
            )
    path = Path(path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), write_end=False)
    lines = path.read_text().splitlines(keepends=True)
    lines = [l for l in lines if not l.startswith("END")]
    path.write_text("".join(lines + _conect_lines(lines, links or set()) + ["END\n"]))
