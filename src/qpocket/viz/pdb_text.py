"""PDB text in and out, for handing structures to a browser viewer.

3Dmol.js parses PDB by fixed columns, so every record it receives has to be
column-correct. Nothing is passed through from disk: records are parsed and
re-emitted, which also repairs ligand files written before the column bug in
:func:`qpocket.utils.extract_pdb_block` was fixed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

#: Backbone atom names, the set a cartoon representation needs.
BACKBONE: Tuple[str, ...] = ("N", "CA", "C", "O")

#: Alternate conformations kept when reading a structure. The emitted records
#: carry a blank altLoc, so a viewer can no longer filter them itself and would
#: otherwise draw every A/B pair twice.
KEEP_ALTLOC: Tuple[str, ...] = (" ", "", "A")


class ProteinModels(NamedTuple):
    """Two PDB blocks describing the protein around one pocket.

    Attributes:
        cartoon: Backbone atoms of every residue inside the context radius.
        sidechains: Non-backbone atoms of the residues that line the pocket.
        n_context: Residue count in ``cartoon``.
        n_contact: Residue count contributing to ``sidechains``.
    """

    cartoon: str
    sidechains: str
    n_context: int
    n_contact: int


def parse_atom(line: str) -> Optional[Dict]:
    """Parse one ATOM/HETATM record, tolerating a one-column shift.

    Ligand files written before the ``extract_pdb_block`` fix pad 4-character
    atom names to five, which shifts every later field one column right. Those
    lines fail fixed-column parsing, so they fall back to whitespace tokens.

    Args:
        line: A single PDB record.

    Returns:
        Field dictionary, or ``None`` if the record cannot be read at all.
    """
    try:
        atom = dict(
            name=line[12:16].strip(),
            altloc=line[16:17],
            resname=line[17:20].strip(),
            chain=(line[21:22].strip() or "A"),
            resseq=int(line[22:26]),
            icode=line[26:27],
            x=float(line[30:38]),
            y=float(line[38:46]),
            z=float(line[46:54]),
            element=line[76:78].strip(),
        )
    except ValueError:
        tok = line.split()
        # record serial name resname chain+resseq x y z occupancy b [element]
        if len(tok) < 9:
            return None
        try:
            digits = "".join(ch for ch in tok[4] if ch.isdigit())
            atom = dict(
                name=tok[2],
                altloc=" ",
                resname=tok[3],
                chain=(tok[4][0] if tok[4][:1].isalpha() else "A"),
                resseq=int(digits or 1),
                icode=" ",
                x=float(tok[-6]),
                y=float(tok[-5]),
                z=float(tok[-4]),
                element=("" if _is_number(tok[-1]) else tok[-1]),
            )
        except (ValueError, IndexError):
            return None
    if not atom["element"]:
        atom["element"] = "".join(ch for ch in atom["name"] if ch.isalpha())[:1]
    return atom


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def atom_line(serial: int, atom: Dict, b: float = 0.0, record: str = "ATOM") -> str:
    """Emit one record in the PDB fixed-column layout.

    A 4-character atom name fills columns 13-16 and takes no leading space;
    giving it one shifts the rest of the line and strict parsers reject the
    record. Fields that could outgrow their columns are truncated for the same
    reason: a misaligned line is worse than a shortened value.

    Args:
        serial: Atom serial number, wrapped at 100000.
        atom: Field dictionary as returned by :func:`parse_atom`.
        b: Value for the B-factor column, which viewers can colour by.
        record: ``"ATOM"`` or ``"HETATM"``.

    Returns:
        An 80-character record, without a trailing newline.
    """
    name, element = atom["name"], atom["element"]
    if len(name) >= 4:
        name_field = name[:4]
    elif len(element) == 1:
        name_field = f" {name:<3}"
    else:
        name_field = f"{name:<4}"
    icode = (atom.get("icode") or " ")[:1] or " "
    return (
        f"{record:<6}{serial % 100000:5d} {name_field} {atom['resname'][:3]:>3} "
        f"{atom['chain'][:1]}{atom['resseq'] % 10000:4d}{icode}   "
        f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
        f"  1.00{b:6.2f}          {element[:2].upper():>2}  "
    )


def blocks_to_text(lines: Sequence[str]) -> str:
    """Join emitted records into a PDB block, or return ``""`` if there are none."""
    return ("\n".join(lines) + "\n") if lines else ""


@lru_cache(maxsize=64)
def read_protein_atoms(path: str) -> Tuple[Dict, ...]:
    """Read heavy protein atoms of the first model.

    Waters, hetero records and hydrogens are dropped, and alternate
    conformations are collapsed to one atom per name per residue.

    Args:
        path: Path to a PDB file.

    Returns:
        Atom dictionaries in file order.
    """
    out: List[Dict] = []
    seen = set()
    with open(path) as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            atom = parse_atom(line)
            if atom is None or atom["resname"] == "HOH" or atom["element"] == "H":
                continue
            if atom["altloc"] not in KEEP_ALTLOC:
                continue
            key = (atom["chain"], atom["resseq"], atom["icode"], atom["name"])
            if key in seen:
                continue
            seen.add(key)
            out.append(atom)
    return tuple(out)


@lru_cache(maxsize=128)
def ligand_pdb(path: str) -> str:
    """Read a ligand file and re-emit it with correct columns.

    Args:
        path: Path to an extracted-ligand PDB file.

    Returns:
        PDB text, or ``""`` if the file holds no readable atoms.
    """
    lines: List[str] = []
    with open(path) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atom = parse_atom(line)
                if atom is not None:
                    lines.append(atom_line(len(lines) + 1, atom, record="HETATM"))
    return blocks_to_text(lines)


def pseudo_atom_pdb(
    coords: Iterable[Sequence[float]],
    b: Optional[Sequence[float]] = None,
    resname: str = "PKT",
    chain: str = "P",
    element: str = "X",
) -> str:
    """Emit a point cloud as HETATM pseudo-atoms.

    Each point gets its own residue number: parsers that collapse duplicate
    (residue, atom name) keys would otherwise keep only the first point of the
    whole cloud.

    Args:
        coords: Sequence of xyz triples.
        b: Optional per-point B-factor values, for colouring by property.
        resname: Residue name for every pseudo-atom.
        chain: Chain identifier for every pseudo-atom.
        element: Element symbol for every pseudo-atom.

    Returns:
        PDB text, or ``""`` for an empty cloud.
    """
    pts = np.asarray(list(coords), dtype=float).reshape(-1, 3)
    values = np.zeros(len(pts)) if b is None else np.asarray(b, dtype=float)
    if len(values) != len(pts):
        raise ValueError(f"got {len(values)} b-factors for {len(pts)} points")
    lines = [
        atom_line(
            i + 1,
            dict(
                name=element,
                resname=resname,
                chain=chain,
                resseq=i + 1,
                icode=" ",
                x=float(pts[i, 0]),
                y=float(pts[i, 1]),
                z=float(pts[i, 2]),
                element=element,
            ),
            b=float(values[i]),
            record="HETATM",
        )
        for i in range(len(pts))
    ]
    return blocks_to_text(lines)


def bbox_corners(points: np.ndarray, pad: float = 2.0) -> np.ndarray:
    """Eight corners of the padded axis-aligned box around ``points``.

    Framing every panel on this box, rather than on its own contents, is what
    makes side-by-side panels the same scale.

    Args:
        points: ``(n, 3)`` coordinates.
        pad: Padding added on every side, in Ångström.

    Returns:
        ``(8, 3)`` corner coordinates.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    return np.array(
        [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=float,
    )


def pocket_context_models(
    pdb_path: str,
    cloud_xyz: np.ndarray,
    context_radius: float,
    contact_radius: float = 4.0,
) -> ProteinModels:
    """Split a protein into a local cartoon plus the side chains lining a pocket.

    Distance is measured per residue as its closest approach to any pocket
    point, so a residue counts as pocket-lining when any of its atoms reaches
    the cloud.

    Args:
        pdb_path: Path to the aligned structure.
        cloud_xyz: ``(n, 3)`` pocket point coordinates.
        context_radius: Residues within this distance appear as cartoon.
        contact_radius: Residues within this distance also get side-chain sticks.

    Returns:
        The two PDB blocks and their residue counts.
    """
    atoms = read_protein_atoms(pdb_path)
    cloud = np.asarray(cloud_xyz, dtype=float).reshape(-1, 3)
    if not atoms or not len(cloud):
        return ProteinModels("", "", 0, 0)

    xyz = np.array([[a["x"], a["y"], a["z"]] for a in atoms])
    distance, _ = cKDTree(cloud).query(xyz, k=1)

    residue_of = [(a["chain"], a["resseq"], a["icode"]) for a in atoms]
    nearest: Dict[Tuple, float] = {}
    for key, dist in zip(residue_of, distance):
        if dist < nearest.get(key, np.inf):
            nearest[key] = dist

    context = {k for k, v in nearest.items() if v <= context_radius}
    contact = {k for k, v in nearest.items() if v <= contact_radius}

    cartoon: List[str] = []
    sidechains: List[str] = []
    for atom, key in zip(atoms, residue_of):
        if key not in context:
            continue
        if atom["name"] in BACKBONE:
            cartoon.append(atom_line(len(cartoon) + 1, atom))
        elif key in contact:
            # CA already sits in the cartoon model, so sticks start at CB.
            sidechains.append(atom_line(len(sidechains) + 1, atom))
    return ProteinModels(
        blocks_to_text(cartoon),
        blocks_to_text(sidechains),
        len(context),
        len(contact),
    )


def residue_model(pdb_path: str, residues: Sequence[str]) -> str:
    """PDB block for named residues, whatever their distance to a pocket.

    Residue keys are the ``RESNAME_NUMBER_CHAIN`` form the interaction tables use,
    e.g. ``"GLN_754_A"``. A residue is drawn regardless of the viewer's context
    radius, so a highlight asked for is never silently absent from a panel.

    Args:
        pdb_path: Path to the aligned structure.
        residues: Residue keys to include.

    Returns:
        PDB text for the matching atoms, or ``""`` when none match.
    """
    wanted = set()
    for key in residues:
        parts = str(key).split("_")
        if len(parts) >= 3:
            try:
                wanted.add((parts[-1], int(parts[-2])))
            except ValueError:
                continue
    if not wanted:
        return ""
    lines: List[str] = []
    for atom in read_protein_atoms(pdb_path):
        if (atom["chain"], atom["resseq"]) in wanted:
            lines.append(atom_line(len(lines) + 1, atom))
    return blocks_to_text(lines)
