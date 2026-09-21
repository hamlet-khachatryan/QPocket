"""Structure files: read PDB/mmCIF, covalent links, write PDB with CONECT."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from Bio.PDB import MMCIFParser, PDBIO, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Structure import Structure

from qpocket.core.structure import AtomKey, atom_key as _key, conect_lines as _conect_lines


def load(path: Path) -> Structure:
    """Read a PDB or mmCIF file, keeping only its first model."""
    path = Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    for model in list(structure)[1:]:
        structure.detach_child(model.id)
    return structure


def read_links(path: Path) -> Set[Tuple[AtomKey, AtomKey]]:
    """Covalent links annotated in a structure file."""
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
    for line in path.read_text(encoding="utf-8").splitlines():
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
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = [l for l in lines if not l.startswith("END")]
    path.write_text("".join(lines + _conect_lines(lines, links or set()) + ["END\n"]), encoding="utf-8")
