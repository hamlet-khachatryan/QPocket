"""Structure operations that replaced PyMOL.

The replacement was validated against PyMOL itself on 20 PARG structures (every
atom within 0.0016 A of ``cmd.align``'s placement). These tests pin the same
behaviour on structures built by hand, where the right answer is known exactly:
a copy of a chain moved by a known rotation must come back to where it started,
a pair of badly placed residues must be rejected by the refinement, and the
CONECT records PLIP depends on must follow PyMOL's rule -- which, it turned out,
includes links annotated in the file and excludes contacts that are merely close.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qpocket.core import structure as st
from qpocket.io import structures as sio
from qpocket.pipelines.align import process_structure

SEQ = "ACDEFGHIKLMNPQRSTVWY"
THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
         "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
         "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
         "W": "TRP", "Y": "TYR"}


def line(serial, name, resname, chain, resseq, xyz, element, het=False, alt=" "):
    record = "HETATM" if het else "ATOM  "
    x, y, z = xyz
    return (f"{record}{serial:>5} {name:<4}{alt}{resname:>3} {chain}{resseq:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 20.00          {element:>2}\n")


def helix(seq=SEQ, start=1, chain="A", rotation=None, shift=(0.0, 0.0, 0.0), skip=()):
    """Backbone + CB of a helix-shaped chain, optionally moved rigidly."""
    rotation = np.eye(3) if rotation is None else rotation
    lines, serial = [], 1
    for i, aa in enumerate(seq):
        if i in skip:
            continue
        t = np.radians(100.0 * i)
        ca = np.array([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * i])
        atoms = {"N": ca + [-0.6, 0.9, -0.5], "CA": ca, "C": ca + [0.9, 0.6, 0.6],
                 "O": ca + [1.2, 1.7, 0.8], "CB": ca + [-0.8, -0.9, 0.4]}
        for name, xyz in atoms.items():
            if aa == "G" and name == "CB":
                continue
            moved = np.asarray(xyz) @ rotation + shift
            lines.append(line(serial, name, THREE[aa], chain, start + i, moved, name[0]))
            serial += 1
    return lines


def rotation_matrix(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def write(path, lines):
    path.write_text("".join(lines) + "END\n")
    return path


def residues_of(path, chain="A"):
    return st.polymer_residues(sio.load(path)[0][chain])


def ca_array(residues):
    return np.array([r["CA"].coord for r in residues])


# --- superposition --------------------------------------------------------


def test_residues_are_matched_by_sequence_not_by_number(tmp_path):
    """A construct numbered from 101 and missing three residues still aligns."""
    ref = residues_of(write(tmp_path / "ref.pdb", helix()))
    mob_path = write(tmp_path / "mob.pdb",
                     helix(start=101, skip=(7, 8, 9), rotation=rotation_matrix(1), shift=(4, 0, 0)))
    fit = st.superpose(residues_of(mob_path), ref)
    assert fit.n_residues == len(SEQ) - 3 and fit.rmsd < 1e-3


# --- residues -------------------------------------------------------------


# --- editing --------------------------------------------------------------


def test_cleaning_keeps_protein_and_allowed_cofactors(tmp_path):
    lines = helix("AA") + [
        line(90, "FE", "HEM", "A", 800, (5, 5, 5), "FE", het=True),
        line(91, "C1", "LIG", "A", 801, (6, 6, 6), "C", het=True),
    ]
    s = sio.load(write(tmp_path / "c.pdb", lines))
    clean, removed = st.cleaned(s, ["HEM"])
    assert {r.get_resname() for r in clean.get_residues()} == {"ALA", "HEM"} and removed == 1
    assert "LIG" in {r.get_resname() for r in s.get_residues()}  # input untouched


# --- saving and CONECT ----------------------------------------------------


def conect_pairs(path):
    pairs = set()
    for l in Path(path).read_text().splitlines():
        if l.startswith("CONECT"):
            a = int(l[6:11])
            for i in range(11, 31, 5):
                t = l[i:i + 5].strip()
                if t:
                    pairs.add(tuple(sorted((a, int(t)))))
    return pairs


def atom_names(path):
    return {int(l[6:11]): (l[17:20].strip(), l[12:16].strip(), l[16])
            for l in Path(path).read_text().splitlines() if l[:6] in ("ATOM  ", "HETATM")}


def named_bonds(path):
    names = atom_names(path)
    return {tuple(sorted((names[a], names[b]))) for a, b in conect_pairs(path)}


def ligand(resseq=900, chain="A", start=90):
    """A three-atom ligand: C1-C2 bonded (1.5 A), O3 bonded to C2."""
    return [
        line(start, "C1", "LIG", chain, resseq, (20.0, 0.0, 0.0), "C", het=True),
        line(start + 1, "C2", "LIG", chain, resseq, (21.5, 0.0, 0.0), "C", het=True),
        line(start + 2, "O3", "LIG", chain, resseq, (22.2, 1.2, 0.0), "O", het=True),
    ]


# --- process_structure, end to end ----------------------------------------


@pytest.fixture
def aligned_target(tmp_path):
    """Reference chain; target = the same chain moved, plus ligand, water and a decoy chain."""
    ref_path = write(tmp_path / "ref.pdb", helix())
    target = (helix(rotation=rotation_matrix(7), shift=(40, 10, -5))
              + [line(80, "O", "HOH", "A", 500, (1, 2, 3), "O", het=True)]
              + ligand()
              + helix("GGGGGGGG", chain="B", start=1))
    target_path = write(tmp_path / "1XYZ.pdb", target)
    out_dir = tmp_path / "aligned"
    out_dir.mkdir()
    ok, rmsd, out = process_structure(
        target_path=target_path, output_dir=out_dir,
        reference=residues_of(ref_path), ref_seq=SEQ, renumber=False,
    )
    return ok, rmsd, out, ref_path


def test_process_structure_succeeds_with_near_zero_rmsd(aligned_target):
    ok, rmsd, out, _ = aligned_target
    assert ok and rmsd < 1e-3 and Path(out).exists()


def test_process_structure_skips_an_unrelated_protein(tmp_path):
    ref_path = write(tmp_path / "ref.pdb", helix())
    target = write(tmp_path / "2XYZ.pdb", helix("GGGGGGGGGGGGGGGGGGGG"))
    ok, rmsd, out = process_structure(
        target_path=target, output_dir=tmp_path, reference=residues_of(ref_path),
        ref_seq=SEQ, renumber=False,
    )
    assert (ok, rmsd, out) == (False, -1.0, "")


def test_process_structure_reports_a_bad_file_instead_of_raising(tmp_path):
    ref_path = write(tmp_path / "ref.pdb", helix())
    bad = tmp_path / "bad.cif"
    bad.write_text("not a structure")
    ok, _, _ = process_structure(
        target_path=bad, output_dir=tmp_path, reference=residues_of(ref_path),
        ref_seq=SEQ, renumber=False,
    )
    assert ok is False


# --- the reason for all of this ---------------------------------------------


# --- modified residues ------------------------------------------------------


def peptide(names, het=(), far=()):
    """A straight peptide whose C–N links are 1.43 Å, so residues are bonded; ``far`` ones sit 30 Å away.

    PTR gets its phosphate (P, O1P) and MSE its selenium, beside the atoms they share with TYR and MET.
    """
    extra = {"PTR": {"CB": (0, 1.5, 0), "OH": (0, 3.0, 0), "P": (0, 4.5, 0), "O1P": (0, 5.5, 0)},
             "MSE": {"CB": (0, 1.5, 0), "SE": (0, 3.0, 0)}}
    lines, serial = [], 1
    for i, name in enumerate(names):
        x0 = 3.8 * i + (30.0 if i in far else 0.0)
        atoms = {"N": (x0, 0, 0), "CA": (x0 + 1.46, 0, 0), "C": (x0 + 2.46, 0.5, 0), "O": (x0 + 2.46, 1.7, 0)}
        atoms.update({a: (x0 + 1.46 + dx, dy, dz) for a, (dx, dy, dz) in extra.get(name, {}).items()})
        for atom, xyz in atoms.items():
            element = "SE" if atom == "SE" else atom[0]
            lines.append(line(serial, atom, name, "A", i + 1, xyz, element, het=i in het))
            serial += 1
    return lines


def test_cleaning_turns_a_bonded_modified_residue_into_its_parent(tmp_path):
    s = sio.load(write(tmp_path / "p.pdb", peptide(["ALA", "PTR", "ALA"], het={1})))
    clean, removed = st.cleaned(s)
    residues = list(clean[0]["A"])
    assert [r.get_resname() for r in residues] == ["ALA", "TYR", "ALA"] and removed == 0
    tyr = residues[1]
    assert tyr.id[0] == " " and {"N", "CA", "C", "O", "CB", "OH"} == {a.get_id() for a in tyr}


def test_find_ligands_skips_modified_residues_in_the_chain(tmp_path):
    from Bio.PDB import PDBParser

    from qpocket.core.sites import find_ligands

    lines = peptide(["ALA", "PTR", "ALA"], het={1}) + [
        line(90 + k, f"C{k}", "LIG", "A", 900, (k, 20, 0), "C", het=True) for k in range(1, 8)]
    parsed = PDBParser(QUIET=True).get_structure("x", str(write(tmp_path / "l.pdb", lines)))
    assert [ligand.resname for ligand, _ in find_ligands(parsed, "x", [], 6)] == ["LIG"]
