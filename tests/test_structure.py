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
from qpocket.core.alignment import process_structure

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


def test_a_rigidly_moved_copy_is_put_back_exactly(tmp_path):
    ref = residues_of(write(tmp_path / "ref.pdb", helix()))
    mob_path = write(tmp_path / "mob.pdb", helix(rotation=rotation_matrix(), shift=(30, -12, 7)))
    s = sio.load(mob_path)
    fit = st.superpose(st.polymer_residues(s[0]["A"]), ref)
    moved = st.moved(s, fit)
    assert np.allclose(ca_array(st.polymer_residues(moved[0]["A"])), ca_array(ref), atol=1e-3)
    assert not np.allclose(ca_array(st.polymer_residues(s[0]["A"])), ca_array(ref), atol=1e-3)


def test_a_perfect_fit_reports_zero_rmsd(tmp_path):
    ref = residues_of(write(tmp_path / "ref.pdb", helix()))
    mob = residues_of(write(tmp_path / "mob.pdb", helix(rotation=rotation_matrix(3), shift=(5, 5, 5))))
    assert st.superpose(mob, ref).rmsd < 1e-3


def test_superpose_never_moves_anything_by_itself(tmp_path):
    ref = residues_of(write(tmp_path / "ref.pdb", helix()))
    mob = residues_of(write(tmp_path / "mob.pdb", helix(shift=(9, 9, 9))))
    before = ca_array(mob).copy()
    st.superpose(mob, ref)
    assert np.array_equal(ca_array(mob), before)


def test_misplaced_residues_are_rejected_by_refinement(tmp_path):
    """cmd.align drops pairs beyond 2x RMSD; two residues shoved 6 A must go."""
    ref = residues_of(write(tmp_path / "ref.pdb", helix()))
    lines = helix()
    moved = []
    for l in lines:
        if int(l[22:26]) in (5, 12):
            xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]) + [6, 0, 0]
            l = l[:30] + f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}" + l[54:]
        moved.append(l)
    fit = st.superpose(residues_of(write(tmp_path / "mob.pdb", moved)), ref)
    assert fit.n_atoms < fit.n_atoms_before and fit.rmsd < fit.rmsd_before and fit.rmsd < 0.1


def test_residues_are_matched_by_sequence_not_by_number(tmp_path):
    """A construct numbered from 101 and missing three residues still aligns."""
    ref = residues_of(write(tmp_path / "ref.pdb", helix()))
    mob_path = write(tmp_path / "mob.pdb",
                     helix(start=101, skip=(7, 8, 9), rotation=rotation_matrix(1), shift=(4, 0, 0)))
    fit = st.superpose(residues_of(mob_path), ref)
    assert fit.n_residues == len(SEQ) - 3 and fit.rmsd < 1e-3


def test_too_few_matching_atoms_is_an_error(tmp_path):
    ref = residues_of(write(tmp_path / "ref.pdb", helix("AC")))
    mob = residues_of(write(tmp_path / "mob.pdb", helix("WY")))
    with pytest.raises(ValueError, match="cannot superpose"):
        st.superpose(mob[:0], ref)


# --- residues -------------------------------------------------------------


def test_a_modified_residue_with_a_backbone_counts_as_polymer(tmp_path):
    lines = helix("AAA")
    lines[5] = lines[5].replace("ALA", "MSE").replace("ATOM  ", "HETATM")
    lines = [l.replace(" ALA A   2", " MSE A   2").replace("ATOM  ", "HETATM") if " A   2 " in l else l
             for l in lines]
    chain = sio.load(write(tmp_path / "m.pdb", lines))[0]["A"]
    assert [r.get_resname() for r in st.polymer_residues(chain)] == ["ALA", "MSE", "ALA"]


def test_a_ligand_is_not_polymer(tmp_path):
    lines = helix("AA") + [line(99, "C1", "LIG", "A", 900, (1, 1, 1), "C", het=True)]
    chain = sio.load(write(tmp_path / "l.pdb", lines))[0]["A"]
    assert len(st.polymer_residues(chain)) == 2


def test_water_is_removed(tmp_path):
    lines = helix("AA") + [line(99, "O", "HOH", "A", 500, (9, 9, 9), "O", het=True)]
    s = sio.load(write(tmp_path / "w.pdb", lines))
    dry, removed = st.without_water(s)
    assert removed == 1 and "HOH" not in {r.get_resname() for r in dry.get_residues()}
    assert "HOH" in {r.get_resname() for r in s.get_residues()}  # input untouched


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


def test_keep_chains_drops_other_chains(tmp_path):
    s = sio.load(write(tmp_path / "two.pdb", helix("AAA", chain="A") + helix("GGG", chain="B")))
    assert [c.id for c in st.keep_chains(s, ["B"])[0]] == ["B"]


def test_protein_only_drops_ligands(tmp_path):
    lines = helix("AA") + [line(99, "C1", "LIG", "A", 900, (1, 1, 1), "C", het=True)]
    s = st.keep_chains(sio.load(write(tmp_path / "p.pdb", lines)), ["A"], protein_only=True)
    assert {r.get_resname() for r in s.get_residues()} == {"ALA"}


def test_renumbering_moves_and_sorts_residues(tmp_path):
    chain = sio.load(write(tmp_path / "r.pdb", helix("ACD")))[0]["A"]
    new, _ = st.renumbered(chain, {1: 101, 2: 102, 3: 103})
    assert [r.id[1] for r in new] == [101, 102, 103]
    assert [r.id[1] for r in chain] == [1, 2, 3]  # input untouched


def test_a_renumbering_clash_keeps_both_residues(tmp_path):
    """Two residues landing on one number cannot both be held; neither is lost."""
    chain = sio.load(write(tmp_path / "r.pdb", helix("ACD")))[0]["A"]
    new, clashes = st.renumbered(chain, {1: 3})
    assert clashes == 1 and len(list(new)) == 3


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


def test_ligand_bonds_are_written_as_conect(tmp_path):
    out = tmp_path / "out.pdb"
    sio.save(sio.load(write(tmp_path / "in.pdb", helix("AA") + ligand())), out)
    assert named_bonds(out) == {
        (("LIG", "C1", " "), ("LIG", "C2", " ")), (("LIG", "C2", " "), ("LIG", "O3", " "))}


def test_protein_alone_gets_no_conect(tmp_path):
    """PyMOL wrote CONECT only for bonds touching a HETATM."""
    out = tmp_path / "out.pdb"
    sio.save(sio.load(write(tmp_path / "in.pdb", helix())), out)
    assert conect_pairs(out) == set()


def test_alternate_locations_are_never_bonded(tmp_path):
    lines = [
        line(1, "C1", "LIG", "A", 900, (0.0, 0.0, 0.0), "C", het=True, alt="A"),
        line(2, "C1", "LIG", "A", 900, (0.2, 0.0, 0.0), "C", het=True, alt="B"),
    ]
    out = tmp_path / "out.pdb"
    sio.save(sio.load(write(tmp_path / "in.pdb", lines)), out)
    assert conect_pairs(out) == set()


def test_a_close_but_unannotated_contact_is_not_bonded(tmp_path):
    """PARG 4A0D: a DTT sulfur 2.04 A from a cysteine SG, which PyMOL left unbonded."""
    lines = [line(1, "SG", "CYS", "A", 10, (0, 0, 0), "S"),
             line(2, "S4", "DTV", "A", 900, (2.04, 0, 0), "S", het=True)]
    out = tmp_path / "out.pdb"
    sio.save(sio.load(write(tmp_path / "in.pdb", lines)), out)
    assert conect_pairs(out) == set()


def test_an_annotated_link_is_bonded(tmp_path):
    """PARG 4B1H: the same geometry, but listed in the file as covalent."""
    lines = [line(1, "SG", "CYS", "A", 10, (0, 0, 0), "S"),
             line(2, "S1", "DTV", "A", 900, (2.04, 0, 0), "S", het=True)]
    src = write(tmp_path / "in.pdb", lines)
    link = ("LINK         SG  CYS A  10                 S1  DTV A 900     1555   1555  2.04  \n")
    src.write_text(link + src.read_text())
    out = tmp_path / "out.pdb"
    sio.save(sio.load(src), out, sio.read_links(src))
    assert named_bonds(out) == {(("CYS", "SG", " "), ("DTV", "S1", " "))}


def test_links_survive_a_save_and_reload(tmp_path):
    """Cleaning reads the aligned file; its covalent links must carry through."""
    lines = [line(1, "SG", "CYS", "A", 10, (0, 0, 0), "S"),
             line(2, "S1", "DTV", "A", 900, (2.04, 0, 0), "S", het=True)]
    src = write(tmp_path / "in.pdb", lines)
    src.write_text("LINK         SG  CYS A  10                 S1  DTV A 900     1555   1555  2.04  \n"
                   + src.read_text())
    first = tmp_path / "first.pdb"
    sio.save(sio.load(src), first, sio.read_links(src))
    second = tmp_path / "second.pdb"
    sio.save(sio.load(first), second, sio.read_links(first))
    assert named_bonds(second) == named_bonds(first)


def test_mmcif_struct_conn_links_are_read(tmp_path):
    cif = tmp_path / "x.cif"
    cif.write_text(
        "data_X\nloop_\n_struct_conn.conn_type_id\n"
        "_struct_conn.ptnr1_auth_asym_id\n_struct_conn.ptnr1_auth_seq_id\n"
        "_struct_conn.pdbx_ptnr1_PDB_ins_code\n_struct_conn.ptnr1_label_comp_id\n"
        "_struct_conn.ptnr1_label_atom_id\n_struct_conn.ptnr2_auth_asym_id\n"
        "_struct_conn.ptnr2_auth_seq_id\n_struct_conn.pdbx_ptnr2_PDB_ins_code\n"
        "_struct_conn.ptnr2_label_comp_id\n_struct_conn.ptnr2_label_atom_id\n"
        "covale A 603 ? CYS SG A 1603 ? BME S2\n"
        "hydrog A 1 ? ALA N A 2 ? ALA O\n"
    )
    assert sio.read_links(cif) == {(("A", 603, "", "CYS", "SG"), ("A", 1603, "", "BME", "S2"))}


def test_a_peptide_link_into_a_modified_residue_is_bonded(tmp_path):
    lines = [line(1, "C", "ALA", "A", 1, (0, 0, 0), "C"),
             line(2, "N", "MSE", "A", 2, (1.33, 0, 0), "N", het=True)]
    out = tmp_path / "out.pdb"
    sio.save(sio.load(write(tmp_path / "in.pdb", lines)), out)
    assert named_bonds(out) == {(("ALA", "C", " "), ("MSE", "N", " "))}


def test_a_long_chain_id_is_refused_rather_than_written_wrongly(tmp_path):
    s = sio.load(write(tmp_path / "in.pdb", helix("AA")))
    s[0]["A"].id = "AB"
    with pytest.raises(ValueError, match="chain"):
        sio.save(s, tmp_path / "out.pdb")


def test_saved_files_load_back_unchanged(tmp_path):
    src = write(tmp_path / "in.pdb", helix() + ligand())
    out = tmp_path / "out.pdb"
    sio.save(sio.load(src), out)
    a = np.array([x.coord for x in sio.load(src).get_atoms()])
    b = np.array([x.coord for x in sio.load(out).get_atoms()])
    assert np.allclose(a, b, atol=1e-3)


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


def test_process_structure_puts_the_chain_in_the_reference_frame(aligned_target):
    _, _, out, ref_path = aligned_target
    assert np.allclose(ca_array(residues_of(Path(out))), ca_array(residues_of(ref_path)), atol=2e-3)


def test_process_structure_keeps_only_the_matching_chain(aligned_target):
    _, _, out, _ = aligned_target
    assert [c.id for c in sio.load(Path(out))[0]] == ["A"]


def test_process_structure_keeps_the_ligand_and_drops_water(aligned_target):
    _, _, out, _ = aligned_target
    names = {r.get_resname() for r in sio.load(Path(out)).get_residues()}
    assert "LIG" in names and "HOH" not in names


def test_process_structure_writes_ligand_conect_for_plip(aligned_target):
    _, _, out, _ = aligned_target
    assert conect_pairs(out)


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


def test_no_module_imports_pymol():
    """PyMOL has no pip wheel for RHEL 8. Every module must import without it.

    Run in a fresh interpreter with the import blocked: in a development
    environment that happens to have PyMOL installed, a stray `import pymol`
    would otherwise pass here and only fail on the cluster.
    """
    import pkgutil
    import subprocess
    import sys

    import qpocket

    modules = sorted(
        m.name for m in pkgutil.walk_packages(qpocket.__path__, "qpocket.")
    )
    script = (
        "import sys, importlib\n"
        "sys.modules['pymol'] = None\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
    )
    src = str(Path(qpocket.__file__).resolve().parents[1])
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": src})
    assert done.returncode == 0, done.stderr[-800:]


def test_with_chain_replaces_in_place_of_order(tmp_path):
    s = sio.load(write(tmp_path / "two.pdb", helix("AAA", chain="A") + helix("GGG", chain="B")))
    new_a, _ = st.renumbered(s[0]["A"], {1: 50})
    out = st.with_chain(s, new_a)
    assert [c.id for c in out[0]] == ["A", "B"] and out[0]["A"].child_list[0].id[1] == 2
    assert s[0]["A"].child_list[0].id[1] == 1  # input untouched
