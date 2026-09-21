import tempfile
import numpy as np
from pathlib import Path
from typing import Tuple, List
from subprocess import Popen, PIPE
from Bio.PDB import PDBParser, is_aa
from eliot import start_action, log_message
from qpocket_pipeline.global_vars import (
    FP_DTYPE,
    GET_ATOMIC_HYDROPHOBICITY,
    DONOR_RESIDUES,
    ACCEPTOR_RESIDUES,
    AROMATIC_RESIDUES,
    ION_PARAMS
)

def repair_pqr_with_missing_ions(pdb_path: Path, pqr_path: Path) -> None:
    """
    Scans the original PDB for ions. If pdb2pqr dropped them, this function
    appends them to the PQR file with standard PARSE parameters.
    """
    pdb_ions = []
    with pdb_path.open("r") as f:
        for line in f:
            if line.startswith("HETATM"):
                parts = line.split()

                atom_name = parts[2]
                res_name = parts[3]

                clean_name = "".join([c for c in atom_name if c.isalpha()]).upper()

                if clean_name in ION_PARAMS:
                    # extract coordinates (PDB columns are fixed width, but split works for most clean PDBs)
                    # standard PDB: X(30-38), Y(38-46), Z(46-54)
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        pdb_ions.append(
                            {
                                "name": clean_name,
                                "coords": (x, y, z),
                                "line": line,
                            }
                        )
                    except ValueError:
                        continue

    if not pdb_ions:
        return

    # read existing PQR to see what survived
    existing_coords = []
    pqr_lines = []
    with pqr_path.open("r") as f:
        for line in f:
            pqr_lines.append(line)
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                try:
                    x, y, z = float(parts[-5]), float(parts[-4]), float(parts[-3])
                    existing_coords.append((x, y, z))
                except:
                    pass

    existing_coords = np.array(existing_coords)

    # missing ions
    injected_count = 0
    with pqr_path.open("a") as f:
        for ion in pdb_ions:
            if len(existing_coords) > 0:
                dists = np.linalg.norm(
                    existing_coords - np.array(ion["coords"]), axis=1
                )
                if np.min(dists) < 0.1:
                    continue

            charge, radius = ION_PARAMS[ion["name"]]
            x, y, z = ion["coords"]

            # formatted PQR line (whitespace separated is fine for APBS)
            # HETATM  1234  MG   MG     1      10.500  20.100  30.200  2.0000 1.4000
            new_line = f"HETATM 9999 {ion['name']} ION     1    {x:8.3f} {y:8.3f} {z:8.3f} {charge:7.4f} {radius:7.4f}\n"
            f.write(new_line)
            injected_count += 1

    if injected_count > 0:
        log_message(
            message_type="pqr_repair",
            injected_ions=injected_count,
        )


def run_pdb2pqr(pdb_path: Path, output_pqr: Path) -> None:
    """Runs PDB2PQR via command line."""
    with start_action(action_type="run_pdb2pqr", pdb=str(pdb_path)):
        cmd = [
            "pdb2pqr",
            "--ff=PARSE",
            "--whitespace",
            "--noopt",
            str(pdb_path),
            str(output_pqr),
        ]

        p = Popen(cmd, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()

        if not output_pqr.exists() or output_pqr.stat().st_size == 0:
            log_message(message_type="error", error=err.decode(), stdout=out.decode())
            raise RuntimeError("PDB2PQR failed or produced empty file.")


def parse_pqr_data(
    pqr_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Robust PQR parser handling variable whitespace"""
    coords = []
    charges = []
    atom_names = []
    res_names = []

    with pqr_path.open("r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            fields = line.strip().split()
            if len(fields) < 10:
                continue

            try:
                x, y, z = float(fields[-5]), float(fields[-4]), float(fields[-3])
                q = float(fields[-2])
                atom_name = fields[2]
                res_name = fields[3]

                coords.append([x, y, z])
                charges.append(q)
                atom_names.append(atom_name)
                res_names.append(res_name)
            except (ValueError, IndexError):
                continue

    coords_arr = np.array(coords, dtype=FP_DTYPE)
    if coords_arr.ndim == 1:
        coords_arr = coords_arr.reshape(-1, 3)

    return (coords_arr, np.array(charges, dtype=FP_DTYPE), atom_names, res_names)


def get_structure_data(
    pdb_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with start_action(action_type="extract_structure_data", pdb=str(pdb_path)):
        with tempfile.NamedTemporaryFile(suffix=".pqr", delete=False) as tmp:
            pqr_path = Path(tmp.name)

        try:
            run_pdb2pqr(pdb_path, pqr_path)
            all_coords, all_chg, atom_names, res_names = parse_pqr_data(pqr_path)

            if len(all_coords) == 0:
                log_message(
                    message_type="warning", msg="PQR parsed 0 atoms! Check PDB format."
                )

            hp_list = [
                GET_ATOMIC_HYDROPHOBICITY(an, rn)
                for an, rn in zip(atom_names, res_names)
            ]
            all_hp = np.array(hp_list, dtype=FP_DTYPE)

        finally:
            if pqr_path.exists():
                pqr_path.unlink()

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("prot", str(pdb_path))

        geo_donors, geo_acceptors, geo_arom = [], [], []

        for model in structure:
            for chain in model:
                for res in chain:
                    if not is_aa(res):
                        continue
                    resname = res.get_resname()

                    if resname in AROMATIC_RESIDUES:
                        ring_atoms = []
                        if resname in ["PHE", "TYR"]:
                            ring_atoms = ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
                        elif resname == "TRP":
                            ring_atoms = ["CG", "CD1", "CD2", "NE1", "CE2", "CE3"]
                        elif resname in ["HIS", "HID", "HIE", "HIP"]:
                            ring_atoms = ["CG", "ND1", "CD2", "CE1", "NE2"]

                        pts = [res[at].get_coord() for at in ring_atoms if at in res]
                        if pts:
                            geo_arom.append(np.mean(pts, axis=0))

                    for atom in res:
                        elem = atom.element
                        if resname in DONOR_RESIDUES and (
                            elem == "N" or (elem == "O" and resname in ["SER", "THR"])
                        ):
                            geo_donors.append(atom.get_coord())
                        if resname in ACCEPTOR_RESIDUES and (
                            elem == "O"
                            or (elem == "N" and resname in ["HIS", "HID", "HIE", "HIP"])
                        ):
                            geo_acceptors.append(atom.get_coord())

        def safe_arr(lst):
            arr = np.array(lst, dtype=FP_DTYPE)
            return arr.reshape(-1, 3) if arr.ndim == 1 else arr

        return (
            all_coords,
            all_chg,
            all_hp,
            safe_arr(geo_donors),
            safe_arr(geo_acceptors),
            safe_arr(geo_arom),
        )
