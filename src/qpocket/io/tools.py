"""External tools: pdb2pqr, APBS and the files they exchange."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from subprocess import PIPE, Popen

import numpy as np
from Bio.PDB import PDBParser
from eliot import log_message, start_action
from plip.structure.preparation import PDBComplex

from cavitomix.cavfind import CavFind

from qpocket.constants import FP_DTYPE, ION_PARAMS, atomic_hydrophobicity
from qpocket.core import features as ft
from qpocket.types import PQRAtoms, StructureData

try:
    from apbs_binary import popen_apbs
except ImportError:
    raise ImportError("pip install apbs-binary")


def repair_pqr_with_missing_ions(pdb_path: Path, pqr_path: Path) -> None:
    """Scans the original PDB for ions."""
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


def protonate(pdb_path: Path, out_pqr: Path) -> Path:
    """Protonate a protein once: pdb2pqr (PARSE charges and radii, AMBER names), then repair ions.

    Args:
        pdb_path: Cleaned protein PDB.
        out_pqr: Where to write the PQR.

    Returns:
        out_pqr.

    Raises:
        RuntimeError: If pdb2pqr writes nothing.
    """
    with start_action(action_type="run_pdb2pqr", pdb=str(pdb_path)):
        out_pqr = Path(out_pqr).resolve()
        cmd = ["pdb2pqr", "--ff=PARSE", "--ffout=AMBER", "--whitespace", "--noopt",
               str(Path(pdb_path).resolve()), str(out_pqr)]
        # cwd: pdb2pqr drops working files beside the process, not its output.
        with tempfile.TemporaryDirectory() as scratch:
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=scratch)
            out, err = p.communicate()
        if not out_pqr.exists() or out_pqr.stat().st_size == 0:
            log_message(message_type="error", error=err.decode(), stdout=out.decode())
            raise RuntimeError("PDB2PQR failed or produced empty file.")
        repair_pqr_with_missing_ions(Path(pdb_path), out_pqr)
        return out_pqr


def read_dx_grid(dx_path: Path):
    """Read an APBS OpenDX file.

    Args:
        dx_path: Path to the .dx file.

    Returns:
        (grid, (x_axis, y_axis, z_axis)).

    Raises:
        RuntimeError: On a malformed file.
    """
    with dx_path.open("r") as f:
        return ft.parse_dx(f.read())


def apbs_potential(pqr_path: Path, point_cloud: np.ndarray) -> np.ndarray:
    """Electrostatic potential (kT/e) at pocket points, from APBS on a PQR.

    Args:
        pqr_path: Protonated protein from protonate.
        point_cloud: (N, 3) pocket points.

    Returns:
        (N,) potential; zeros if APBS fails.
    """
    with start_action(action_type="calculate_apbs_potential"):
        pqr = Path(pqr_path).resolve()
        if not pqr.exists() or pqr.stat().st_size == 0:
            log_message(message_type="pdb2pqr_fail", error=f"missing PQR {pqr}")
            return np.zeros(len(point_cloud), dtype=FP_DTYPE)

        prot_coords = []
        with pqr.open() as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        parts = line.split()
                        prot_coords.append([float(parts[-5]), float(parts[-4]), float(parts[-3])])
                    except (ValueError, IndexError):
                        pass
        if not prot_coords:
            return np.zeros(len(point_cloud), dtype=FP_DTYPE)

        box = ft.apbs_grid_box(np.array(prot_coords), point_cloud)
        if box is None:
            log_message(
                message_type="error",
                stage="calculate_apbs_potential",
                error="NaN in grid center/dims",
            )
            return np.zeros(len(point_cloud), dtype=FP_DTYPE)

        # cwd=temp_dir: APBS always writes an io.mc beside the process, which
        # concurrent workers would otherwise share.
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            base = temp_dir / "mol"
            apbs_in = base.with_suffix(".in")
            with apbs_in.open("w") as f:
                f.write(ft.apbs_input(pqr, base, *box))

            p = popen_apbs([str(apbs_in)], stdout=PIPE, stderr=PIPE, cwd=str(temp_dir))
            out, err = p.communicate()

            dx_file = base.with_suffix(".dx")
            if not dx_file.exists():
                dx_file = base.with_suffix(".pot.dx")
            if not dx_file.exists():
                log_message(message_type="apbs_binary_fail", stderr=err.decode())
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)

            try:
                grid, axes = read_dx_grid(dx_file)
                return ft.interpolate_grid(grid, axes, point_cloud)
            except Exception as e:
                log_message(message_type="interpolation_error", error=str(e))
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)


def structure_data(pdb_path: Path, pqr: PQRAtoms) -> StructureData:
    """Charges, hydrophobicity and pharmacophore sites of a protein.

    Args:
        pdb_path: Cleaned protein PDB (for donor, acceptor and ring geometry).
        pqr: Its protonated atoms from read_pqr.

    Returns:
        The protein's featurisation inputs.
    """
    with start_action(action_type="extract_structure_data", pdb=str(pdb_path)):
        if len(pqr.coords) == 0:
            log_message(message_type="warning", msg="PQR parsed 0 atoms! Check PDB format.")
        hydrophobicity = np.array(
            [atomic_hydrophobicity(an, rn) for an, rn in zip(pqr.atom_names, pqr.res_names)],
            dtype=FP_DTYPE,
        )
        structure = PDBParser(QUIET=True).get_structure("prot", str(pdb_path))
        donors, acceptors, aromatics = ft.structure_sites(structure)
        return StructureData(pqr.coords, pqr.charges, hydrophobicity, donors, acceptors, aromatics)


def run_cavfind(pdb_path: Path, name: str) -> List[np.ndarray]:
    """Cavities detected by CavitOmiX in a PDB file.

    Args:
        pdb_path: Protein PDB.
        name: Label passed to CavFind.

    Returns:
        One (N, 3) point cloud per cavity, in detection order.
    """
    with open(pdb_path, "r") as f:
        cf = CavFind(name=name, obj_name=name)
        cf.struct_from_pdb(f.read())
        # CavFind keeps the points on a private attribute; read it in one place.
        return [cavity.__dict__["coords"] for cavity in cf.run().cavities]


def plip_interactions(pdb_path: Path, ligand_name: str) -> List[str]:
    """Analyses protein–ligand interactions with PLIP.

    Args:
        pdb_path: Path to a PDB structure file.
        ligand_name: Residue name of the ligand to analyse (case-insensitive).

    Returns:
        List of interaction strings formatted as ``"RESNAME_RESNUM|InteractionType"``.
    """
    interactions: List[str] = []

    try:
        protlig = PDBComplex()
        protlig.load_pdb(str(pdb_path))
        protlig.analyze()

        for k, binding_site in protlig.interaction_sets.items():
            # PLIP key format: "RESNAME:CHAIN:RESNUM:" — match on residue name only
            # to capture all chains that carry the same ligand.
            if ligand_name.upper() != k.upper().split(":")[0]:
                continue

            bind_site_interactions = [
                i for i in binding_site.all_itypes if "waterbridge" not in str(i)
            ]

            for inter in bind_site_interactions:
                inter_type = str(inter).split("(")[0]

                # Interaction strings include the chain ID so that residues on
                # different chains are unambiguously distinguished (critical for
                # interface pockets with two chains carrying the same residue numbers).
                res_key = (
                    f"{inter.restype.upper()}_{inter.resnr}_{inter.reschain}"
                )

                if inter_type == "pistack":
                    interactions.append(
                        f"{res_key}|Pi-stacking ({inter.type})"
                    )
                elif inter_type == "hbond":
                    itype = "H-bond Donor" if inter.protisdon else "H-bond Acceptor"
                    interactions.append(f"{res_key}|{itype}")
                elif inter_type == "pication":
                    itype = "Pi-Cation (Cat)" if inter.protcharged else "Pi-Cation (Pi)"
                    interactions.append(f"{res_key}|{itype}")
                elif inter_type == "saltbridge":
                    itype = "Saltbridge (+)" if inter.protispos else "Saltbridge (-)"
                    interactions.append(f"{res_key}|{itype}")
                elif inter_type == "metal_complex":
                    interactions.append(f"{res_key}|Metal Complex")
                elif inter_type == "halogenbond":
                    interactions.append(f"{res_key}|Halogen Bond")
                elif inter_type == "hydroph_interaction":
                    interactions.append(f"{res_key}|Hydrophobic")

    except Exception as e:
        log_message(
            message_type="error",
            stage="plip_analysis",
            pdb_path=str(pdb_path),
            error=str(e),
        )
        return ["analysis_error"]

    return interactions if interactions else ["no_ligand_found"]
