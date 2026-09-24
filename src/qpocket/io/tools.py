"""External tools: pdb2pqr, APBS and the files they exchange."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from subprocess import PIPE, Popen

import numpy as np
from Bio.PDB import PDBParser
from eliot import log_message, start_action

from cavitomix.cavfind import CavFind

from qpocket.constants import FP_DTYPE, ION_PARAMS, atomic_hydrophobicity
from qpocket.core import features as ft
from qpocket.types import PQRAtoms, StructureData



class ToolNotFoundError(RuntimeError):
    """An external program Q-Pocket needs is not installed."""


_HINTS = {
    "apbs": "install APBS (github.com/Electrostatics/apbs/releases) and put it on PATH or set QPOCKET_APBS; "
            "on Linux x86_64 and macOS `uv sync` installs it",
    "pdb2pqr": "`uv sync` installs it; activate the environment",
    "obabel": "optional: install Open Babel separately for the SDF to PDBQT fallback",
    "vina": "set docking.engine_path or put vina on PATH",
}


def executable(name: str, env_var: Optional[str] = None) -> str:
    """Path of an external program: env override, then PATH (resolves .exe on Windows).

    Args:
        name: Program name without extension.
        env_var: Environment variable that overrides the lookup.

    Returns:
        Path to the program.

    Raises:
        ToolNotFoundError: If it cannot be found.
    """
    override = os.environ.get(env_var) if env_var else None
    if override:
        if Path(override).exists():
            return override
        raise ToolNotFoundError(f"{env_var}={override} does not exist")
    found = shutil.which(name)
    if found:
        return found
    raise ToolNotFoundError(f"{name} not found: {_HINTS.get(name, 'install it and put it on PATH')}")


def apbs_command() -> Tuple[str, str]:
    """Where APBS comes from on this machine.

    Returns:
        (source, detail): "env" and the QPOCKET_APBS path, "wheel" and the apbs-binary
        version, or "path" and the apbs on PATH.

    Raises:
        ToolNotFoundError: If there is no APBS at all.
    """
    if os.environ.get("QPOCKET_APBS"):
        return "env", executable("apbs", "QPOCKET_APBS")
    try:
        import apbs_binary  # no Windows wheel

        return "wheel", f"apbs-binary {getattr(apbs_binary, '__version__', '')}".strip()
    except ImportError:
        return "path", executable("apbs")


def apbs_process(args: List[str], **popen_kwargs) -> Popen:
    """Start APBS with arguments, from whichever source apbs_command finds.

    Args:
        args: APBS arguments (usually the input file).
        **popen_kwargs: Passed to Popen (stdout, stderr, cwd).

    Returns:
        The running process.
    """
    source, detail = apbs_command()
    if source == "wheel":
        from apbs_binary import popen_apbs

        return popen_apbs(args, **popen_kwargs)
    return Popen([detail] + list(args), **popen_kwargs)


def repair_pqr_with_missing_ions(pdb_path: Path, pqr_path: Path) -> None:
    """Append ions that pdb2pqr dropped to its PQR, with PARSE-style charges and radii."""
    pdb_ions = []
    with pdb_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("HETATM"):
                # Fixed PDB columns: atom name 13-16, residue name 18-20.
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip().upper()

                clean_name = "".join([c for c in atom_name if c.isalpha()]).upper()

                # An ion is its own residue (ZN in ZN). Matching the atom name alone
                # turned a cofactor's alpha carbon (CA in CME) into a calcium ion.
                if clean_name in ION_PARAMS and res_name == clean_name:
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
    with pqr_path.open("r", encoding="utf-8") as f:
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
    with pqr_path.open("a", encoding="utf-8") as f:
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
        cmd = [executable("pdb2pqr"), "--ff=PARSE", "--ffout=AMBER", "--whitespace", "--noopt",
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


def protonate_for_interactions(pdb_path: Path, out_pqr: Path) -> Path:
    """Protonate a protein with the hydrogen-bond network optimised.

    Args:
        pdb_path: Cleaned protein PDB.
        out_pqr: Where to write the PQR.

    Returns:
        out_pqr.

    Raises:
        RuntimeError: If pdb2pqr writes nothing.
    """
    # Unlike protonate(), this omits --noopt: without the optimiser pdb2pqr leaves
    # Ser, Thr and Tyr hydroxyl hydrogens in an arbitrary rotamer, which makes
    # those donors unreliable. The feature PQR keeps --noopt so its numbers stand.
    with start_action(action_type="run_pdb2pqr_optimised", pdb=str(pdb_path)):
        out_pqr = Path(out_pqr).resolve()
        cmd = [executable("pdb2pqr"), "--ff=PARSE", "--ffout=AMBER", "--whitespace",
               str(Path(pdb_path).resolve()), str(out_pqr)]
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
    with dx_path.open("r", encoding="utf-8") as f:
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
        with pqr.open(encoding="utf-8") as f:
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
            with apbs_in.open("w", encoding="utf-8") as f:
                f.write(ft.apbs_input(pqr, base, *box))

            try:
                p = apbs_process([str(apbs_in)], stdout=PIPE, stderr=PIPE, cwd=str(temp_dir))
            except ToolNotFoundError as e:
                log_message(message_type="apbs_binary_fail", stderr=str(e))
                return np.zeros(len(point_cloud), dtype=FP_DTYPE)
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
    with open(pdb_path, "r", encoding="utf-8") as f:
        cf = CavFind(name=name, obj_name=name)
        cf.struct_from_pdb(f.read())
        # CavFind keeps the points on a private attribute; read it in one place.
        return [cavity.__dict__["coords"] for cavity in cf.run().cavities]

