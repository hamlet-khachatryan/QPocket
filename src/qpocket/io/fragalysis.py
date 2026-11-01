"""Turn a Fragalysis download into the inputs of a QPocket fragment-screen run.

Each aligned observation becomes the chain its ligand binds, the ligand renamed and defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem

from qpocket.io import complex_prep as cp

#: metadata.csv columns copied into inputs.csv, under these names, when present.
METADATA_COLUMNS = {
    "Experiment code": "experiment_code",
    "Compound code": "compound_code",
    "CanonSites alias": "canon_site",
    "ConformerSites alias": "conformer_site",
    "CrystalformSites alias": "crystalform_site",
}


@dataclass(frozen=True)
class Options:
    """How the prepared run is configured.

    Attributes:
        path_prefix: Where the output directory will live on the machine that runs QPocket.
        workdir: QPocket ``paths.workdir``; ``<prefix>/runs`` when None.
        run_name: ``reference.pdb_id``, naming the run; the target's name when None.
        chain: Chain ID every prepared structure is written under.
        workers: ``compute.workers``.
        cofactors: ``cleaning.allowed_cofactors`` ('|'-separated), or None.
    """

    path_prefix: Optional[Path] = None
    workdir: Optional[str] = None
    run_name: Optional[str] = None
    chain: str = "A"
    workers: int = 0
    cofactors: Optional[str] = None


@dataclass(frozen=True)
class Summary:
    """What a preparation produced."""

    prepared: int
    compounds: int
    reference: Optional[str]
    reference_resolution: Optional[float]
    skipped: Dict[str, int]
    config: Optional[Path]


def target_name(download: Path) -> str:
    """The target's name, from ``<target>_combined.sdf`` or else the folder name."""
    combined = sorted(download.glob("*_combined.sdf"))
    return combined[0].name[: -len("_combined.sdf")] if combined else download.name


def observations(download: Path) -> List[Tuple[str, Path, Path]]:
    """Every observation folder with its complex and ligand files, sorted by code.

    Args:
        download: The unpacked download.

    Returns:
        (code, complex PDB, ligand SDF); a missing file is still listed, so it can be reported.

    Raises:
        FileNotFoundError: If the download has no ``aligned_files`` folder.
    """
    aligned = download / "aligned_files"
    if not aligned.is_dir():
        raise FileNotFoundError(f"{download} has no aligned_files folder; is it a Fragalysis download?")
    found = []
    for folder in sorted(p for p in aligned.iterdir() if p.is_dir()):
        code = folder.name
        sdf = folder / f"{code}_ligand.sdf"
        found.append((code, folder / f"{code}.pdb", sdf if sdf.exists() else folder / f"{code}.sdf"))
    return found


def _metadata(download: Path) -> Dict[str, Dict[str, object]]:
    path = download / "metadata.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype=str).fillna("")
    if "Code" not in table.columns:
        return {}
    columns = [c for c in METADATA_COLUMNS if c in table.columns]
    return {row["Code"]: {METADATA_COLUMNS[c]: row[c] for c in columns} for _, row in table.iterrows()}


def config_yaml(csv_path: Path, reference: Path, workdir: str, run_name: str, opts: Options) -> str:
    """The fragment-screen configuration around the prepared structures."""
    return "\n".join([
        "mode: fragment_screen",
        "fragment_screen:",
        "  input_type: csv",
        f"  csv_path: {csv_path}",
        f"  reference_path: {reference}",
        "reference:",
        f"  pdb_id: {run_name}",
        f"  chain_id: {opts.chain}",
        "cleaning:",
        f'  allowed_cofactors: "{opts.cofactors}"' if opts.cofactors else "  allowed_cofactors: null",
        "paths:",
        f"  workdir: {workdir}",
        "compute:",
        f"  workers: {opts.workers}",
        ""])


def prepare(download: Path, out: Path, opts: Options = Options()) -> Summary:
    """Write structures/, ligand_definitions/, inputs.csv, report.csv, qpocket.yaml and run.sh.

    Args:
        download: The unpacked download.
        out: Output directory, created when missing.
        opts: How the run is configured.

    Returns:
        Counts, the reference and the config path (None when nothing could be prepared).
    """
    download, out = download.resolve(), out.resolve()
    prefix = opts.path_prefix or out
    metadata = _metadata(download)
    report: List[Dict[str, object]] = []
    anchored = []
    for code, pdb, sdf in observations(download):
        entry = {"code": code, "status": "skipped", "reason": "", "source_chain": "",
                 "anchor": "", "moved": "", "renumbered": "", "dropped": ""}
        report.append(entry)
        if not pdb.exists() or not sdf.exists():
            entry["reason"] = "missing_file"
            continue
        mol = Chem.MolFromMolFile(str(sdf), removeHs=True)
        if mol is None:
            entry["reason"] = "unreadable_sdf"
            continue
        text = pdb.read_text(encoding="utf-8", errors="replace")
        atoms = cp.read_atoms(text)
        anchor = cp.match_anchor(atoms, mol)
        if anchor is None:
            entry["reason"] = "no_anchor"
            continue
        anchored.append((code, entry, text, atoms, mol, anchor, Chem.MolToSmiles(mol)))

    codes = cp.assign_codes(a[-1] for a in anchored)
    bonds_of_code: Dict[str, frozenset] = {}
    (out / "structures").mkdir(parents=True, exist_ok=True)
    (out / "ligand_definitions").mkdir(exist_ok=True)
    inputs: List[Dict[str, object]] = []
    best: Optional[Tuple[float, str]] = None
    for code, entry, text, atoms, mol, anchor, smiles in anchored:
        resname = codes[smiles]
        bonds = cp.named_bonds(mol, anchor)
        if resname not in bonds_of_code:
            bonds_of_code[resname] = bonds
            (out / "ligand_definitions" / f"{resname}.cif").write_text(
                cp.component_cif(resname, mol, anchor), encoding="utf-8")
        elif bonds != bonds_of_code[resname]:
            # The definition is keyed by atom name: another naming would get the wrong bonds.
            entry["reason"] = "bonds_differ"
            continue
        prepared = cp.prepare_complex(atoms, anchor, resname, cp.header(text), out_chain=opts.chain)
        (out / "structures" / f"{code}.pdb").write_text("\n".join(prepared.lines) + "\n", encoding="utf-8")
        entry.update(status="ok", source_chain=prepared.kept_chain, anchor=cp.label(anchor.key),
                     moved=";".join(prepared.moved), renumbered=";".join(prepared.renumbered),
                     dropped=";".join(prepared.dropped))
        res = cp.resolution(text)
        inputs.append({"pdb_path": str(prefix / "structures" / f"{code}.pdb"), "code": code,
                       "residue_code": resname, "smiles": smiles, "source_chain": prepared.kept_chain,
                       "resolution": res, **metadata.get(code, {})})
        if res is not None and (best is None or (res, code) < best):
            best = (res, code)

    pd.DataFrame(report).to_csv(out / "report.csv", index=False)
    skipped = pd.DataFrame(report).query("status == 'skipped'")["reason"].value_counts().to_dict()
    if not inputs:
        return Summary(0, 0, None, None, skipped, None)
    pd.DataFrame(inputs).to_csv(out / "inputs.csv", index=False)
    reference = best[1] if best else str(inputs[0]["code"])
    workdir = opts.workdir or str(prefix / "runs")
    config = out / "qpocket.yaml"
    config.write_text(config_yaml(prefix / "inputs.csv", prefix / "structures" / f"{reference}.pdb",
                                  workdir, opts.run_name or target_name(download), opts), encoding="utf-8")
    run_sh = out / "run.sh"
    run_sh.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "# The ligands' bonds live in ligand_definitions; every qpocket command on this run needs it.\n"
        f"export QPOCKET_CCD_CACHE={prefix / 'ligand_definitions'}\n"
        f'qpocket run --config {prefix / "qpocket.yaml"} "$@"\n', encoding="utf-8")
    run_sh.chmod(0o755)
    return Summary(len(inputs), len({i["residue_code"] for i in inputs}), reference,
                   best[0] if best else None, skipped, config)
