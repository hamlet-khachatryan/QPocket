"""One structure from input file to its ligands, contacts and cavities: a pipe of small steps."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from eliot import log_message, start_action

from qpocket.config import QPocketConfig, plugin_params
from qpocket.core import pocket
from qpocket.core.features import DEFAULT_PROVIDERS, pocket_features
from qpocket.core.ports import CHARGES, CONTACTS, FEATURES
from qpocket.core.result import Failure, pipe
from qpocket.core.sites import find_ligands
from qpocket.core.types import RawStructure, Reference, Structure
from qpocket.io import layout, tables, tools
from qpocket.io.tools import run_cavfind
from qpocket.pipelines import align as alignment
from qpocket.pipelines import clean as cleaning
from qpocket.pipelines.reference import reference_residues
from qpocket.types import StructureData


@dataclass(frozen=True)
class StructureParams:
    """Everything the chain reads from the configuration.

    Attributes:
        allowed_cofactors: "|"-separated residue names kept by cleaning and skipped as ligands.
        min_heavy_atoms: Smaller HET groups are not ligands.
        contacts: CONTACTS plug-in names.
        contact_params: Parameters per contact plug-in.
        min_auxiliary_identity: Interface mode alignment threshold.
        max_auxiliary_centroid_deviation: Interface mode alignment threshold.
        require_all_chains: Interface mode alignment rule.
    """

    allowed_cofactors: str
    min_heavy_atoms: int
    contacts: Tuple[str, ...]
    contact_params: Mapping[str, Any]
    min_auxiliary_identity: float
    max_auxiliary_centroid_deviation: float
    require_all_chains: bool

    def _replace(self, **changes) -> "StructureParams":
        import dataclasses

        return dataclasses.replace(self, **changes)


def structure_params(cfg: QPocketConfig) -> StructureParams:
    """The chain's parameters, read from the configuration.

    Args:
        cfg: QPocket configuration.

    Returns:
        Frozen parameters.
    """
    prolif_params = CONTACTS.get("prolif").params(
        enabled=tuple(cfg.interactions.enabled), parameters=cfg.interactions.parameters)
    return StructureParams(
        allowed_cofactors=cfg.cleaning.allowed_cofactors or "",
        min_heavy_atoms=cfg.sites.min_heavy_atoms,
        contacts=tuple(cfg.contacts.use),
        contact_params={name: prolif_params if name == "prolif"
                        else plugin_params(CONTACTS, name, cfg.contacts.options)
                        for name in cfg.contacts.use},
        min_auxiliary_identity=cfg.alignment.min_auxiliary_identity,
        max_auxiliary_centroid_deviation=cfg.alignment.max_auxiliary_centroid_deviation,
        require_all_chains=cfg.alignment.require_all_chains,
    )


# ── files ────────────────────────────────────────────────────────────────────

def clean_file(record: Structure, ws: layout.Workspace, allowed_cofactors: str) -> Optional[str]:
    """Clean one aligned structure.

    Args:
        record: Carries aligned_file.
        ws: The run.
        allowed_cofactors: Residues to keep.

    Returns:
        The cleaned file, or None when cleaning failed.
    """
    aligned = record.aligned_file
    if not aligned:
        return None
    output_path = ws.cleaned / (Path(aligned).stem.replace("_aligned", "_clean") + ".pdb")
    if cleaning.clean_structure(input_pdb_path=aligned, output_pdb_path=str(output_path),
                               allowed_cofactors=allowed_cofactors):
        log_message(message_type="cleaning_ok", pdb_id=record.pdb_id, output=str(output_path))
        return str(output_path)
    log_message(message_type="cleaning_failed", pdb_id=record.pdb_id)
    return None


def featurize(csv_path: Path, pdb_path: Path, features: Sequence[str] = DEFAULT_PROVIDERS,
              charges: str = "pdb2pqr", options: Optional[Mapping[str, Any]] = None,
              charge_params: Any = None) -> None:
    """Write featurized.csv beside a pocket's point cloud.

    Args:
        csv_path: Pocket point cloud CSV (x, y, z).
        pdb_path: Cleaned protein PDB.
        features: FEATURES plug-in names.
        charges: CHARGES plug-in that provides atoms and charges.
        options: Settings per feature plug-in.
        charge_params: The charge model's parameters; its defaults when None.
    """
    with start_action(action_type="process_point_cloud"):
        if not csv_path.exists():
            log_message(message_type="error", error="CSV missing", path=str(csv_path))
            return

        try:
            table, points = tables.read_point_table(csv_path)
        except Exception as e:
            log_message(message_type="error", error=f"CSV read error: {e}")
            return

        chosen = FEATURES.resolve(features)
        needs_pqr = any("pqr" in provider.requires for provider in chosen)
        wants_apbs = any(provider.name == "apbs" for provider in chosen)

        pqr_path = csv_path.with_name("protein.pqr")
        data = StructureData.empty()
        apbs = np.zeros(len(points))
        try:
            if needs_pqr:
                plugin = CHARGES.get(charges)
                data = plugin.fn(Path(pdb_path), pqr_path,
                                 charge_params if charge_params is not None else plugin.params())
        except Exception as e:
            log_message(message_type="fatal_error", error=str(e))
            return

        out_path = csv_path.with_name("featurized.csv")
        if wants_apbs:
            log_message(message_type="status", msg="Running APBS")
            apbs = tools.apbs_potential(pqr_path, points)
        log_message(message_type="status", msg="Running the feature features",
                    features=list(features))
        tables.write_featurized(out_path, table,
                                pocket_features(points, data, apbs, features, options))
        log_message(message_type="success", output=str(out_path))


# ── steps ────────────────────────────────────────────────────────────────────

def align(ref: Reference, p: StructureParams, ws: layout.Workspace, raw: RawStructure):
    """Superpose a structure onto the reference; the reference itself is already done.

    Returns:
        Its record, or Failure("align").
    """
    if raw.pdb_id == ref.pdb_id:
        return ref.record
    ctx = ref.alignment
    reference = reference_residues(ctx.reference_path, ctx.reference_chain)
    if ctx.interface_mode:
        success, rmsd, aligned_file, aux_mapping = alignment.process_structure_interface(
            target_path=raw.path, output_dir=ws.aligned, reference=reference,
            ref_seq=ctx.reference_sequence, ref_aux_seqs=ctx.aux_sequences,
            ref_aux_centroids=ctx.aux_centroids, uniprot_id=ctx.uniprot_id,
            auxiliary_uniprot_ids=ctx.aux_uniprots, renumber=ctx.renumber,
            min_auxiliary_identity=p.min_auxiliary_identity,
            max_auxiliary_centroid_deviation=p.max_auxiliary_centroid_deviation,
            require_all_chains=p.require_all_chains)
    else:
        success, rmsd, aligned_file = alignment.process_structure(
            target_path=raw.path, output_dir=ws.aligned, reference=reference,
            ref_seq=ctx.reference_sequence, uniprot_id=ctx.uniprot_id, renumber=ctx.renumber)
        aux_mapping = None
    if not success:
        log_message(message_type="alignment_failed", pdb_id=raw.pdb_id)
        return Failure(raw.pdb_id, "align", "alignment failed")
    log_message(message_type="alignment_ok", pdb_id=raw.pdb_id, rmsd=rmsd)
    return Structure(raw.pdb_id, str(raw.path), rmsd, aligned_file,
                     str(aux_mapping) if ctx.interface_mode else None)


def clean(p: StructureParams, ws: layout.Workspace, record: Structure):
    """Strip to protein and allowed cofactors.

    Returns:
        The record with cleaned_file, or Failure("clean").
    """
    if record.cleaned_file:
        return record
    cleaned = clean_file(record, ws, p.allowed_cofactors)
    if cleaned is None:
        return Failure(record.pdb_id, "clean", "cleaning failed")
    return record._replace(cleaned_file=cleaned)


def find_ligands_step(p: StructureParams, ws: layout.Workspace, record: Structure) -> Structure:
    """Extract every ligand of the aligned structure, with its SMILES.

    Returns:
        The record with ligands and BOUND_STATE, UNBOUND_STATE or FAILED.
    """
    from qpocket.io import interactions

    try:
        parsed = PDBParser(QUIET=True).get_structure(record.pdb_id, str(record.aligned_file))
    except Exception as e:
        log_message(message_type="error", stage="find_ligands", pdb_id=record.pdb_id, error=str(e))
        return record._replace(protein_state="FAILED")
    allowed = p.allowed_cofactors.split("|") if p.allowed_cofactors else []
    ligands = []
    for ligand, residue in find_ligands(parsed, record.pdb_id, allowed, p.min_heavy_atoms):
        out = ws.extracted_ligands / (f"{ligand.pdb_id}_{ligand.chain}_{ligand.resname}_"
                                      f"{ligand.resseq}{ligand.icode.strip()}.pdb")
        out.write_text(pocket.extract_pdb_block(residue, ligand.chain), encoding="utf-8")
        ligands.append(ligand._replace(file=str(out), smiles=interactions.ligand_smiles(out)))
    log_message(message_type="ligands_found", pdb_id=record.pdb_id,
                ligands=[ligand.key for ligand in ligands])
    return record._replace(ligands=tuple(ligands),
                           protein_state="BOUND_STATE" if ligands else "UNBOUND_STATE")


def _protonated_ligand(path: str):
    from qpocket.io.interactions import protonated_ligand

    try:
        return protonated_ligand(Path(path))
    except Exception as e:
        # One unreadable ligand leaves a gap in the complex, not a structure without contacts.
        log_message(message_type="error", stage="ligand_protonation", ligand=path, error=str(e))
        return None


def protonate(ws: layout.Workspace, record: Structure) -> Structure:
    """Write the docking-ready protonated complex holding every ligand (protein only when apo).

    Returns:
        The record with protonated_file, or None in it when protonation failed.
    """
    from qpocket.core.hydroxyl import optimise_hydroxyls
    from qpocket.io.interactions import load_molecule, write_complex, write_molecule
    from qpocket.io.tools import protonate_for_interactions

    pdb_id = record.pdb_id
    cleaned = record.cleaned_file
    if not cleaned or not Path(cleaned).exists():
        return record._replace(protonated_file=None)

    out_dir = ws.structures / "protonated"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{pdb_id}.pdb"
    # Ligand i sits at residue 900 + i, so a complex is reused only for the same ligand list.
    keys_file = out_dir / f"{pdb_id}.ligands.json"
    keys = json.dumps([ligand.key for ligand in record.ligands])
    if destination.exists() and keys_file.exists() and keys_file.read_text(encoding="utf-8") == keys:
        return record._replace(protonated_file=str(destination))
    keys_file.unlink(missing_ok=True)

    try:
        pqr = protonate_for_interactions(Path(cleaned), out_dir / f"{pdb_id}.pqr")
        write_complex(destination, pqr,
                      [(_protonated_ligand(ligand.file), ligand.resname) for ligand in record.ligands])
        # pdb2pqr placed the hydroxyl hydrogens before the ligands existed, so
        # their rotamers are re-chosen now that they are present.
        write_molecule(destination, optimise_hydroxyls(load_molecule(destination)))
        keys_file.write_text(keys, encoding="utf-8")
    except Exception as error:
        log_message(message_type="error", stage="protonation",
                    pdb_id=pdb_id, error=str(error))
        return record._replace(protonated_file=None)
    return record._replace(protonated_file=str(destination))


def read_contacts(p: StructureParams, ws: layout.Workspace, record: Structure) -> Structure:
    """Each ligand's observed contacts, from the contact plug-ins.

    Returns:
        The record with contacts, one entry per ligand.
    """
    with start_action(action_type="analyse_interactions", pdb_id=record.pdb_id):
        if not record.protonated_file:
            # Name and number alone cannot tell same-named ligands on two chains apart.
            if record.ligands:
                log_message(message_type="warning", stage="read_contacts", pdb_id=record.pdb_id,
                            warning="no protonated complex; contacts not read")
            return record._replace(contacts=tuple((ligand.key, ("no_ligand_found",))
                                                  for ligand in record.ligands))
        # The protonated complex: interactions are read from the hydrogens the pipeline placed.
        contacts = []
        for index, ligand in enumerate(record.ligands):
            found: list = []
            for name in p.contacts:
                # write_complex numbers ligand i as 900 + i.
                found += CONTACTS.get(name).fn(Path(record.protonated_file), ligand.resname,
                                               p.contact_params[name], ligand_resseq=900 + index)
            contacts.append((ligand.key, tuple(found)))
        return record._replace(contacts=tuple(contacts))


def detect_cavities(ws: layout.Workspace, record: Structure) -> Structure:
    """Detect every cavity of the cleaned protein and write structures/cavities/<pdb_id>.csv.

    Returns:
        The record with cavities_file, or with NO_ANY_CAVITY_FOUND / FAILED and none.
    """
    if not record.cleaned_file or not Path(record.cleaned_file).exists():
        return record._replace(protein_state="FAILED")
    try:
        found = run_cavfind(Path(record.cleaned_file), record.pdb_id)
    except Exception as e:
        log_message(message_type="warning", stage="detect_cavities", pdb_id=record.pdb_id,
                    warning=f"cavity detection failed: {e}")
        return record._replace(protein_state="FAILED")
    if not found:
        return record._replace(protein_state="NO_ANY_CAVITY_FOUND")
    table = pd.DataFrame(np.vstack(found), columns=["x", "y", "z"])
    table["cavity"] = np.repeat(np.arange(len(found)), [len(c) for c in found])
    out = ws.cavities / f"{record.pdb_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    log_message(message_type="cavities_detected", pdb_id=record.pdb_id, n_cavities=len(found),
                n_points=len(table))
    return record._replace(cavities_file=str(out))


def fingerprint(ref: Reference, p: StructureParams, raw: RawStructure) -> str:
    """What a structure's result depends on, as one hash.

    Args:
        ref: The reference.
        p: Chain parameters.
        raw: The ingested structure.

    Returns:
        A sha256 hex digest of the input file, the parameters and the reference.
    """
    import hashlib
    import json

    digest = hashlib.sha256(Path(raw.path).read_bytes())
    digest.update(repr(p).encode())
    digest.update(json.dumps(ref.to_json(), sort_keys=True).encode())
    return digest.hexdigest()


def process(ref: Reference, p: StructureParams, ws: layout.Workspace, raw: RawStructure):
    """One structure, from ingested file to collected record; writes its sidecar.

    Args:
        ref: The reference.
        p: Chain parameters.
        ws: The run.
        raw: The ingested structure.

    Returns:
        Its record, or a Failure.
    """
    with start_action(action_type="process_structure", pdb_id=raw.pdb_id):
        key = fingerprint(ref, p, raw)
        try:
            done = layout.read_part(ws, raw.pdb_id)
        except ValueError:
            done = None
        # A re-run reuses a structure finished with the same inputs; failures are retried.
        if done and done.get("fingerprint") == key and "failure" not in done:
            log_message(message_type="skip", stage="structure", pdb_id=raw.pdb_id,
                        reason="finished with the same inputs")
            return Structure.from_json(done)
        result = pipe(
            raw,
            partial(align, ref, p, ws),
            partial(clean, p, ws),
            partial(find_ligands_step, p, ws),
            partial(protonate, ws),
            partial(read_contacts, p, ws),
            partial(detect_cavities, ws),
        )
        layout.write_part(ws, raw.pdb_id, {**result.to_json(), "fingerprint": key})
        log_message(message_type="structure_processed", pdb_id=raw.pdb_id,
                    status=getattr(result, "protein_state", None) or getattr(result, "stage", None))
        return result
