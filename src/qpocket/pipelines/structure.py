"""One structure from input file to featurised pocket: a pipe of small steps."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from biopandas.pdb import PandasPdb
from eliot import log_message, start_action

from qpocket.config import QPocketConfig, plugin_params
from qpocket.core import pocket
from qpocket.core.features import DEFAULT_PROVIDERS, pocket_features
from qpocket.core.geometry import crop_pocket_by_interactions
from qpocket.core.ports import CHARGES, CONTACTS, FEATURES, POCKETS
from qpocket.core.result import Failure, pipe
from qpocket.core.types import RawStructure, Reference, Structure
from qpocket.io import layout, tables, tools
from qpocket.io.tables import write_points
from qpocket.pipelines import align as alignment
from qpocket.pipelines import clean as cleaning
from qpocket.pipelines.distance import compute_geodesic
from qpocket.pipelines.reference import reference_residues
from qpocket.types import StructureData


@dataclass(frozen=True)
class StructureParams:
    """Everything the chain reads from the configuration.

    Attributes:
        allowed_cofactors: "|"-separated residue names kept by cleaning and skipped as ligands.
        pocket: POCKETS plug-in name.
        pocket_params: Its parameters.
        pocket_definition: "Solvent" or "Interactions" (crop to the observed contacts).
        interaction_padding: Crop radius around interacting residues, Å.
        reference_ligand_id: The reference ligand, or None.
        save_intermediate: Also write binding_site_initial.csv.
        features: FEATURES plug-in names.
        feature_options: Settings per feature plug-in.
        charges: CHARGES plug-in name.
        charge_params: Its parameters.
        contacts: CONTACTS plug-in names.
        contact_params: Parameters per contact plug-in.
        geodesic_radius: Neighbour radius of the geodesic graph.
        min_auxiliary_identity: Interface mode alignment threshold.
        max_auxiliary_centroid_deviation: Interface mode alignment threshold.
        require_all_chains: Interface mode alignment rule.
    """

    allowed_cofactors: str
    pocket: str
    pocket_params: Any
    pocket_definition: str
    interaction_padding: float
    reference_ligand_id: Optional[str]
    save_intermediate: bool
    features: Tuple[str, ...]
    feature_options: Mapping[str, Any]
    charges: str
    charge_params: Any
    contacts: Tuple[str, ...]
    contact_params: Mapping[str, Any]
    geodesic_radius: float
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
    if cfg.reference.pocket_mode == "bounding_box":
        box = cfg.reference.bounding_box
        pocket_name = "box"
        pocket_params = POCKETS.get("box").params(
            center=tuple(box.center) if box.center else None,
            dimensions=tuple(box.dimensions) if box.dimensions else None)
    else:
        pocket_name, pocket_params = "cavities", POCKETS.get("cavities").params()
    prolif_params = CONTACTS.get("prolif").params(
        enabled=tuple(cfg.interactions.enabled), parameters=cfg.interactions.parameters)
    return StructureParams(
        allowed_cofactors=cfg.cleaning.allowed_cofactors or "",
        pocket=pocket_name, pocket_params=pocket_params,
        pocket_definition=cfg.reference.pocket_definition,
        interaction_padding=cfg.reference.interaction_padding,
        reference_ligand_id=cfg.reference.ligand_id or None,
        save_intermediate=bool(cfg.io.save_intermediate),
        features=tuple(cfg.features.enabled), feature_options=dict(cfg.features.options or {}),
        charges=cfg.charges.use,
        charge_params=plugin_params(CHARGES, cfg.charges.use, cfg.charges.options),
        contacts=tuple(cfg.contacts.use),
        contact_params={name: prolif_params if name == "prolif"
                        else plugin_params(CONTACTS, name, cfg.contacts.options)
                        for name in cfg.contacts.use},
        geodesic_radius=cfg.distance.geo_graph_neighbor_radius,
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


def _write_ligand(record: Structure, ws: layout.Workspace, points: np.ndarray, allowed: str,
                  unbound: str, error: str) -> Tuple[Optional[str], str, Optional[str]]:
    """(ligand file, state, ligand id) for the largest ligand in the pocket."""
    try:
        structure = PDBParser(QUIET=True).get_structure(record.pdb_id, str(record.aligned_file))
    except Exception:
        return None, error, None
    choice = pocket.pick_ligand(structure, points, allowed.split("|") if allowed else [])
    if choice is None:
        return None, unbound, None
    res_name = choice.residue.get_resname().strip()
    lig_out = ws.extracted_ligands / f"{record.pdb_id}_{res_name}.pdb"
    lig_out.write_text(pocket.extract_pdb_block(choice.residue, choice.chain_id), encoding="utf-8")
    log_message(message_type="ligand_extracted", pdb_id=record.pdb_id, ligand=res_name,
                n_atoms=choice.n_atoms, n_candidates=choice.n_candidates)
    return str(lig_out), "BOUND_STATE", res_name


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


def _detect(p: StructureParams, record: Structure, ref: Reference):
    plugin = POCKETS.get(p.pocket)
    return plugin.fn(record, ref, p.pocket_params)


def locate(ref: Reference, p: StructureParams, ws: layout.Workspace, record: Structure) -> Structure:
    """Find the pocket and write binding_site.csv.

    Returns:
        The record with points, or with a pocket state and no points.
    """
    if not Path(record.cleaned_file).exists():
        return record._replace(protein_state="FAILED")
    pocket_dir = ws.pockets / record.pdb_id
    pocket_dir.mkdir(parents=True, exist_ok=True)
    points, state = _detect(p, record, ref)
    if points is None:
        return record._replace(protein_state=state)
    if p.save_intermediate:
        write_points(points, pocket_dir / "binding_site_initial.csv")
    write_points(points, pocket_dir / "binding_site.csv")
    return record._replace(points=points)


def _reference_branch(ref: Reference, record: Structure) -> bool:
    # The reference's own pocket came from its ligand; others are matched to its box.
    return record.pdb_id == ref.pdb_id and ref.box is not None


def first_ligand(ref: Reference, p: StructureParams, ws: layout.Workspace, record: Structure) -> Structure:
    """Pick the ligand in the uncropped pocket.

    Returns:
        The record with a state and, when bound, its ligand.
    """
    if record.protein_state is not None:
        return record
    if _reference_branch(ref, record):
        if not p.reference_ligand_id:
            return record._replace(protein_state="REFERENCE")
        # Today's reference branch picks from binding_site.csv as written (rounded), not the
        # in-memory cloud; reading it back keeps a boundary atom on the same side.
        cloud = pd.read_csv(ws.pockets / record.pdb_id / "binding_site.csv")[["x", "y", "z"]].to_numpy()
        lf, state, lig = _write_ligand(record, ws, cloud, p.allowed_cofactors,
                                       unbound="REFERENCE", error="REFERENCE")
    else:
        lf, state, lig = _write_ligand(record, ws, record.points, p.allowed_cofactors,
                                       unbound="UNBOUND_STATE", error="FAILED")
    return record._replace(protein_state=state, ligand_file=lf, ligand_id=lig)


def protonate(ws: layout.Workspace, record: Structure) -> Structure:
    """Write the docking-ready protonated complex (protein only when apo).

    Returns:
        The record with protonated_file, or None in it when protonation failed.
    """
    from qpocket.core.hydroxyl import optimise_hydroxyls
    from qpocket.io.interactions import (load_molecule, protonated_ligand,
                                         write_complex, write_molecule)
    from qpocket.io.tools import protonate_for_interactions

    pdb_id = record.pdb_id
    cleaned = record.cleaned_file
    if not cleaned or not Path(cleaned).exists():
        return record._replace(protonated_file=None)

    out_dir = ws.structures / "protonated"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{pdb_id}.pdb"
    if destination.exists():
        return record._replace(protonated_file=str(destination))

    try:
        pqr = protonate_for_interactions(Path(cleaned), out_dir / f"{pdb_id}.pqr")
        ligand_mol = None
        ligand_id = record.ligand_id
        ligand_file = record.ligand_file
        if ligand_id and ligand_file and Path(ligand_file).exists():
            ligand_mol = protonated_ligand(Path(ligand_file))
        write_complex(destination, pqr, ligand_mol, ligand_id)
        # pdb2pqr placed the hydroxyl hydrogens before the ligand existed, so
        # their rotamers are re-chosen now that it is present.
        write_molecule(destination, optimise_hydroxyls(load_molecule(destination)))
    except Exception as error:
        log_message(message_type="error", stage="protonation",
                    pdb_id=pdb_id, error=str(error))
        return record._replace(protonated_file=None)
    return record._replace(protonated_file=str(destination))


def _potential(record: Structure, ws: layout.Workspace) -> Tuple[str, ...]:
    cav_csv = ws.pockets / record.pdb_id / "binding_site.csv"
    if not cav_csv.exists() or not record.cleaned_file or not Path(record.cleaned_file).exists():
        return ()
    try:
        from scipy.spatial.distance import cdist

        atoms = PandasPdb().read_pdb(str(record.cleaned_file)).df["ATOM"]
        cloud = pd.read_csv(cav_csv)[["x", "y", "z"]].to_numpy()
        dists = cdist(atoms[["x_coord", "y_coord", "z_coord"]].to_numpy(), cloud)
        nearby = atoms[dists.min(axis=1) <= 4.0]
        # Chain ID included so potential and observed keys intersect in utilisation.
        unique = (nearby["residue_name"] + "_" + nearby["residue_number"].astype(str)
                  + "_" + nearby["chain_id"]).unique()
        found = tuple(sorted(unique.tolist()))
        log_message(message_type="potential_interactions", pdb_id=record.pdb_id, n_residues=len(found))
        return found
    except Exception as e:
        log_message(message_type="error", stage="potential_interactions", pdb_id=record.pdb_id,
                    error=str(e))
        return ()


def read_contacts(p: StructureParams, ws: layout.Workspace, record: Structure) -> Structure:
    """Potential contacts from the pocket cloud; observed ones from the contact plug-ins.

    Returns:
        The record with potential and observed.
    """
    with start_action(action_type="analyse_interactions", pdb_id=record.pdb_id):
        potential = _potential(record, ws)
        ligand = record.ligand_id
        if record.protein_state == "BOUND_STATE" and ligand and str(ligand).strip():
            # The protonated complex: interactions are read from the hydrogens the pipeline placed.
            source = Path(record.protonated_file or record.aligned_file)
            observed: list = []
            for name in p.contacts:
                observed += CONTACTS.get(name).fn(source, str(ligand), p.contact_params[name])
        else:
            observed = ["no_ligand_found"]
        return record._replace(potential=potential, observed=tuple(observed))


def _interacting_coords(aligned: str, observed: Sequence[str]) -> Optional[np.ndarray]:
    try:
        atoms = PandasPdb().read_pdb(str(aligned)).df["ATOM"]
        return pocket.interacting_residue_coords(atoms, list(observed))
    except Exception as e:
        log_message(message_type="error", stage="get_interacting_residues_coords", error=str(e))
        return None


def refine_and_featurize(ref: Reference, p: StructureParams, ws: layout.Workspace,
                         record: Structure) -> Structure:
    """Crop to the observed contacts when asked, re-pick the ligand there, then featurise once.

    Returns:
        The record, with the cropped pocket's ligand when the crop ran.
    """
    pocket_dir = ws.pockets / record.pdb_id
    crop = (p.pocket_definition == "Interactions" and record.observed
            and record.protein_state not in ("FAILED", "NO_ANY_CAVITY_FOUND"))
    if record.points is None:
        return record
    if not crop:
        if (pocket_dir / "featurized.csv").exists():
            log_message(message_type="skip", stage="featurise", pdb_id=record.pdb_id,
                        reason="already featurised")
        else:
            featurize(pocket_dir / "binding_site.csv", Path(record.cleaned_file), p.features, p.charges,
                      options=p.feature_options, charge_params=p.charge_params)
        return record

    points = record.points
    interacting = _interacting_coords(record.aligned_file, record.observed)
    if interacting is not None:
        cropped = crop_pocket_by_interactions(points, interacting, padding=p.interaction_padding)
        if len(cropped) > 0:
            log_message(message_type="pocket_cropped", pdb_id=record.pdb_id,
                        n_before=len(points), n_after=len(cropped))
            points = cropped
    write_points(points, pocket_dir / "binding_site.csv")
    (pocket_dir / "featurized.csv").unlink(missing_ok=True)  # the cloud changed
    featurize(pocket_dir / "binding_site.csv", Path(record.cleaned_file), p.features, p.charges,
                      options=p.feature_options, charge_params=p.charge_params)
    lf, state, lig = _write_ligand(record, ws, points, p.allowed_cofactors,
                                   unbound="UNBOUND_STATE", error="FAILED")
    if state == "FAILED":
        return record
    return record._replace(protein_state=state, ligand_file=lf, ligand_id=lig)


def geodesics(p: StructureParams, ws: layout.Workspace, record: Structure) -> Structure:
    """Cache the pocket's geodesic distances while it is warm.

    Returns:
        The record, unchanged.
    """
    featurized = ws.pockets / record.pdb_id / "featurized.csv"
    if featurized.exists():
        compute_geodesic(featurized, p.geodesic_radius)
    return record


def finish(record: Structure) -> Structure:
    """Attach the ligand's SMILES and drop the point cloud.

    Returns:
        The finished record.
    """
    smiles = None
    if record.ligand_file and record.ligand_id and Path(record.ligand_file).exists():
        from qpocket.io.interactions import ligand_smiles

        found = ligand_smiles(Path(record.ligand_file))
        smiles = (found, str(record.ligand_id)) if found else None
    return record._replace(smiles=smiles, points=None)


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
            partial(locate, ref, p, ws),
            partial(first_ligand, ref, p, ws),
            partial(protonate, ws),
            partial(read_contacts, p, ws),
            partial(refine_and_featurize, ref, p, ws),
            partial(geodesics, p, ws),
            finish,
        )
        layout.write_part(ws, raw.pdb_id, {**result.to_json(), "fingerprint": key})
        log_message(message_type="structure_processed", pdb_id=raw.pdb_id,
                    status=getattr(result, "protein_state", None) or getattr(result, "stage", None))
        return result
