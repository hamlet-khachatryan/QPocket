"""Immutable values passed between core, io and pipelines."""
import re
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np

__all__ = ["Scaler", "Fingerprint", "StructureData", "FeatureContext", "PocketSpec",
           "ComparisonResult", "PQRAtoms", "AlignmentContext", "RawStructure", "Sources",
           "Structure", "Ligand", "Reference"]


class Scaler(NamedTuple):
    """Per-feature standardisation fitted over a set of pockets.

    Attributes:
        mean: Mean of each feature column.
        scale: Standard deviation of each column (1.0 where constant).
    """

    mean: np.ndarray
    scale: np.ndarray


class Fingerprint(NamedTuple):
    """Spectral fingerprint of one pocket.

    Attributes:
        eigenvalues: Laplacian eigenvalues, padded or truncated to global_k.
        spectrum: Flattened graph-Fourier power spectrum of the features.
    """

    eigenvalues: np.ndarray
    spectrum: np.ndarray


class StructureData(NamedTuple):
    """Protein atoms and pharmacophore sites used to featurise a pocket.

    Attributes:
        coords: (M, 3) atom coordinates from the PQR.
        charges: (M,) partial charges.
        hydrophobicity: (M,) per-atom hydrophobicity.
        donors: (K, 3) H-bond donor atoms.
        acceptors: (K, 3) H-bond acceptor atoms.
        aromatics: (K, 3) aromatic ring centres.
    """

    coords: np.ndarray
    charges: np.ndarray
    hydrophobicity: np.ndarray
    donors: np.ndarray
    acceptors: np.ndarray
    aromatics: np.ndarray

    @classmethod
    def empty(cls) -> "StructureData":
        """Atom-free inputs, for runs whose providers need no PQR.

        Returns:
            All arrays zero-length.
        """
        return cls(np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                   np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))


class FeatureContext(NamedTuple):
    """Everything a provider may read about the structure being featurised.

    Attributes:
        points: (N, 3) pocket points.
        pdb_path: Cleaned protein PDB, when one exists.
        pqr_path: Protonated PQR, when a provider needed it.
        structure_data: Atoms, charges, hydrophobicity and pharmacophore sites.
        options: This run's options, including values the pipeline precomputed.
    """

    points: np.ndarray
    pdb_path: Optional[Path]
    pqr_path: Optional[Path]
    structure_data: StructureData
    options: Dict[str, object]


class PocketSpec(NamedTuple):
    """Where a pocket is: a structure, an optional chain, and a ligand or residues.

    Attributes:
        structure: PDB ID (downloaded) or a local PDB/mmCIF path.
        chain: Keep only this chain, or None for all.
        ligand: Ligand residue name marking the pocket.
        residues: Residues lining the pocket, e.g. ("TYR795", "GLU756").
        padding: Distance (A) from those residues that counts as the pocket.
    """

    structure: str
    chain: Optional[str] = None
    ligand: Optional[str] = None
    residues: Tuple[str, ...] = ()
    padding: float = 5.0

    def validate(self) -> "PocketSpec":
        """Check that exactly one pocket definition is given.

        Returns:
            The spec itself.

        Raises:
            ValueError: If neither or both of ligand and residues are set.
        """
        if bool(self.ligand) == bool(self.residues):
            raise ValueError(f"{self.structure}: give a ligand or residues, not both or neither")
        return self

    @classmethod
    def parse(cls, text: str, padding: float = 5.0) -> "PocketSpec":
        """Parse STRUCTURE[@CHAIN]:POCKET, e.g. 7KG6:WDG or a.pdb@A:TYR795,GLU756.

        Args:
            text: Spec string; POCKET is a ligand code or comma-separated residues.
            padding: Pocket padding for residue definitions.

        Returns:
            The spec.

        Raises:
            ValueError: On a malformed string.
        """
        if ":" not in text:
            raise ValueError(f"{text!r}: expected STRUCTURE[@CHAIN]:POCKET")
        where, pocket = text.rsplit(":", 1)  # right split keeps Windows drive letters
        structure, _, chain = where.rpartition("@") if "@" in where else (where, "", "")
        items = tuple(p.strip().upper() for p in pocket.split(",") if p.strip())
        is_residues = len(items) > 1 or bool(re.fullmatch(r"[A-Z]{3}-?\d+[A-Z]?", items[0] if items else ""))
        spec = cls(structure=structure, chain=chain or None,
                   ligand=None if is_residues else (items[0] if items else None),
                   residues=items if is_residues else (), padding=padding)
        return spec.validate()


class ComparisonResult(NamedTuple):
    """Pairwise distances between compared pockets.

    Attributes:
        ids: Pocket labels, in matrix order.
        shape: Shape distance matrix.
        chem: Chemistry distance matrix.
        combined: Blended distance matrix.
        scaler: Feature scaling used for the fingerprints.
    """

    ids: Tuple[str, ...]
    shape: np.ndarray
    chem: np.ndarray
    combined: np.ndarray
    scaler: Scaler

    def to_frame(self, kind: str = "combined"):
        """One matrix as a labelled DataFrame.

        Args:
            kind: "combined", "shape" or "chem".

        Returns:
            Square pandas DataFrame.
        """
        import pandas as pd

        return pd.DataFrame(getattr(self, kind), index=list(self.ids), columns=list(self.ids))


class PQRAtoms(NamedTuple):
    """Atoms of a PQR file (pdb2pqr output, ions repaired).

    Attributes:
        coords: (M, 3) coordinates.
        charges: (M,) partial charges.
        radii: (M,) atomic radii.
        atom_names: Atom names.
        res_names: Residue names (AMBER protonation names).
        chain_ids: Chain IDs; empty for repaired ions.
        res_seqs: Residue numbers.
    """

    coords: np.ndarray
    charges: np.ndarray
    radii: np.ndarray
    atom_names: list
    res_names: list
    chain_ids: list
    res_seqs: list


class AlignmentContext(NamedTuple):
    """What a worker needs to align one structure onto the reference.

    Attributes:
        reference_path: Aligned reference PDB; workers reload it themselves.
        reference_chain: Chain every target is fitted onto.
        reference_sequence: That chain's sequence.
        uniprot_id: Accession used for renumbering, or None.
        aux_sequences: Auxiliary chain sequences, interface mode only.
        aux_centroids: Auxiliary chain CA centroids, interface mode only.
        aux_uniprots: Auxiliary chain accessions, interface mode only.
        renumber: Whether to renumber onto UniProt positions.
        interface_mode: Whether auxiliary chains are matched too.
    """

    reference_path: str
    reference_chain: str
    reference_sequence: str
    uniprot_id: Optional[str]
    aux_sequences: Dict[str, str]
    aux_centroids: Dict[str, Tuple[float, float, float]]
    aux_uniprots: Dict[str, str]
    renumber: bool
    interface_mode: bool


class RawStructure(NamedTuple):
    """One input structure as ingested.

    Attributes:
        pdb_id: Identifier, also the file stem.
        path: The ingested file.
    """

    pdb_id: str
    path: Path


class Sources(NamedTuple):
    """What a structure source ingested.

    Attributes:
        reference: The reference structure.
        structures: Every ingested structure, sorted by path, the reference included.
    """

    reference: RawStructure
    structures: Tuple[RawStructure, ...]


def _array(value) -> Optional[list]:
    return None if value is None else np.asarray(value, dtype=float).tolist()


class Structure(NamedTuple):
    """One structure as it moves through the chain; each step fills more fields.

    Attributes:
        pdb_id: Identifier.
        source: Ingested file.
        rmsd_to_ref: RMSD to the reference after superposition.
        aligned_file: Superposed structure.
        aux_chain_mapping: Auxiliary-chain mapping, interface mode only.
        cleaned_file: Protein and allowed cofactors only.
        points: Pocket cloud before the contact crop; never written to a sidecar.
        protein_state: BOUND_STATE, UNBOUND_STATE, REFERENCE, CAVITY_CLOSED, NO_ANY_CAVITY_FOUND or FAILED.
        ligand_file: Extracted ligand.
        ligand_id: Its chemical-component ID.
        protonated_file: Docking-ready protonated complex.
        potential: Residues within reach of the pocket cloud.
        observed: RESIDUE|InteractionType entries.
        smiles: (SMILES, ligand ID) when a ligand was extracted.
    """

    pdb_id: str
    source: str
    rmsd_to_ref: Optional[float] = None
    aligned_file: Optional[str] = None
    aux_chain_mapping: Optional[str] = None
    cleaned_file: Optional[str] = None
    points: Optional[np.ndarray] = None
    protein_state: Optional[str] = None
    ligand_file: Optional[str] = None
    ligand_id: Optional[str] = None
    protonated_file: Optional[str] = None
    potential: Tuple[str, ...] = ()
    observed: Tuple[str, ...] = ()
    smiles: Optional[Tuple[str, str]] = None

    def master_row(self) -> Dict[str, Any]:
        """This structure's master.csv row, in today's column order.

        Returns:
            Column name to value.
        """
        row: Dict[str, Any] = {"pdb_id": self.pdb_id, "rmsd_to_ref": self.rmsd_to_ref,
                               "aligned_file": self.aligned_file}
        if self.aux_chain_mapping is not None:
            row["aux_chain_mapping"] = self.aux_chain_mapping
        row.update(cleaned_file=self.cleaned_file, ligand_file=self.ligand_file,
                   ligand_id=self.ligand_id, protein_state=self.protein_state or "FAILED")
        return row

    def to_json(self) -> Dict[str, Any]:
        """Plain form for a sidecar, without the point cloud.

        Returns:
            JSON-serialisable dict.
        """
        payload = self._asdict()
        payload.pop("points")
        payload["potential"] = list(self.potential)
        payload["observed"] = list(self.observed)
        payload["smiles"] = None if self.smiles is None else list(self.smiles)
        return payload

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "Structure":
        """Rebuild from :meth:`to_json`; absent fields take their defaults.

        Args:
            payload: Output of to_json.

        Returns:
            The record.
        """
        known = {k: v for k, v in payload.items() if k in cls._fields and k != "points"}
        known["potential"] = tuple(known.get("potential") or ())
        known["observed"] = tuple(known.get("observed") or ())
        smiles = known.get("smiles")
        known["smiles"] = None if smiles is None else (smiles[0], smiles[1])
        return cls(**known)


class Ligand(NamedTuple):
    """One ligand residue found in a structure.

    Attributes:
        pdb_id: Structure identifier.
        chain: Chain holding the residue.
        resname: Chemical-component ID.
        resseq: Residue number.
        icode: Insertion code (blank when none).
        n_heavy: Heavy-atom count.
        file: Extracted ligand file, once written.
        smiles: SMILES, once derived.
    """

    pdb_id: str
    chain: str
    resname: str
    resseq: int
    icode: str
    n_heavy: int
    file: str = ""
    smiles: str = ""

    @property
    def key(self) -> str:
        """Unique within a run: structure, chain, component and number.

        Returns:
            ``pdb:chain:resname:resseq`` with the insertion code appended.
        """
        return f"{self.pdb_id}:{self.chain}:{self.resname}:{self.resseq}{self.icode.strip()}"

    def to_json(self) -> Dict[str, Any]:
        """Plain form for a sidecar.

        Returns:
            JSON-serialisable dict.
        """
        return self._asdict()

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "Ligand":
        """Rebuild from :meth:`to_json`; absent optional fields take their defaults.

        Args:
            payload: Output of to_json.

        Returns:
            The record.
        """
        return cls(**{k: v for k, v in payload.items() if k in cls._fields})


class Reference(NamedTuple):
    """The structure every other one is aligned to and measured against.

    Attributes:
        pdb_id: Its identifier.
        alignment: What a worker needs to align onto it.
        record: Its own aligned and cleaned record.
        centroid: Reference ligand centroid, or None.
        box: (lower, upper) corners of its pocket, or None.
    """

    pdb_id: str
    alignment: AlignmentContext
    record: Structure
    centroid: Optional[np.ndarray]
    box: Optional[Tuple[np.ndarray, np.ndarray]]

    def to_json(self) -> Dict[str, Any]:
        """Plain form, for metadata/reference.json.

        Returns:
            JSON-serialisable dict.
        """
        alignment = self.alignment._asdict()
        alignment["aux_centroids"] = {k: list(v) for k, v in self.alignment.aux_centroids.items()}
        return {"pdb_id": self.pdb_id, "alignment": alignment, "record": self.record.to_json(),
                "centroid": _array(self.centroid),
                "box": None if self.box is None else [_array(self.box[0]), _array(self.box[1])]}

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "Reference":
        """Rebuild from :meth:`to_json`.

        Args:
            payload: Output of to_json.

        Returns:
            The reference.
        """
        alignment = dict(payload["alignment"])
        alignment["aux_centroids"] = {k: tuple(v) for k, v in alignment["aux_centroids"].items()}
        box = payload.get("box")
        centroid = payload.get("centroid")
        return cls(payload["pdb_id"], AlignmentContext(**alignment),
                   Structure.from_json(payload["record"]),
                   None if centroid is None else np.asarray(centroid, dtype=float),
                   None if box is None else (np.asarray(box[0], dtype=float),
                                             np.asarray(box[1], dtype=float)))
