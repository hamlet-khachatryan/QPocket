from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, List, Optional

from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class BoundingBoxParams:
    """Parameters for user-defined bounding box pocket definition."""
    center: Optional[List[float]] = None
    dimensions: Optional[List[float]] = None


@dataclass
class ReferenceParams:
    """Parameters for the reference structure."""
    pdb_id: Optional[str] = None
    chain_id: str = "A"
    ligand_id: Optional[str] = None
    pocket_definition: Literal["Interactions", "Solvent"] = "Interactions"
    pocket_mode: Literal["cavity_detection", "bounding_box"] = "cavity_detection"
    bounding_box: BoundingBoxParams = field(default_factory=BoundingBoxParams)


@dataclass
class PathConfig:
    """Configuration for file paths."""
    workdir: str = field(default_factory=lambda: str(Path.cwd()))


@dataclass
class FragmentScreenParams:
    """Parameters for fragment screening mode."""
    input_type: Literal["csv", "paths"] = "csv"
    csv_path: Optional[str] = None
    structure_paths: Optional[List[str]] = None
    reference_index: int = 0
    interaction_padding: float = 5.0


@dataclass
class QueryParams:
    """Parameters specific for PDB query mode."""
    sequence_identity: float = 1.0
    resolution_limit: float = 2.5
    min_length: int = 20
    max_length: Optional[int] = None
    experimental_method: Optional[str] = None
    enforce_same_uniprot: bool = False


@dataclass
class AlignParams:
    """Parameters specific for sequence alignment."""
    renumber_residues: bool = True
    manual_uniprot_id: Optional[str] = None


@dataclass
class CleanParams:
    """Parameters specific for cleaning structures."""
    allowed_cofactors: Optional[str] = None


@dataclass
class IOConfig:
    """Configuration for I/O behaviour.

    Attributes:
        log_format: Eliot log serialisation format.  ``"json"`` produces
            machine-readable NDJSON; ``"text"`` produces human-readable output.
        save_intermediate: When ``True`` the initial (pre-cropping) pocket
            point cloud is written alongside the final one.
        compression: Compression algorithm used when writing HDF5 datasets.
            ``"gzip"`` reduces file size at the cost of slightly higher write
            latency; ``"none"`` disables compression.
    """
    log_format: Literal["json", "text"] = "json"
    save_intermediate: bool = False
    compression: Literal["gzip", "none"] = "gzip"


@dataclass
class DistanceMatrix:
    """Parameters for spectral distance calculation and clustering.

    Clustering uses Affinity Propagation which determines the number of
    clusters automatically via message passing — no ``k`` needs to be
    specified in advance.

    Attributes:
        geo_graph_neighbor_radius: Maximum Euclidean distance (Å) between
            two pocket points to be considered neighbours in the geodesic
            graph.
        paralell_jobs: Number of parallel workers passed to ``joblib``
            (``-1`` = all available cores).
        global_k: Length of the spectral fingerprint vector.  Eigenvalues
            and GFT features are padded / truncated to this size.
        sigma: Standard deviation of the Heat Gaussian kernel used when
            constructing the normalised graph Laplacian.
        max_points: Point clouds with more points than this threshold are
            skipped during spectral embedding to avoid excessive memory use.
        alpha: Relative weight of the geodesic (eigenvalue) component in
            the combined distance metric.  Must be in ``[0, 1]``.
        beta: Scaling factor applied to the spectral feature component.
        affinity_damping: Damping factor for Affinity Propagation, controls
            the extent to which the current value is maintained relative to
            the incoming values.  Must be in ``[0.5, 1.0)``.
        affinity_max_iter: Maximum number of iterations for Affinity
            Propagation.
    """
    geo_graph_neighbor_radius: float = 2.0
    paralell_jobs: int = -1
    global_k: int = 100
    sigma: float = 10.0
    max_points: int = 5000
    alpha: float = 0.85
    beta: float = 12.5
    affinity_damping: float = 0.9
    affinity_max_iter: int = 300


@dataclass
class QPocketConfig:
    """Aggregated configuration for QPocket pipeline execution."""
    mode: Literal["pdb_query", "fragment_screen"] = "pdb_query"
    reference: ReferenceParams = field(default_factory=ReferenceParams)
    paths: PathConfig = field(default_factory=PathConfig)
    fragment_screen: FragmentScreenParams = field(default_factory=FragmentScreenParams)
    pdb_query: QueryParams = field(default_factory=QueryParams)
    alignment: AlignParams = field(default_factory=AlignParams)
    cleaning: CleanParams = field(default_factory=CleanParams)
    io: IOConfig = field(default_factory=IOConfig)
    spectral_distance_matrix: DistanceMatrix = field(default_factory=DistanceMatrix)

    def __str__(self) -> str:
        return OmegaConf.to_yaml(self)


def validate_config(cfg: "QPocketConfig") -> None:
    """Validates a QPocketConfig instance against definition logic rules.

    Raises:
        ValueError: with a descriptive message on the first failed rule.
    """
    # --- ReferenceParams ---
    ref = cfg.reference

    if not ref.pdb_id or not re.fullmatch(r"[A-Za-z0-9]{4}", ref.pdb_id):
        raise ValueError(
            f"reference.pdb_id must be exactly 4 alphanumeric characters, got: {ref.pdb_id!r}"
        )

    if not re.fullmatch(r"[A-Z]{1,2}", ref.chain_id):
        raise ValueError(
            f"reference.chain_id must be 1-2 uppercase letters, got: {ref.chain_id!r}"
        )

    if ref.pocket_mode == "bounding_box":
        bb = ref.bounding_box
        if bb.center is None or bb.dimensions is None:
            raise ValueError(
                "reference.bounding_box.center and .dimensions must both be set "
                "when pocket_mode == 'bounding_box'"
            )
        if len(bb.center) != 3:
            raise ValueError(
                f"reference.bounding_box.center must be a list of 3 values, got {len(bb.center)}"
            )
        if len(bb.dimensions) != 3:
            raise ValueError(
                f"reference.bounding_box.dimensions must be a list of 3 values, got {len(bb.dimensions)}"
            )
        if any(d <= 0 for d in bb.dimensions):
            raise ValueError(
                "All values in reference.bounding_box.dimensions must be positive"
            )

    if ref.pocket_definition == "Interactions" and ref.pocket_mode == "cavity_detection":
        if not ref.ligand_id:
            raise ValueError(
                "reference.ligand_id must be provided when pocket_definition == 'Interactions' "
                "and pocket_mode == 'cavity_detection'"
            )

    # --- QueryParams ---
    if cfg.mode == "pdb_query":
        q = cfg.pdb_query

        if not (0.0 < q.sequence_identity <= 1.0):
            raise ValueError(
                f"pdb_query.sequence_identity must be in (0.0, 1.0], got: {q.sequence_identity}"
            )
        if q.resolution_limit <= 0.0:
            raise ValueError(
                f"pdb_query.resolution_limit must be > 0.0, got: {q.resolution_limit}"
            )
        if q.min_length < 1:
            raise ValueError(
                f"pdb_query.min_length must be >= 1, got: {q.min_length}"
            )
        if q.max_length is not None and q.max_length <= q.min_length:
            raise ValueError(
                f"pdb_query.max_length ({q.max_length}) must be > min_length ({q.min_length})"
            )

    # --- FragmentScreenParams ---
    if cfg.mode == "fragment_screen":
        fs = cfg.fragment_screen

        if fs.input_type == "csv":
            if not fs.csv_path:
                raise ValueError(
                    "fragment_screen.csv_path must be non-empty when input_type == 'csv'"
                )
        elif fs.input_type == "paths":
            if not fs.structure_paths:
                raise ValueError(
                    "fragment_screen.structure_paths must be a non-empty list when input_type == 'paths'"
                )

        if fs.reference_index < 0:
            raise ValueError(
                f"fragment_screen.reference_index must be >= 0, got: {fs.reference_index}"
            )
        if isinstance(fs.structure_paths, list) and fs.reference_index >= len(fs.structure_paths):
            raise ValueError(
                f"fragment_screen.reference_index ({fs.reference_index}) is out of bounds "
                f"for structure_paths of length {len(fs.structure_paths)}"
            )
        if fs.interaction_padding <= 0:
            raise ValueError(
                f"fragment_screen.interaction_padding must be > 0, got: {fs.interaction_padding}"
            )

    # --- IOConfig ---
    io = cfg.io

    if io.log_format not in ("json", "text"):
        raise ValueError(
            f"io.log_format must be 'json' or 'text', got: {io.log_format!r}"
        )
    if io.compression not in ("gzip", "none"):
        raise ValueError(
            f"io.compression must be 'gzip' or 'none', got: {io.compression!r}"
        )

    # --- DistanceMatrix ---
    dm = cfg.spectral_distance_matrix

    if dm.geo_graph_neighbor_radius <= 0:
        raise ValueError(
            f"spectral_distance_matrix.geo_graph_neighbor_radius must be > 0, "
            f"got: {dm.geo_graph_neighbor_radius}"
        )
    if dm.global_k < 1:
        raise ValueError(
            f"spectral_distance_matrix.global_k must be >= 1, got: {dm.global_k}"
        )
    if dm.sigma <= 0:
        raise ValueError(
            f"spectral_distance_matrix.sigma must be > 0, got: {dm.sigma}"
        )
    if dm.max_points < 5:
        raise ValueError(
            f"spectral_distance_matrix.max_points must be >= 5, got: {dm.max_points}"
        )
    if not (0.0 <= dm.alpha <= 1.0):
        raise ValueError(
            f"spectral_distance_matrix.alpha must be in [0.0, 1.0], got: {dm.alpha}"
        )
    if dm.beta <= 0:
        raise ValueError(
            f"spectral_distance_matrix.beta must be > 0, got: {dm.beta}"
        )
    if dm.paralell_jobs < -1:
        raise ValueError(
            f"spectral_distance_matrix.paralell_jobs must be >= -1 (joblib convention), "
            f"got: {dm.paralell_jobs}"
        )
    if not (0.5 <= dm.affinity_damping < 1.0):
        raise ValueError(
            f"spectral_distance_matrix.affinity_damping must be in [0.5, 1.0), "
            f"got: {dm.affinity_damping}"
        )
    if dm.affinity_max_iter < 1:
        raise ValueError(
            f"spectral_distance_matrix.affinity_max_iter must be >= 1, "
            f"got: {dm.affinity_max_iter}"
        )