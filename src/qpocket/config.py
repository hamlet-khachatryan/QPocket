"""Run configuration: dataclasses, legacy-key migration, defaults and validation."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, List, Literal, Optional

from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


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
    # When set, each entry names a chain in the reference structure that forms the
    # binding-site interface together with chain_id.  Chain IDs may differ across
    # structures — matching is done by sequence identity + spatial proximity.
    auxiliary_chain_ids: Optional[List[str]] = None
    ligand_id: Optional[str] = None
    # Chain that contains the reference ligand.  When null the pipeline searches
    # chain_id first; if the ligand is not found there it falls back to searching
    # all chains (required for homodimers where the ligand sits in a third chain).
    ligand_chain_id: Optional[str] = None
    pocket_definition: Literal["Interactions", "Solvent"] = "Solvent"
    pocket_mode: Literal["cavity_detection", "bounding_box"] = "cavity_detection"
    bounding_box: BoundingBoxParams = field(default_factory=BoundingBoxParams)
    # Padding (Å) around interacting residues when pocket_definition == "Interactions".
    # Applies in both pdb_query and fragment_screen modes.
    interaction_padding: float = 5.0


@dataclass
class PathConfig:
    """Configuration for file paths."""
    workdir: str = field(default_factory=lambda: str(Path.cwd()))


@dataclass
class FragmentScreenParams:
    """Parameters for fragment screening mode.

    Attributes:
        input_type: ``"csv"`` reads paths from a ``pdb_path`` column; ``"paths"`` takes ``structure_paths`` directly.
        csv_path: CSV of structure paths, when ``input_type == "csv"``.
        structure_paths: Structure paths, when ``input_type == "paths"``.
        reference_path: The reference structure.
        reference_index: Index of the reference among the ingested structures, used only when ``reference_path`` is unset.
    """
    input_type: Literal["csv", "paths"] = "csv"
    csv_path: Optional[str] = None
    structure_paths: Optional[List[str]] = None
    # Path to the reference structure. It does not have to appear in
    # structure_paths; when it does it is still only ingested once.
    reference_path: Optional[str] = None
    # Fallback used only when reference_path is unset: index into the ingested
    # structures, in input order.
    reference_index: int = 0


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
    """Parameters specific for sequence alignment.

    Attributes:
        renumber_residues: Remap PDB residue numbers to canonical UniProt positions.
        manual_uniprot_id: Override the auto-detected UniProt ID for the primary chain.
        auxiliary_uniprot_ids: Per-chain UniProt overrides for auxiliary chains, keyed by the **reference** chain ID (e.g. ``{"B": "P00533"}``).
        min_auxiliary_identity: Minimum sequence identity (fraction, ``(0, 1]``) for a target chain to be considered a match to a reference auxiliary chain.
        max_auxiliary_centroid_deviation: Maximum allowed Cα centroid distance (Å) between the matched target chain and the expected reference auxiliary chain position.
        require_all_chains: When ``True`` (default), structures for which any auxiliary chain cannot be matched are silently skipped.
    """
    renumber_residues: Optional[bool] = None
    manual_uniprot_id: Optional[str] = None
    auxiliary_uniprot_ids: Optional[Dict[str, str]] = None
    min_auxiliary_identity: float = 0.70
    max_auxiliary_centroid_deviation: float = 15.0
    require_all_chains: bool = True


@dataclass
class CleanParams:
    """Parameters specific for cleaning structures."""
    allowed_cofactors: Optional[str] = None


@dataclass
class IOConfig:
    """Configuration for I/O behaviour.

    Attributes:
        log_format: Eliot log serialisation format.
        save_intermediate: When ``True`` the initial (pre-cropping) pocket point cloud is written alongside the final one.
        compression: Compression algorithm used when writing HDF5 datasets.
    """
    log_format: Literal["json", "text"] = "json"
    save_intermediate: bool = False
    compression: Literal["gzip", "none"] = "gzip"


@dataclass
class StageClustering:
    """Clustering settings for one stage of a two-stage grouping.

    Attributes:
        method: ``"hierarchical"`` cuts a linkage tree, ``"kmedoids"`` partitions around medoids, ``"affinity"`` lets Affinity Propagation choose its own cluster count and ignores ``k`` / ``k_range``.
        linkage: Linkage used when ``method`` is ``"hierarchical"``.
        k: Pinned cluster count.
        k_range: Inclusive ``[lo, hi]`` bounds scanned when ``k`` is ``None``.
        damping: Affinity Propagation damping, in ``[0.5, 1.0)``.
        max_iter: Affinity Propagation iteration cap.
        criterion: ``"count"`` uses k (or silhouette); ``"threshold"`` cuts a hierarchical tree at ``threshold``.
        threshold: Cut height for a threshold stage, in that stage's distance units.
    """
    method: Literal["hierarchical", "kmedoids", "affinity"] = "hierarchical"
    linkage: Literal["average", "complete", "single", "weighted"] = "average"
    k: Optional[int] = None
    k_range: List[int] = field(default_factory=lambda: [2, 10])
    damping: float = 0.9
    max_iter: int = 300
    criterion: Literal["count", "threshold"] = "count"
    threshold: Optional[float] = None


@dataclass
class ClusteringParams:
    """How pockets are grouped once the distance matrices exist.

    Attributes:
        mode: ``"two_stage"`` groups on shape, then subdivides each group on physicochemistry.
        min_cluster_size: A stage-one cluster with fewer members than this is not subdivided: a split that small cannot be supported by a silhouette.
        single: Settings used when ``mode == "single"``.
        shape: Stage-one settings, applied to the shape distance matrix.
        chem: Stage-two settings, applied to the physicochemical distance submatrix of each stage-one cluster.
    """
    mode: Literal["two_stage", "single"] = "two_stage"
    min_cluster_size: int = 4
    single: StageClustering = field(
        default_factory=lambda: StageClustering(method="affinity")
    )
    shape: StageClustering = field(
        default_factory=lambda: StageClustering(method="hierarchical")
    )
    chem: StageClustering = field(
        default_factory=lambda: StageClustering(method="kmedoids", k_range=[2, 6])
    )


@dataclass
class SlurmStageResources:
    """Resource request for one SLURM stage.

    Attributes:
        cpus: Value of ``--cpus-per-task``.
        mem: Value of ``--mem``, e.g. ``"8G"``.
        time: Walltime as ``HH:MM:SS`` or ``D-HH:MM:SS``.
    """
    cpus: int = 1
    mem: str = "4G"
    time: str = "01:00:00"


@dataclass
class SlurmConfig:
    """Cluster submission settings for ``qpocket slurm``.

    Attributes:
        account: ``--account``.
        partition: ``--partition``.
        qos: ``--qos``.
        max_concurrent: Simultaneously running array tasks, the ``%N`` in ``--array``.
        max_array_size: Cluster's ``MaxArraySize``.
        dependency: ``"afterany"`` lets later stages start once an array has finished and skip whatever is missing; ``"afterok"`` blocks the whole chain if any single task fails.
        split_apbs_stage: Give APBS its own array stage, caching the potential grid per structure, instead of running it inside featurisation.
        environment: Shell lines injected verbatim at the top of every script, for ``module load`` / ``conda activate`` / ``source .../activate``.
        defaults: Resources for any stage without an entry in ``stages``.
        stages: Per-stage resource overrides, keyed by stage name.
    """
    account: Optional[str] = None
    partition: str = "compute"
    qos: Optional[str] = None
    max_concurrent: int = 20
    max_array_size: int = 1000
    dependency: Literal["afterany", "afterok"] = "afterany"
    split_apbs_stage: bool = False
    environment: List[str] = field(default_factory=list)
    defaults: SlurmStageResources = field(default_factory=SlurmStageResources)
    stages: Dict[str, SlurmStageResources] = field(default_factory=dict)


@dataclass
class DistanceConfig:
    """Parameters for spectral distance calculation and clustering.

    Attributes:
        geo_graph_neighbor_radius: Maximum Euclidean distance (Å) between two pocket points to be considered neighbours in the geodesic graph.
        parallel_jobs: Number of parallel workers passed to ``joblib`` (``-1`` = all available cores).
        global_k: Length of the spectral fingerprint vector.
        sigma: Standard deviation of the Heat Gaussian kernel used when constructing the normalised graph Laplacian.
        max_points: Point clouds with more points than this threshold are skipped during spectral embedding to avoid excessive memory use.
        shape_weight: Weight of the shape (Laplacian eigenvalue) term in the blended matrix, in ``[0, 1]``.
        chem_scale: Scale applied to the physicochemical term of that blend.
    """
    geo_graph_neighbor_radius: float = 2.0
    parallel_jobs: int = -1
    global_k: int = 100
    sigma: float = 10.0
    max_points: int = 5000
    shape_weight: float = 0.85
    chem_scale: float = 12.5
    clustering: ClusteringParams = field(default_factory=ClusteringParams)


def _plain(value):
    """Recursively reduce a config to dicts, lists and scalars."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass
class FeaturesConfig:
    """Which feature providers run, and their options.

    Attributes:
        enabled: Provider names; each contributes its columns to featurized.csv.
        options: Per-provider settings, keyed by provider name.
    """

    enabled: List[str] = field(
        default_factory=lambda: ["apbs", "cp_lp", "hbond", "aromatic"]
    )
    options: Dict[str, Dict] = field(default_factory=dict)


def _default_interactions() -> List[str]:
    """Interaction names requested from ProLIF by default.

    Returns:
        The detector's default list.
    """
    from qpocket.core.detect import INTERACTION_NAMES

    return list(INTERACTION_NAMES)


@dataclass
class InteractionsConfig:
    """Which interactions to detect, and with what geometry.

    Attributes:
        enabled: ProLIF interaction names.
        parameters: Per-interaction geometry overrides, keyed by interaction name.
    """

    enabled: List[str] = field(default_factory=_default_interactions)
    parameters: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class ComputeConfig:
    """How much of this machine a local run may use.

    Attributes:
        workers: Parallel workers: 0 uses half the cores, -1 all of them, N exactly N.
        backend: "process" (separate interpreters) or "thread" (shared memory).
    """

    workers: int = 0
    backend: Literal["process", "thread"] = "process"


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
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    interactions: InteractionsConfig = field(default_factory=InteractionsConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def __str__(self) -> str:
        """YAML rendering of this configuration."""
        try:
            return OmegaConf.to_yaml(OmegaConf.create(_plain(self)))
        except Exception:  # pragma: no cover - display must never break a command
            return repr(self)



#: Keys renamed in 0.2, mapped old dotted path -> new dotted path.
#:
#: The base ``config.yaml`` carries only the new names, so any legacy key still
#: present after Hydra has composed and merged came from the user's own file or
#: command line, which is what makes this detectable at all.
LEGACY_KEYS: Dict[str, Optional[str]] = {
    "spectral_distance_matrix.alpha": "distance.shape_weight",
    "spectral_distance_matrix.beta": "distance.chem_scale",
    "spectral_distance_matrix.affinity_damping": "distance.clustering.single.damping",
    "spectral_distance_matrix.affinity_max_iter": "distance.clustering.single.max_iter",
    "spectral_distance_matrix.geo_graph_neighbor_radius": "distance.geo_graph_neighbor_radius",
    "spectral_distance_matrix.parallel_jobs": "distance.parallel_jobs",
    "spectral_distance_matrix.global_k": "distance.global_k",
    "spectral_distance_matrix.sigma": "distance.sigma",
    "spectral_distance_matrix.max_points": "distance.max_points",
    "spectral_distance_matrix.clustering": "distance.clustering",
    "spectral_distance_matrix.clustering.single.min_cluster_size":
        "distance.clustering.min_cluster_size",
    "spectral_distance_matrix.clustering.shape.min_cluster_size":
        "distance.clustering.min_cluster_size",
    "spectral_distance_matrix.clustering.chem.min_cluster_size":
        "distance.clustering.min_cluster_size",
    "fragment_screen.interaction_padding": "reference.interaction_padding",
    # Documented once in a design note, never implemented, still present in
    # experiments/USP7.yaml. There is nowhere to send it.
    "spectral_distance_matrix.max_clusters": None,
    "slurm.enabled": None,
}


def _schema_keys(node: object, prefix: str = "") -> set:
    """Every dotted leaf path the dataclasses define."""
    out = set()
    if is_dataclass(node) and not isinstance(node, type):
        for f in fields(node):
            child = getattr(node, f.name)
            path = f"{prefix}.{f.name}" if prefix else f.name
            out.add(path)
            out |= _schema_keys(child, path)
    return out


def _config_keys(node: object, prefix: str = "") -> set:
    """Every dotted leaf path present in a merged config."""
    out = set()
    if isinstance(node, (dict, DictConfig)):
        for key in node:
            path = f"{prefix}.{key}" if prefix else str(key)
            out.add(path)
            out |= _config_keys(node[key], path)
    return out


def migrate_legacy(cfg: DictConfig) -> List[str]:
    """Rewrite renamed keys in place and report what needs changing.

    Args:
        cfg: The merged configuration.

    Returns:
        Human-readable warnings, one per legacy or unrecognised key.
    """
    warnings: List[str] = []
    renamed: Dict[str, List[str]] = {}
    template = QPocketConfig()

    # Deepest path first: a specific rule such as
    # clustering.shape.min_cluster_size must be consumed before the parent
    # clustering block is moved wholesale, or it rides along to a home that no
    # longer exists.
    for legacy in sorted(LEGACY_KEYS, key=lambda k: k.count("."), reverse=True):
        replacement = LEGACY_KEYS[legacy]
        value = OmegaConf.select(cfg, legacy, default=None)
        if value is None:
            continue

        if replacement is None:
            warnings.append(
                f"{legacy} is obsolete and is being ignored; remove it"
            )
        else:
            already = OmegaConf.select(cfg, replacement, default=None)
            default = OmegaConf.select(
                OmegaConf.create(_plain(template)), replacement, default=None
            )

            if isinstance(value, DictConfig):
                # A whole block: merge rather than replace, so settings a more
                # specific rule already moved into place are not overwritten and
                # the rest of the legacy block is not dropped.
                merged = value if already is None else OmegaConf.merge(value, already)
                OmegaConf.update(cfg, replacement, merged, force_add=True)
                renamed.setdefault(legacy.split(".")[0], []).append(legacy)
            elif already is not None and already != default:
                warnings.append(
                    f"{legacy} is deprecated and {replacement} is also set; "
                    f"using {replacement}"
                )
            else:
                OmegaConf.update(cfg, replacement, value, force_add=True)
                renamed.setdefault(legacy.split(".")[0], []).append(legacy)

        parent, _, leaf = legacy.rpartition(".")
        node = OmegaConf.select(cfg, parent) if parent else cfg
        if node is not None and leaf in node:
            del node[leaf]

    # One line per renamed section rather than one per key: a config written
    # against 0.1 trips a dozen of these, and a dozen near-identical lines are
    # read as noise rather than as an instruction.
    for section, keys in sorted(renamed.items()):
        target = LEGACY_KEYS[keys[0]].split(".")[0]
        warnings.append(
            f"{section} was renamed to {target} in 0.2 "
            f"({len(keys)} setting{'s' if len(keys) != 1 else ''} moved: "
            f"{', '.join(sorted(k.split('.', 1)[1] for k in keys))}); "
            f"the old names still work but will be removed"
        )

    # Drop legacy containers emptied by the moves above, so they are not then
    # reported as unrecognised settings.
    for legacy in sorted({k.split(".")[0] for k in LEGACY_KEYS}, reverse=True):
        node = OmegaConf.select(cfg, legacy, default=None)
        if isinstance(node, DictConfig) and len(node) == 0:
            del cfg[legacy]

    # Anything the schema does not define is almost always a typo. Warn rather
    # than fail: Hydra composition legitimately carries extra keys, and aborting
    # a long run over one stray line would be worse than the stray line.
    known = _schema_keys(template) | {"defaults"}
    for key in sorted(_config_keys(cfg) - known):
        if key.split(".")[0] in ("slurm",) and ".stages." in key:
            continue  # slurm.stages is a free-form mapping of stage name -> resources
        warnings.append(f"{key} is not a recognised setting and will be ignored")

    return warnings


def fill_defaults(cfg: DictConfig) -> DictConfig:
    """Insert every schema key the config does not mention, with its default.

    Args:
        cfg: The merged configuration, modified in place.

    Returns:
        The same config, now complete.
    """
    def walk(node: DictConfig, template: object) -> None:
        if not is_dataclass(template) or isinstance(template, type):
            return
        for f in fields(template):
            default = getattr(template, f.name)
            if f.name not in node:
                node[f.name] = _plain(default)
            elif is_dataclass(default) and not isinstance(default, type):
                child = node[f.name]
                if isinstance(child, DictConfig):
                    walk(child, default)

    OmegaConf.set_struct(cfg, False)
    walk(cfg, QPocketConfig())
    return cfg


def validate_config(cfg: "QPocketConfig") -> None:
    """Validates a QPocketConfig instance against definition logic rules.

    Raises:
        ValueError: with a descriptive message on the first failed rule.
    """
    # --- ReferenceParams ---
    ref = cfg.reference

    # A fragment-screen run has no PDB accession: its structures are local files.
    # pdb_id stays optional there and only names the run directory.
    if cfg.mode == "pdb_query":
        if not ref.pdb_id or not re.fullmatch(r"[A-Za-z0-9]{4}", ref.pdb_id):
            raise ValueError(
                f"reference.pdb_id must be exactly 4 alphanumeric characters, got: {ref.pdb_id!r}"
            )
    elif ref.pdb_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", ref.pdb_id):
        raise ValueError(
            f"reference.pdb_id names the run directory in fragment_screen mode, so it "
            f"must be 1-64 characters of letters, digits, '_' or '-', got: {ref.pdb_id!r}"
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

    if ref.interaction_padding <= 0:
        raise ValueError(
            f"reference.interaction_padding must be > 0, got: {ref.interaction_padding}"
        )

    if ref.ligand_chain_id is not None and not re.fullmatch(r"[A-Z]{1,2}", ref.ligand_chain_id):
        raise ValueError(
            f"reference.ligand_chain_id must be 1–2 uppercase letters, "
            f"got: {ref.ligand_chain_id!r}"
        )

    if ref.auxiliary_chain_ids is not None:
        for aux_chain in ref.auxiliary_chain_ids:
            if not re.fullmatch(r"[A-Z]{1,2}", aux_chain):
                raise ValueError(
                    f"reference.auxiliary_chain_ids contains an invalid chain ID "
                    f"{aux_chain!r} (must be 1–2 uppercase letters)"
                )
        if ref.chain_id in ref.auxiliary_chain_ids:
            raise ValueError(
                "reference.chain_id must not also appear in reference.auxiliary_chain_ids"
            )

    # --- AlignParams ---
    aln = cfg.alignment

    if not (0.0 < aln.min_auxiliary_identity <= 1.0):
        raise ValueError(
            f"alignment.min_auxiliary_identity must be in (0.0, 1.0], "
            f"got: {aln.min_auxiliary_identity}"
        )
    if aln.max_auxiliary_centroid_deviation <= 0:
        raise ValueError(
            f"alignment.max_auxiliary_centroid_deviation must be > 0, "
            f"got: {aln.max_auxiliary_centroid_deviation}"
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
        # Sequence, not list: OmegaConf hands back a ListConfig, so an isinstance
        # check against list never fires for a config loaded from YAML.
        if (
            fs.reference_path is None
            and isinstance(fs.structure_paths, Sequence)
            and not isinstance(fs.structure_paths, (str, bytes))
            and fs.reference_index >= len(fs.structure_paths)
        ):
            raise ValueError(
                f"fragment_screen.reference_index ({fs.reference_index}) is out of bounds "
                f"for structure_paths of length {len(fs.structure_paths)}"
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
    dm = cfg.distance

    if dm.geo_graph_neighbor_radius <= 0:
        raise ValueError(
            f"distance.geo_graph_neighbor_radius must be > 0, "
            f"got: {dm.geo_graph_neighbor_radius}"
        )
    if dm.global_k < 1:
        raise ValueError(
            f"distance.global_k must be >= 1, got: {dm.global_k}"
        )
    if dm.sigma <= 0:
        raise ValueError(
            f"distance.sigma must be > 0, got: {dm.sigma}"
        )
    if dm.max_points < 5:
        raise ValueError(
            f"distance.max_points must be >= 5, got: {dm.max_points}"
        )
    if not (0.0 <= dm.shape_weight <= 1.0):
        raise ValueError(
            f"distance.shape_weight must be in [0.0, 1.0], got: {dm.shape_weight}"
        )
    if dm.chem_scale <= 0:
        raise ValueError(
            f"distance.chem_scale must be > 0, got: {dm.chem_scale}"
        )
    if dm.parallel_jobs < -1:
        raise ValueError(
            f"distance.parallel_jobs must be >= -1 (joblib convention), "
            f"got: {dm.parallel_jobs}"
        )

    # --- DistanceMatrix.clustering ---
    clust = dm.clustering

    if clust.min_cluster_size < 2:
        raise ValueError(
            f"distance.clustering.min_cluster_size must be >= 2, "
            f"got: {clust.min_cluster_size}"
        )
    if clust.mode not in ("two_stage", "single"):
        raise ValueError(
            f"distance.clustering.mode must be 'two_stage' or "
            f"'single', got: {clust.mode!r}"
        )

    for stage_name in ("single", "shape", "chem"):
        stage = getattr(clust, stage_name)
        prefix = f"distance.clustering.{stage_name}"

        if stage.method not in ("hierarchical", "kmedoids", "affinity"):
            raise ValueError(
                f"{prefix}.method must be 'hierarchical', 'kmedoids' or "
                f"'affinity', got: {stage.method!r}"
            )
        if stage.linkage not in ("average", "complete", "single", "weighted"):
            raise ValueError(
                f"{prefix}.linkage must be one of average/complete/single/weighted "
                f"(ward, centroid and median assume Euclidean coordinates and are "
                f"invalid for a precomputed distance matrix), got: {stage.linkage!r}"
            )
        if stage.k is not None and stage.k < 2:
            raise ValueError(
                f"{prefix}.k must be >= 2 when set, or null to choose it by "
                f"silhouette, got: {stage.k}"
            )
        if len(stage.k_range) != 2:
            raise ValueError(
                f"{prefix}.k_range must be a list of 2 values [lo, hi], "
                f"got {len(stage.k_range)}"
            )
        if stage.k_range[0] < 2:
            raise ValueError(
                f"{prefix}.k_range lower bound must be >= 2 (silhouette is "
                f"undefined below two clusters), got: {stage.k_range[0]}"
            )
        if stage.k_range[1] < stage.k_range[0]:
            raise ValueError(
                f"{prefix}.k_range upper bound ({stage.k_range[1]}) must be >= "
                f"lower bound ({stage.k_range[0]})"
            )
        if not (0.5 <= stage.damping < 1.0):
            raise ValueError(
                f"{prefix}.damping must be in [0.5, 1.0), got: {stage.damping}"
            )
        if stage.max_iter < 1:
            raise ValueError(
                f"{prefix}.max_iter must be >= 1, got: {stage.max_iter}"
            )
        criterion = getattr(stage, "criterion", "count")
        if criterion not in ("count", "threshold"):
            raise ValueError(f"{prefix}.criterion must be 'count' or 'threshold', got: {criterion!r}")
        if criterion == "threshold":
            threshold = getattr(stage, "threshold", None)
            if stage.method != "hierarchical" or threshold is None or threshold <= 0:
                raise ValueError(
                    f"{prefix}.threshold must be > 0 with method 'hierarchical' when "
                    f"criterion is 'threshold', got method {stage.method!r}, threshold {threshold!r}"
                )

    # --- FeaturesConfig ---
    from qpocket.core.providers import base as _providers

    if not cfg.features.enabled:
        raise ValueError("features.enabled must name at least one provider")
    try:
        _providers.resolve(list(cfg.features.enabled))
    except ValueError as exc:
        raise ValueError(f"features.enabled: {exc}") from exc

    # --- InteractionsConfig ---
    from qpocket.core.detect import LABELS as _interaction_labels

    if not cfg.interactions.enabled:
        raise ValueError("interactions.enabled must name at least one interaction")
    for name in cfg.interactions.enabled:
        if name == "PiStacking":
            raise ValueError(
                "interactions.enabled: use FaceToFace and EdgeToFace instead of "
                "PiStacking, which is their union and would count each stack twice"
            )
        if name not in _interaction_labels:
            raise ValueError(
                f"interactions.enabled: unknown interaction {name!r}; "
                f"choose from {', '.join(sorted(_interaction_labels))}"
            )
    for name in cfg.interactions.parameters:
        if name not in cfg.interactions.enabled:
            raise ValueError(
                f"interactions.parameters: {name!r} is configured but not enabled; "
                f"add it to interactions.enabled or drop its parameters"
            )

    # --- ComputeConfig ---
    if cfg.compute.workers < -1:
        raise ValueError(
            f"compute.workers must be >= -1 (0 = half the cores, -1 = all), "
            f"got: {cfg.compute.workers}"
        )
    if cfg.compute.backend not in ("process", "thread"):
        raise ValueError(
            f"compute.backend must be 'process' or 'thread', got: {cfg.compute.backend!r}"
        )

    # --- SlurmConfig ---
    sl = cfg.slurm

    if sl.dependency not in ("afterany", "afterok"):
        raise ValueError(
            f"slurm.dependency must be 'afterany' or 'afterok', got: {sl.dependency!r}"
        )
    if not sl.partition:
        raise ValueError("slurm.partition must be a non-empty string")
    if sl.max_concurrent < 1:
        raise ValueError(
            f"slurm.max_concurrent must be >= 1, got: {sl.max_concurrent}"
        )
    if sl.max_array_size < 1:
        raise ValueError(
            f"slurm.max_array_size must be >= 1, got: {sl.max_array_size}"
        )

    known_stages = ("prepare", "apbs", "featurize", "collect", "geodesic", "distance")
    for stage_name in (sl.stages or {}):
        if stage_name not in known_stages:
            raise ValueError(
                f"slurm.stages contains unknown stage {stage_name!r}; "
                f"expected one of {known_stages}"
            )

    resource_blocks = [("slurm.defaults", sl.defaults)]
    resource_blocks += [
        (f"slurm.stages.{name}", (sl.stages or {})[name]) for name in (sl.stages or {})
    ]
    for prefix, res in resource_blocks:
        if res.cpus < 1:
            raise ValueError(f"{prefix}.cpus must be >= 1, got: {res.cpus}")
        if not re.fullmatch(r"\d+(\.\d+)?[KMGT]B?", str(res.mem)):
            raise ValueError(
                f"{prefix}.mem must look like '4G', '512M' or '32GB', "
                f"got: {res.mem!r}"
            )
        if not re.fullmatch(r"(\d+-)?\d{1,2}:\d{2}:\d{2}", str(res.time)):
            raise ValueError(
                f"{prefix}.time must be 'HH:MM:SS' or 'D-HH:MM:SS', "
                f"got: {res.time!r}"
            )
