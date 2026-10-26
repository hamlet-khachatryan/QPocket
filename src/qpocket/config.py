"""Run configuration: dataclasses, legacy-key migration, defaults and validation."""
from __future__ import annotations

import dataclasses
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, List, Literal, Optional, Union

from pathlib import Path

from eliot import log_message
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from qpocket.io.layout import STRUCTURE_SUFFIXES  # noqa: F401  (re-exported)


@dataclass(frozen=True)
class ReferenceParams:
    """Parameters for the reference structure."""
    pdb_id: Optional[str] = None
    chain_id: str = "A"
    # When set, each entry names a chain in the reference structure that forms the
    # binding-site interface together with chain_id.  Chain IDs may differ across
    # structures — matching is done by sequence identity + spatial proximity.
    auxiliary_chain_ids: Optional[List[str]] = None
    pocket_definition: Literal["Interactions", "Solvent"] = "Solvent"
    # Padding (Å) around interacting residues when pocket_definition == "Interactions".
    # Applies in both pdb_query and fragment_screen modes.
    interaction_padding: float = 5.0


@dataclass(frozen=True)
class PathConfig:
    """Configuration for file paths."""
    workdir: str = field(default_factory=lambda: str(Path.cwd()))


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class QueryParams:
    """Parameters specific for PDB query mode."""
    sequence_identity: float = 1.0
    resolution_limit: float = 2.5
    min_length: int = 20
    max_length: Optional[int] = None
    experimental_method: Optional[str] = None
    enforce_same_uniprot: bool = False


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class CleanParams:
    """Parameters specific for cleaning structures."""
    allowed_cofactors: Optional[str] = None


@dataclass(frozen=True)
class IOConfig:
    """Configuration for I/O behaviour.

    Attributes:
        log_format: Eliot log serialisation format.
        compression: Compression algorithm used when writing HDF5 datasets.
    """
    log_format: Literal["json", "text"] = "json"
    compression: Literal["gzip", "none"] = "gzip"


@dataclass(frozen=True)
class StageClustering:
    """Clustering settings for one stage of a two-stage grouping.

    Attributes:
        method: ``"hierarchical"`` cuts a linkage tree, ``"kmedoids"`` partitions around medoids, ``"affinity"`` lets Affinity Propagation choose its own cluster count and ignores ``k`` and ``criterion``.
        linkage: Linkage used when ``method`` is ``"hierarchical"``.
        k: Cluster count for the ``"count"`` criterion.
        damping: Affinity Propagation damping, in ``[0.5, 1.0)``.
        max_iter: Affinity Propagation iteration cap.
        criterion: ``"count"`` makes k clusters; ``"threshold"`` cuts a hierarchical tree at the height ``threshold``; ``"relative"`` cuts it at ``threshold`` times its tallest merge.
        threshold: Cut height for a threshold stage, in that stage's distance units; for a relative stage, a fraction in ``(0, 1]``.
    """
    method: Literal["hierarchical", "kmedoids", "affinity"] = "hierarchical"
    linkage: Literal["average", "complete", "single", "weighted"] = "average"
    k: Optional[int] = None
    damping: float = 0.9
    max_iter: int = 300
    criterion: Literal["count", "threshold", "relative"] = "relative"
    threshold: Optional[float] = 0.5


@dataclass(frozen=True)
class ClusteringParams:
    """How pockets are grouped once the distance matrices exist.

    Attributes:
        mode: ``"two_stage"`` groups on shape, then subdivides each group on physicochemistry; ``"shape_only"`` stops after the shape stage.
        min_cluster_size: A stage-one cluster with fewer members than this is not subdivided.
        single: Settings used when ``mode == "single"``.
        shape: Stage-one settings, applied to the shape distance matrix.
        chem: Stage-two settings, applied to the physicochemical distance submatrix of each stage-one cluster.
    """
    mode: Literal["two_stage", "single", "shape_only"] = "two_stage"
    min_cluster_size: int = 4
    single: StageClustering = field(
        default_factory=lambda: StageClustering(method="affinity")
    )
    shape: StageClustering = field(
        default_factory=lambda: StageClustering(method="hierarchical")
    )
    chem: StageClustering = field(
        default_factory=lambda: StageClustering(method="hierarchical")
    )


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class SlurmConfig:
    """Cluster submission settings for ``qpocket slurm``.

    Attributes:
        account: ``--account``.
        partition: ``--partition``.
        qos: ``--qos``.
        max_concurrent: Simultaneously running array tasks, the ``%N`` in ``--array``.
        max_array_size: Cluster's ``MaxArraySize``.
        dependency: ``"afterany"`` lets later stages start once an array has finished and skip whatever is missing; ``"afterok"`` blocks the whole chain if any single task fails.
        environment: Shell lines injected verbatim at the top of every script, for ``module load`` / ``conda activate`` / ``source .../activate``.
        defaults: Resources for any stage without an entry in ``stages``.
        stages: Per-stage resource overrides, keyed by stage name: prepare, structure, sites, site_featurize, site_cluster.
    """
    account: Optional[str] = None
    partition: str = "compute"
    qos: Optional[str] = None
    max_concurrent: int = 20
    max_array_size: int = 1000
    dependency: Literal["afterany", "afterok"] = "afterany"
    environment: List[str] = field(default_factory=list)
    defaults: SlurmStageResources = field(default_factory=SlurmStageResources)
    stages: Dict[str, SlurmStageResources] = field(default_factory=dict)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ChargesConfig:
    """Which charge model gives the features their atoms and charges.

    Attributes:
        use: CHARGES plug-in name.
        options: Per-plug-in settings, keyed by plug-in name.
    """

    use: str = "pdb2pqr"
    options: Dict[str, Dict] = field(default_factory=dict)


@dataclass(frozen=True)
class ContactsConfig:
    """Which detectors read protein-ligand contacts.

    Attributes:
        use: CONTACTS plug-in names; prolif reads its settings from ``interactions``.
        options: Per-plug-in settings, keyed by plug-in name.
    """

    use: List[str] = field(default_factory=lambda: ["prolif"])
    options: Dict[str, Dict] = field(default_factory=dict)


def _default_interactions() -> List[str]:
    """Interaction names requested from ProLIF by default.

    Returns:
        The detector's default list.
    """
    from qpocket.core.detect import INTERACTION_NAMES

    return list(INTERACTION_NAMES)


@dataclass(frozen=True)
class InteractionsConfig:
    """Which interactions to detect, and with what geometry.

    Attributes:
        enabled: ProLIF interaction names.
        parameters: Per-interaction geometry overrides, keyed by interaction name.
    """

    enabled: List[str] = field(default_factory=_default_interactions)
    parameters: Dict[str, Dict] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputeConfig:
    """How much of this machine a local run may use.

    Attributes:
        workers: Parallel workers: 0 uses half the cores, -1 all of them, N exactly N.
        backend: "process" (separate interpreters) or "thread" (shared memory).
    """

    workers: int = 0
    backend: Literal["process", "thread"] = "process"


@dataclass(frozen=True)
class SitesConfig:
    """Which ligand-occupied binding sites to discover.

    Attributes:
        min_heavy_atoms: Smaller HETATM groups are not ligands; carbon-free groups such as ions are never sites.
        link_cutoff: Distance (A) between any heavy atoms of two ligands under which they join one site.
        box_padding: Padding (A) added around the union of a site's ligands when boxing the site.
    """

    min_heavy_atoms: int = 6
    link_cutoff: float = 4.0
    box_padding: float = 4.0


@dataclass(frozen=True)
class QPocketConfig:
    """Aggregated configuration for QPocket pipeline execution."""
    mode: Literal["pdb_query", "fragment_screen"] = "pdb_query"
    reference: ReferenceParams = field(default_factory=ReferenceParams)
    sites: SitesConfig = field(default_factory=SitesConfig)
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
    charges: ChargesConfig = field(default_factory=ChargesConfig)
    contacts: ContactsConfig = field(default_factory=ContactsConfig)
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
    # Replaced by binding-site discovery: every ligand-occupied site is found.
    "reference.ligand_id": None,
    "reference.ligand_chain_id": None,
    "reference.pocket_mode": None,
    "reference.bounding_box": None,
    # Site runs write no pre-crop cloud, so there is nothing left to save.
    "io.save_intermediate": None,
    # The SLURM stages were rebuilt around sites. featurize, collect and distance
    # have no 1:1 successor whose base default would not mask the user's value.
    "slurm.split_apbs_stage": None,
    "slurm.stages.apbs": None,
    "slurm.stages.featurize": None,
    "slurm.stages.collect": None,
    "slurm.stages.geodesic": None,
    "slurm.stages.distance": None,
    # Bounded an automatic choice of k that no longer exists; a stage cuts by count, height or relative height.
    "distance.clustering.single.k_range": None,
    "distance.clustering.shape.k_range": None,
    "distance.clustering.chem.k_range": None,
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

        if replacement is None and legacy.startswith("slurm.stages."):
            warnings.append(
                f"{legacy} is obsolete; set slurm.stages.structure (per-structure chain), "
                "slurm.stages.site_featurize (features/APBS) or slurm.stages.site_cluster instead"
            )
        elif replacement is None:
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
    sites = cfg.sites
    if sites.min_heavy_atoms < 1:
        raise ValueError(f"sites.min_heavy_atoms must be >= 1, got: {sites.min_heavy_atoms}")
    if sites.link_cutoff <= 0:
        raise ValueError(f"sites.link_cutoff must be > 0, got: {sites.link_cutoff}")
    if sites.box_padding < 0:
        raise ValueError(f"sites.box_padding must be >= 0, got: {sites.box_padding}")

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

    if ref.interaction_padding <= 0:
        raise ValueError(
            f"reference.interaction_padding must be > 0, got: {ref.interaction_padding}"
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
    if clust.mode not in ("two_stage", "single", "shape_only"):
        raise ValueError(
            f"distance.clustering.mode must be 'two_stage', "
            f"'single' or 'shape_only', got: {clust.mode!r}"
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
            raise ValueError(f"{prefix}.k must be >= 2 when set, got: {stage.k}")
        if not (0.5 <= stage.damping < 1.0):
            raise ValueError(
                f"{prefix}.damping must be in [0.5, 1.0), got: {stage.damping}"
            )
        if stage.max_iter < 1:
            raise ValueError(
                f"{prefix}.max_iter must be >= 1, got: {stage.max_iter}"
            )
        criterion, threshold = stage.criterion, stage.threshold
        if stage.method == "affinity":  # chooses its own count; criterion and threshold are ignored
            continue
        if criterion not in ("count", "threshold", "relative"):
            raise ValueError(f"{prefix}.criterion must be 'count', 'threshold' or 'relative', "
                             f"got: {criterion!r}")
        if criterion == "count" and stage.k is None:
            raise ValueError(f"{prefix}.k must be set when criterion is 'count'; "
                             f"set k or use criterion 'threshold' or 'relative'")
        if criterion == "threshold" and (
                stage.method != "hierarchical" or threshold is None or threshold <= 0):
            raise ValueError(
                f"{prefix}.threshold must be > 0 with method 'hierarchical' when "
                f"criterion is 'threshold', got method {stage.method!r}, threshold {threshold!r}"
            )
        if criterion == "relative" and (
                stage.method != "hierarchical" or threshold is None or not 0 < threshold <= 1):
            raise ValueError(
                f"{prefix}.threshold must be in (0, 1] with method 'hierarchical' when "
                f"criterion is 'relative', got method {stage.method!r}, threshold {threshold!r}"
            )

    # --- FeaturesConfig ---
    from qpocket.core.ports import FEATURES as _features

    if not cfg.features.enabled:
        raise ValueError("features.enabled must name at least one provider")
    from qpocket.core.ports import CHARGES as _charges, CONTACTS as _contacts

    if not cfg.contacts.use:
        raise ValueError("contacts.use must name at least one detector")
    for section, key, registry, names in (
            ("features", "enabled", _features, list(cfg.features.enabled)),
            ("charges", "use", _charges, [cfg.charges.use]),
            ("contacts", "use", _contacts, list(cfg.contacts.use))):
        try:
            registry.resolve(names)
        except ValueError as exc:
            raise ValueError(f"{section}.{key}: {exc}") from exc
        options = getattr(cfg, section).options or {}
        idle = sorted(set(options) - set(names))
        if idle:
            raise ValueError(f"{section}.options: settings for {idle}, which are not in "
                             f"{section}.{key}")
        for name in names:
            try:
                plugin_params(registry, name, options)
            except ValueError as exc:
                raise ValueError(f"{section}.options: {exc}") from exc

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

    known_stages = ("prepare", "structure", "sites", "site_featurize", "site_cluster")
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


def _find_conf_dir() -> Path:
    """Locate the Hydra config directory.

    Returns:
        The first directory that contains ``config.yaml``.

    Raises:
        FileNotFoundError: If neither location has one, naming both.
    """
    packaged = Path(__file__).resolve().parent / "conf"
    repo_root = Path(__file__).resolve().parents[2] / "conf"

    for candidate in (packaged, repo_root):
        if (candidate / "config.yaml").exists():
            return candidate

    raise FileNotFoundError(
        "Q-Pocket's default configuration was not found. Looked in:\n"
        f"  {packaged / 'config.yaml'}\n"
        f"  {repo_root / 'config.yaml'}\n"
        "In a git checkout this usually means src/qpocket/conf/ was "
        "never committed."
    )


CONF_PATH = _find_conf_dir()


def load_qpocket_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
) -> QPocketConfig:
    """Loads, merges, and validates QPocket configuration.

    Args:
        config_path: Path to an external YAML configuration file.
        overrides: Dot-notation override strings (e.g. ``["mode=fragment_screen", "reference.pdb_id=6HP0"]``).

    Returns:
        Fully validated :class:`~qpocket.config.QPocketConfig`.

    Raises:
        ValueError: If the assembled configuration fails validation.
    """
    GlobalHydra.instance().clear()

    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONF_PATH.resolve()),
    ):
        cfg = compose(config_name="config", overrides=overrides)

    # Hydra composes in struct mode, which refuses unknown keys outright. A
    # config written against an older release carries keys this one has renamed,
    # and they have to survive the merge long enough for migrate_legacy to move
    # them; unrecognised leftovers are reported there instead.
    OmegaConf.set_struct(cfg, False)

    if config_path:
        user_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, user_cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    for notice in migrate_legacy(cfg):
        log_message(message_type="config_deprecation", notice=notice)
        print(f"config: {notice}", file=sys.stderr)

    fill_defaults(cfg)

    OmegaConf.set_readonly(cfg, True)
    final_cfg = QPocketConfig(
        mode=cfg.mode,
        paths=cfg.paths,
        reference=cfg.reference,
        sites=cfg.get("sites", SitesConfig()),
        fragment_screen=cfg.fragment_screen,
        pdb_query=cfg.pdb_query,
        alignment=cfg.alignment,
        cleaning=cfg.cleaning,
        io=cfg.io,
        distance=cfg.distance,
        # `slurm` is only read by the `qpocket slurm` commands. Falling back to
        # the dataclass default keeps a hand-written config that predates the
        # block loadable.
        compute=cfg.get("compute", ComputeConfig()),
        features=cfg.get("features", FeaturesConfig()),
        interactions=cfg.get("interactions", InteractionsConfig()),
        charges=cfg.get("charges", ChargesConfig()),
        contacts=cfg.get("contacts", ContactsConfig()),
        slurm=cfg.get("slurm", SlurmConfig()),
    )

    validate_config(final_cfg)
    return final_cfg


RUN_CONFIG = "config.yaml"


def save_run_config(cfg: QPocketConfig, ws) -> Path:
    """Snapshot the config a run used, so later per-site steps reuse it.

    Args:
        cfg: The run's configuration.
        ws: The run's Workspace.

    Returns:
        Path of ``metadata/config.yaml``.
    """
    path = Path(ws.metadata) / RUN_CONFIG
    path.write_text(str(cfg), encoding="utf-8")
    return path


def load_run_config(run_dir: Union[str, Path]) -> QPocketConfig:
    """The config a run saved, loaded and validated like a user's config file.

    Args:
        run_dir: The run directory.

    Returns:
        The validated configuration.
    """
    return load_qpocket_config(config_path=Path(run_dir) / "metadata" / RUN_CONFIG)




def sanitise_id(name: str) -> str:
    """Turn a filename into an identifier safe as a directory and a CSV key.

    Args:
        name: A file stem, or any label.

    Returns:
        The stem with every character outside ``[A-Za-z0-9_-]`` replaced by an underscore, truncated to 64 characters.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(name)).strip("_")[:64]
    return cleaned or "structure"


def run_id(cfg: QPocketConfig) -> str:
    """Name of this run's workspace directory.

    Args:
        cfg: QPocket configuration.

    Returns:
        A directory-safe identifier.

    Raises:
        ValueError: If neither a ``pdb_id`` nor a ``reference_path`` is set, so the run has no name at all.
    """
    if cfg.reference.pdb_id:
        return sanitise_id(cfg.reference.pdb_id)

    if cfg.mode == "fragment_screen" and cfg.fragment_screen.reference_path:
        return sanitise_id(Path(cfg.fragment_screen.reference_path).stem)

    raise ValueError(
        "the run has no name: set reference.pdb_id, or "
        "fragment_screen.reference_path in fragment_screen mode"
    )


def should_renumber(cfg: QPocketConfig) -> bool:
    """Whether to remap residue numbers onto UniProt positions."""
    configured = cfg.alignment.renumber_residues
    if configured is None:
        return cfg.mode != "fragment_screen"
    return bool(configured)


def evolve(cfg, path: str, value):
    """A copy of a config with one dotted path replaced.

    Args:
        cfg: A config dataclass or OmegaConf node.
        path: Dotted attribute path, e.g. "fragment_screen.reference_path".
        value: The new value.

    Returns:
        The changed copy; ``cfg`` is untouched.

    Raises:
        AttributeError: If a segment of ``path`` does not exist.
    """
    head, _, rest = path.partition(".")
    if isinstance(cfg, DictConfig):
        if head not in cfg:
            raise AttributeError(f"config has no {head!r}")
        copy = OmegaConf.create(OmegaConf.to_container(cfg))
        copy[head] = evolve(cfg[head], rest, value) if rest else value
        OmegaConf.set_readonly(copy, True)
        return copy
    if not hasattr(cfg, head):
        raise AttributeError(f"{type(cfg).__name__} has no {head!r}")
    new = evolve(getattr(cfg, head), rest, value) if rest else value
    return dataclasses.replace(cfg, **{head: new})


def plugin_params(registry, name: str, options):
    """A plug-in's params, built from its entry in an options mapping.

    Args:
        registry: The port's Registry.
        name: Plug-in name.
        options: Mapping of plug-in name to settings.

    Returns:
        An instance of the plug-in's params class.

    Raises:
        ValueError: If a setting is not one of the plug-in's params.
    """
    plugin = registry.get(name)
    given = dict((options or {}).get(name) or {})
    allowed = {f.name for f in fields(plugin.params)}
    unknown = sorted(set(given) - allowed)
    if unknown:
        raise ValueError(f"{registry.kind} {name}: unknown option(s) {unknown}; "
                         f"allowed: {sorted(allowed)}")
    return plugin.params(**given)
