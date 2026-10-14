"""Screening configuration dataclasses and loaders."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import OmegaConf


@dataclass
class ChemblDatasetConfig:
    """Where the ChEMBL decoy pool is downloaded from and cached."""
    url: str = "PLACEHOLDER_GOOGLE_DRIVE_URL"
    local_dir: str = "~/.qpocket/chembl_data"


@dataclass
class DecoyConfig:
    """Decoy strategy and its property-matching limits."""
    strategy: str = "lude"  # DECOYS plug-in
    n_decoys_per_active: int = 50
    max_similarity: float = 0.2
    mw_diff: float = 20.0
    logp_diff: float = 0.5
    rb_diff: int = 1
    hba_diff: int = 1
    hbd_diff: int = 1
    charge_diff: int = 0


@dataclass
class BoxConfig:
    """Docking box: from the ligands or the cavities, plus padding (A)."""
    mode: str = "ligand_based"  # ligand_based | cavity_based
    padding: float = 5.0


@dataclass
class PreparationConfig:
    """Ligand protonation pH and conformer generation."""
    ph: float = 7.4
    num_conformers: int = 50
    max_iters: int = 500


@dataclass
class DockingConfig:
    """Docking engine and its search settings."""
    engine: str = "quickvina"  # DOCKING plug-in
    engine_path: Optional[str] = None  # default: the engine's own executable
    exhaustiveness: int = 8
    num_modes: int = 10
    energy_range: float = 3.0
    cpus: int = 4
    seed: int = 42
    custom_engines: Dict[str, Dict] = field(default_factory=dict)  # any binary, by command template


@dataclass
class SlurmConfig:
    """SLURM resources for docking arrays."""
    enabled: bool = False
    partition: str = "standard"
    cpus_per_task: int = 8
    mem: str = "4G"
    time: str = "02:00:00"
    max_concurrent: int = 100
    max_array_size: int = 1000
    # Lines run before each job, e.g. module loads or `source .venv/bin/activate`.
    environment: List[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """Pose RMSD threshold and enrichment metric settings."""
    rmsd_threshold: float = 2.0
    top_k: int = 3
    bedroc_alpha: float = 20.0
    ef_fractions: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.03, 0.05, 0.1])


@dataclass
class ReceptorConfig:
    """How representatives become docking receptors."""
    repair: bool = True          # PDBFixer: missing side-chain atoms, nearby loops
    loop_radius: float = 10.0    # build a missing loop only this close (A) to the pocket
    use: str = "pdb2pqr"         # RECEPTORS plug-in: protonation and partial charges
    options: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class CrossdockConfig:
    """The one search box every cross-docking run shares."""
    box_mode: str = "representative_pockets"  # or interacting_residues
    box_padding: float = 5.0


@dataclass
class ScreenConfig:
    """Full screening configuration."""
    run_mode: str = "cross_docking"  # cross_docking | virtual_screening
    chembl: ChemblDatasetConfig = field(default_factory=ChemblDatasetConfig)
    decoy: DecoyConfig = field(default_factory=DecoyConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    receptor: ReceptorConfig = field(default_factory=ReceptorConfig)
    crossdock: CrossdockConfig = field(default_factory=CrossdockConfig)


def load_screen_config(path: Optional[str] = None, **overrides) -> ScreenConfig:
    """Load ScreenConfig from a YAML file, optionally applying overrides.

    Args:
        path: Path to a YAML config file.
        **overrides: Dot-separated key=value pairs that override any field (e.g. ``run_mode="virtual_screening"``).

    Returns:
        A fully-populated :class:`ScreenConfig` instance.
    """
    base = OmegaConf.structured(ScreenConfig)

    if path is not None:
        file_cfg = OmegaConf.load(path)
        base = OmegaConf.merge(base, file_cfg)

    if overrides:
        override_cfg = OmegaConf.create(overrides)
        base = OmegaConf.merge(base, override_cfg)

    return OmegaConf.to_object(base)  # type: ignore[return-value]


def validate_screen_config(cfg: ScreenConfig) -> None:
    """Validate a :class:`ScreenConfig` and raise ``ValueError`` on errors."""
    errors: List[str] = []

    if cfg.run_mode not in ("cross_docking", "virtual_screening"):
        errors.append(f"run_mode must be 'cross_docking' or 'virtual_screening', got '{cfg.run_mode}'")

    if cfg.decoy.n_decoys_per_active < 1:
        errors.append("decoy.n_decoys_per_active must be >= 1")

    if not (0.0 < cfg.decoy.max_similarity <= 1.0):
        errors.append("decoy.max_similarity must be in (0, 1]")

    if cfg.box.mode not in ("ligand_based", "cavity_based"):
        errors.append(f"box.mode must be 'ligand_based' or 'cavity_based', got '{cfg.box.mode}'")

    if cfg.box.padding < 0:
        errors.append("box.padding must be >= 0")

    if not (0.0 < cfg.preparation.ph <= 14.0):
        errors.append("preparation.ph must be in (0, 14]")

    if cfg.preparation.num_conformers < 1:
        errors.append("preparation.num_conformers must be >= 1")

    from qpocket.core.ports import DECOYS

    if cfg.decoy.strategy not in DECOYS.available():
        errors.append(f"decoy.strategy must be one of {list(DECOYS.available())}, "
                      f"got '{cfg.decoy.strategy}'")

    from qpocket.core.ports import DOCKING
    from qpocket.core.screening import PLACEHOLDERS, template_fields

    for name, spec in (cfg.docking.custom_engines or {}).items():
        where = f"docking.custom_engines.{name}"
        if name in DOCKING.available():
            errors.append(f"{where}: '{name}' is a built-in engine; choose another name")
        missing = {"executable", "command"} - set(spec)
        if missing:
            errors.append(f"{where}: missing {sorted(missing)}")
            continue
        used = template_fields(str(spec["command"]))
        for required in ("receptor", "ligand", "output"):
            if required not in used:
                errors.append(f"{where}.command must contain {{{required}}}")
        unknown = sorted(used - set(PLACEHOLDERS))
        if unknown:
            errors.append(f"{where}.command uses unknown placeholders {unknown}; "
                          f"allowed: {list(PLACEHOLDERS)}")
        if spec.get("output", "pdbqt") not in ("pdbqt", "sdf"):
            errors.append(f"{where}.output must be 'pdbqt' or 'sdf'")
    known = list(DOCKING.available()) + list(cfg.docking.custom_engines or {})
    if cfg.docking.engine not in known:
        errors.append(f"docking.engine must be one of {known}, got '{cfg.docking.engine}'")

    if cfg.docking.exhaustiveness < 1:
        errors.append("docking.exhaustiveness must be >= 1")

    if cfg.docking.num_modes < 1:
        errors.append("docking.num_modes must be >= 1")

    if cfg.analysis.rmsd_threshold <= 0:
        errors.append("analysis.rmsd_threshold must be > 0")

    if cfg.analysis.bedroc_alpha <= 0:
        errors.append("analysis.bedroc_alpha must be > 0")

    for frac in cfg.analysis.ef_fractions:
        if not (0.0 < frac <= 1.0):
            errors.append(f"analysis.ef_fractions contains invalid value {frac} (must be in (0, 1])")

    if cfg.crossdock.box_mode not in ("representative_pockets", "interacting_residues"):
        errors.append("crossdock.box_mode must be 'representative_pockets' or 'interacting_residues', "
                      f"got '{cfg.crossdock.box_mode}'")
    if cfg.crossdock.box_padding < 0:
        errors.append("crossdock.box_padding must be >= 0")
    if cfg.receptor.loop_radius < 0:
        errors.append("receptor.loop_radius must be >= 0")
    if cfg.analysis.top_k < 1:
        errors.append("analysis.top_k must be >= 1")

    if errors:
        raise ValueError("Invalid screen config:\n" + "\n".join(f"  - {e}" for e in errors))
