"""Screening configuration dataclasses and loaders."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

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
    engine: str = "vina"  # vina | quickvina
    engine_path: str = "vina"
    exhaustiveness: int = 8
    num_modes: int = 10
    energy_range: float = 3.0
    cpus: int = 4


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
    bedroc_alpha: float = 20.0
    ef_fractions: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.03, 0.05, 0.1])


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

    if cfg.docking.engine not in ("vina", "quickvina"):
        errors.append(f"docking.engine must be 'vina' or 'quickvina', got '{cfg.docking.engine}'")

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

    if errors:
        raise ValueError("Invalid screen config:\n" + "\n".join(f"  - {e}" for e in errors))
