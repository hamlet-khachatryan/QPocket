"""PDB query: find similar entries on RCSB, then the conformational stages."""
from pathlib import Path

from qpocket.config import QPocketConfig
from qpocket.pipelines import stages

STAGES = (
    stages.download_structures,
    # One fan-out: each structure goes from file to geodesic on its own worker.
    stages.prepare_and_featurize,
    stages.compute_spectral_distance_matrix,
)


def run(cfg: QPocketConfig) -> Path:
    """Run every stage.

    Args:
        cfg: Validated config with mode pdb_query.

    Returns:
        Run root directory.
    """
    return stages.execute(cfg, STAGES)
