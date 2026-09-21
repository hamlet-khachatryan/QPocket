"""Visualisation helpers behind the Q-Pocket explorer notebook.

The notebook at ``experiments/visualize_qpocket.py`` is the user interface; the
logic it needs lives here, where pytest can reach it:

:mod:`~qpocket_pipeline.viz.pdb_text`
    Reading structures and emitting column-correct PDB for a browser viewer.
:mod:`~qpocket_pipeline.viz.ramps`
    Per-feature colour ramps, and the gradient maths a legend has to mirror.
:mod:`~qpocket_pipeline.viz.embed`
    Two-dimensional layouts of a precomputed distance matrix.
:mod:`~qpocket_pipeline.viz.cluster`
    Re-clustering that matrix, and comparing the result to the stored labels.
:mod:`~qpocket_pipeline.viz.pockets`
    Exact set arithmetic over pocket clouds: occupancy, overlap, field differences.
:mod:`~qpocket_pipeline.viz.assets`
    The vendored 3Dmol.js build, for pages that must work offline.

Nothing here imports plotly, marimo, umap or scikit-learn at module level, so the
package stays importable without the notebook's own dependencies.
"""

from __future__ import annotations

from qpocket_pipeline.viz.assets import (
    THREEDMOL_CDN,
    THREEDMOL_SHA256,
    THREEDMOL_VERSION,
    threedmol_path,
    threedmol_script,
    threedmol_source,
)
from qpocket_pipeline.viz.cluster import (
    AFFINITY,
    CRITERIA,
    LINKAGES,
    PIPELINE_DAMPING,
    PIPELINE_PREFERENCE_PCT,
    Clustering,
    ClusterComparison,
    DendrogramData,
    affinity_labels,
    block_order,
    cluster_labels,
    compare_labelings,
    dendrogram_segments,
    exemplars,
    linkage_matrix,
    silhouette,
    threshold_for_clusters,
)
from qpocket_pipeline.viz.cluster import METHODS as CLUSTER_METHODS
from qpocket_pipeline.viz.embed import (
    METHODS as EMBED_METHODS,
    Embedding,
    MDSResult,
    classical_mds,
    embed_distances,
    max_perplexity,
    symmetrise,
)
from qpocket_pipeline.viz.pdb_text import (
    BACKBONE,
    ProteinModels,
    atom_line,
    bbox_corners,
    blocks_to_text,
    ligand_pdb,
    parse_atom,
    pocket_context_models,
    pseudo_atom_pdb,
    read_protein_atoms,
    residue_model,
)
from qpocket_pipeline.viz.pockets import (
    LATTICE_TOLERANCE,
    FieldDifference,
    Occupancy,
    PocketMetrics,
    field_difference,
    infer_spacing,
    jaccard,
    lattice_deviation,
    occupancy,
    overlap_matrix,
    pocket_metrics,
    shares_lattice,
    volume_table,
    voxel_index,
)
from qpocket_pipeline.viz.ramps import (
    DEFAULT_RAMP,
    FEATURE_RAMP,
    N_STOPS,
    PERCEPTUAL_SCHEME,
    RAMP_TABLES,
    SCHEMES,
    ColourDomain,
    RampSpec,
    colour_domain,
    feature_ramp,
    gradient_spec,
    legend_gradient,
    linear_hex,
    ramp_stops,
    scale_to_percent,
)

__all__ = [
    "AFFINITY",
    "BACKBONE",
    "CLUSTER_METHODS",
    "CRITERIA",
    "ClusterComparison",
    "Clustering",
    "ColourDomain",
    "DEFAULT_RAMP",
    "DendrogramData",
    "EMBED_METHODS",
    "Embedding",
    "FEATURE_RAMP",
    "FieldDifference",
    "LATTICE_TOLERANCE",
    "LINKAGES",
    "MDSResult",
    "N_STOPS",
    "Occupancy",
    "PERCEPTUAL_SCHEME",
    "PIPELINE_DAMPING",
    "PIPELINE_PREFERENCE_PCT",
    "PocketMetrics",
    "ProteinModels",
    "RAMP_TABLES",
    "RampSpec",
    "SCHEMES",
    "THREEDMOL_CDN",
    "THREEDMOL_SHA256",
    "THREEDMOL_VERSION",
    "affinity_labels",
    "atom_line",
    "bbox_corners",
    "block_order",
    "blocks_to_text",
    "classical_mds",
    "cluster_labels",
    "colour_domain",
    "compare_labelings",
    "dendrogram_segments",
    "embed_distances",
    "exemplars",
    "feature_ramp",
    "field_difference",
    "gradient_spec",
    "infer_spacing",
    "jaccard",
    "lattice_deviation",
    "legend_gradient",
    "ligand_pdb",
    "linear_hex",
    "linkage_matrix",
    "max_perplexity",
    "occupancy",
    "overlap_matrix",
    "parse_atom",
    "pocket_context_models",
    "pocket_metrics",
    "pseudo_atom_pdb",
    "ramp_stops",
    "read_protein_atoms",
    "residue_model",
    "scale_to_percent",
    "shares_lattice",
    "silhouette",
    "symmetrise",
    "threedmol_path",
    "threedmol_script",
    "threedmol_source",
    "threshold_for_clusters",
    "volume_table",
    "voxel_index",
]
