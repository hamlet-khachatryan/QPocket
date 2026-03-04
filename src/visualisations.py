"""
visualisation.py

A module for generating high-quality, publication-ready visualizations
of protein structural and interaction metadata.
Designed for ACS Journal compatibility (minimalistic, clear, high-DPI).
"""

import hashlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Union

# Dimensionality Reduction
from sklearn.manifold import TSNE
try:
    import umap
except ImportError:
    umap = None

# Statistics
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
from scipy.sparse import csr_matrix

# --- ACS / Datawrapper Style Configuration ---
def set_clean_style():
    """Sets matplotlib parameters for a clean, Datawrapper-inspired aesthetic."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,

        # Datawrapper-like cleanliness
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,  # Often minimal y-axis
        'axes.spines.bottom': True,
        'axes.linewidth': 0.8,
        'axes.grid': True,          # Light grid for reference
        'grid.color': '#dedede',
        'grid.alpha': 0.6,
        'grid.linestyle': '-',
        'axes.axisbelow': True,     # Grid behind data

        'xtick.bottom': True,
        'ytick.left': False,        # Minimal y-ticks
        'xtick.major.size': 3,
        'ytick.major.size': 0,

        'pdf.fonttype': 42
    })

set_clean_style()

# --- Helper Functions ---

def _get_project_paths(base_dir: Union[str, Path]) -> dict:
    base = Path(base_dir)
    return {
        'metadata': base / 'metadata',
        'matrices': base / 'metadata' / 'spectral_distance_matrix.csv',
        'clustering': base / 'metadata' / 'clustering_metadata.csv',
        'transport': base / 'metadata' / 'transport_matrices', # Adjusted based on your error path
        'obs_freq': base / 'metadata' / 'observed_interactions_freq.csv',
        'pot_freq': base / 'metadata' / 'potential_interactions_freq.csv',
        'utilization': base / 'metadata' / 'binding_pocket_utilization.csv',
        'plots': base / 'metadata' / 'plots'
    }

def _get_custom_palette(n_colors: int = 12) -> List[tuple]:
    """
    Returns the specific XGFS Normal 12 palette provided by the user.
    """
    # RGB values (0-255)
    xgfs_normal12 = [
        (235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0),
        (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135),
        (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108),
        (189, 189, 189)
    ]
    # Convert to 0-1 for Matplotlib
    palette = [(r/255., g/255., b/255.) for r, g, b in xgfs_normal12]

    # If we need more colors than available, cycle or extend
    if n_colors > len(palette):
        return sns.color_palette("tab20", n_colors)

    return palette[:n_colors]

def _get_consistent_colors(columns: List[str], seed: int = 42) -> List[tuple]:
    """
    Maps column names to the custom palette consistently using hashing.
    """
    base_palette = _get_custom_palette(len(columns) + 5)

    colors = []
    for i, col in enumerate(columns):
        if i < len(base_palette):
            colors.append(base_palette[i])
        else:
            # Deterministic hash fallback
            hash_val = int(hashlib.sha256(str(col).encode('utf-8')).hexdigest(), 16)
            color_idx = (hash_val + seed) % len(base_palette)
            colors.append(base_palette[color_idx])
    return colors

def _save_plot(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path.with_suffix('.png')), dpi=300, bbox_inches='tight')
    fig.savefig(str(output_path.with_suffix('.pdf')), format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {output_path.name}")

# ==================================================================================
#                                PLOTTING FUNCTIONS
# ==================================================================================

def plot_fgw_heatmap(project_dir: str, method: str = 'average'):
    """
    Plots a clustered heatmap of the FGW distance matrix.
    """
    paths = _get_project_paths(project_dir)
    if not paths['matrices'].exists():
        print(f"Matrix file not found at {paths['matrices']}")
        return

    # Adapt loading based on file extension
    if paths['matrices'].suffix == '.csv':
        # Load CSV (assuming index_col=0 contains labels)
        df_temp = pd.read_csv(paths['matrices'], index_col=0)
        matrix = df_temp.values
    else:
        matrix = np.load(paths['matrices'])

    if paths['clustering'].exists():
        meta = pd.read_csv(paths['clustering'])
        labels = meta['pdb_id'].values
    else:
        # Fallback to CSV labels or generic indices
        if paths['matrices'].suffix == '.csv':
            labels = df_temp.index.astype(str)
        else:
            labels = [f"P{i}" for i in range(matrix.shape[0])]

    df_mat = pd.DataFrame(matrix, index=labels, columns=labels)
    linkage = hc.linkage(sp.distance.squareform(matrix), method=method)

    # Clean Heatmap Style
    g = sns.clustermap(
        df_mat,
        row_linkage=linkage,
        col_linkage=linkage,
        cmap='viridis_r',
        figsize=(8, 8),
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        xticklabels=False,
        yticklabels=True
    )

    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_cbar.set_title("FGW\nDist", fontsize=9)
    g.ax_heatmap.tick_params(axis='y', labelsize=6)

    _save_plot(g.fig, paths['plots'] / "fgw_heatmap.png")


def plot_conformational_space(project_dir: str, mode: str = 'umap', n_neighbors: int = 15):
    """
    Projects the distance matrix into 2D space.
    Minimalist style: No axis ticks, just data and structure.
    """
    paths = _get_project_paths(project_dir)
    if not paths['matrices'].exists() or not paths['clustering'].exists():
        print("Missing matrix or clustering metadata.")
        return

    # Adapt loading based on file extension
    if paths['matrices'].suffix == '.csv':
        matrix = pd.read_csv(paths['matrices'], index_col=0).values
    else:
        matrix = np.load(paths['matrices'])

    meta = pd.read_csv(paths['clustering'])

    clusters = meta['cluster_id'].values
    is_rep = meta['is_representative'].values

    # Projection Logic
    if mode == 'tsne':
        coords = TSNE(n_components=2, metric='precomputed', init='random', random_state=42).fit_transform(matrix)
    elif mode == 'pca':
        from sklearn.manifold import MDS
        coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(matrix)
    elif mode == 'umap':
        if umap is None:
            print("UMAP not installed. Fallback to t-SNE.")
            plot_conformational_space(project_dir, 'tsne')
            return
        coords = umap.UMAP(metric='precomputed', n_neighbors=n_neighbors, random_state=42).fit_transform(matrix)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 5))

    # Get custom palette for clusters
    unique_clusters = np.unique(clusters)
    palette = _get_custom_palette(len(unique_clusters))

    # Map cluster IDs to colors
    cluster_colors = [palette[c % len(palette)] for c in clusters]

    # 1. Plot Conformers (Small dots)
    ax.scatter(
        coords[~is_rep, 0], coords[~is_rep, 1],
        c=np.array(cluster_colors)[~is_rep],
        s=40, alpha=0.7, linewidth=0, label='Conformer'
    )

    # 2. Plot Representatives (Diamonds with outline)
    ax.scatter(
        coords[is_rep, 0], coords[is_rep, 1],
        c=np.array(cluster_colors)[is_rep],
        s=100, edgecolors='black', linewidth=1.2, marker='D', label='Representative'
    )

    # Clean Aesthetics (Remove Ticks/Spines)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_title(f"Conformational Landscape ({mode.upper()})", loc='left', pad=15, fontsize=14, fontweight='bold')

    # Simple Legend
    # Create custom handles for legend to show Cluster Colors
    legend_handles = []
    for i, c_id in enumerate(unique_clusters):
        color = palette[i % len(palette)]
        legend_handles.append(mpatches.Patch(color=color, label=f'Cluster {c_id}'))

    # Add Rep marker to legend
    legend_handles.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=8, markeredgecolor='k', label='Representative'))

    ax.legend(handles=legend_handles, loc='best', frameon=False, fontsize=9)

    _save_plot(fig, paths['plots'] / f"space_{mode}.png")


def plot_transport_matrix(project_dir: str, idx_i: int, idx_j: int):
    """
    Visualizes pairwise Transport Plan.
    FIXED: Loads from .npz component arrays instead of load_npz().
    """
    paths = _get_project_paths(project_dir)

    # Try finding file i_j or j_i
    t_file = paths['transport'] / f"transport_{idx_i}_{idx_j}.npz"
    if not t_file.exists():
        t_file = paths['transport'] / f"transport_{idx_j}_{idx_i}.npz"
        if not t_file.exists():
            print(f"Transport file {t_file.name} not found.")
            return

    # --- BUG FIX START ---
    try:
        # Load the custom compressed format from geofgw_benchmark.py
        with np.load(t_file) as data:
            # Reconstruct CSR matrix
            matrix_csr = csr_matrix(
                (data['matrix'], data['indices'], data['indptr']),
                shape=data['shape']
            )
            matrix = matrix_csr.toarray()
    except Exception as e:
        print(f"Error loading transport matrix: {e}")
        return
    # --- BUG FIX END ---

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap='magma', aspect='auto', interpolation='nearest')

    ax.set_title(f"Transport Plan: {idx_i} $\\to$ {idx_j}", loc='left', fontsize=12, fontweight='bold')
    ax.set_xlabel(f"Target Voxels ({idx_j})")
    ax.set_ylabel(f"Source Voxels ({idx_i})")

    # Clean axes
    ax.grid(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)

    _save_plot(fig, paths['plots'] / f"transport_{idx_i}_{idx_j}.png")


def plot_interaction_frequencies(project_dir: str, mode: str = "observed"):
    """
    Stacked bar chart using the custom palette.
    """
    paths = _get_project_paths(project_dir)
    file_key = 'obs_freq' if mode == 'observed' else 'pot_freq'
    csv_path = paths[file_key]

    if not csv_path.exists():
        print(f"Frequency file {csv_path} missing.")
        return

    # Load
    df = pd.read_csv(csv_path, index_col=0)

    if mode == 'potential':
        # Simple Bar
        df_plot = df[['Frequency (%)']].sort_values('Frequency (%)', ascending=False).head(25)
        colors = _get_custom_palette(1)[2] # Use the 'Blue' from palette

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(df_plot.index, df_plot['Frequency (%)'], color=colors, width=0.7)

    else:
        # Stacked Bar
        totals = df.sum(axis=1).sort_values(ascending=False).head(20)
        df_plot = df.loc[totals.index]
        colors = _get_consistent_colors(df_plot.columns)

        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot.plot(kind='bar', stacked=True, width=0.7, color=colors, ax=ax, legend=True)

        # Legend styling
        ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', frameon=False, title="Interaction Type")

    # Annotations & Style
    title_text = "Observed Interactions" if mode == 'observed' else "Potential Pocket Residues"
    ax.set_title(title_text, loc='left', fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Frequency (%)")
    ax.tick_params(axis='x', rotation=45)

    # Remove y-axis line, keep grid
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True, color='#dedede', linestyle='-')
    ax.set_axisbelow(True)

    plt.tight_layout()
    _save_plot(fig, paths['plots'] / f"freq_{mode}.png")


def plot_pocket_utilization(project_dir: str):
    """
    Bar plot of Utilization %.
    """
    paths = _get_project_paths(project_dir)
    if not paths['utilization'].exists():
        print("Utilization file missing.")
        return

    df = pd.read_csv(paths['utilization'])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use the 'Teal' from custom palette
    color = _get_custom_palette(1)[4]

    if len(df) > 40:
        # Histogram for many structures
        sns.histplot(df['utilization_percentage'], kde=True, color=color, ax=ax, edgecolor='white')
        ax.set_ylabel("Count")
        ax.set_xlabel("Pocket Utilization (%)")
    else:
        # Bar plot for fewer
        df = df.sort_values('utilization_percentage', ascending=False)
        sns.barplot(x='pdb_id', y='utilization_percentage', data=df, color=color, ax=ax)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Utilization (%)")
        ax.set_xlabel("")

        # Add labels on top
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.0f}',
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', xytext=(0, 3),
                           textcoords='offset points', fontsize=8)

    ax.set_title("Binding Pocket Utilization", loc='left', fontsize=16, pad=20, fontweight='bold')
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True, color='#dedede')

    plt.tight_layout()
    _save_plot(fig, paths['plots'] / "utilization.png")

# ==================================================================================
#                                EXECUTION
# ==================================================================================
