"""Two-dimensional layouts of a precomputed pocket distance matrix."""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np

# ``symmetrise`` is a distance-matrix utility, not an embedding one, so it lives
# in the layer the pipeline can import. Re-exported here for the notebook.
from qpocket.core.clustering import symmetrise

#: Layout names, in the order a UI should offer them.
METHODS: Tuple[str, ...] = ("PCA (classical MDS)", "UMAP", "t-SNE")

PCA_MDS, UMAP, TSNE = METHODS

#: Fixed seed, so a layout is reproducible between runs of the notebook.
RANDOM_STATE = 42


class MDSResult(NamedTuple):
    """Classical MDS of a distance matrix.

    Attributes:
        coords: ``(n, 2)`` embedding.
        eigenvalues: All eigenvalues, descending, for a scree plot.
        variance_share: Share of the positive eigenvalue mass on each of the two axes.
        non_euclidean_share: Negative eigenvalue mass over positive mass.
    """

    coords: np.ndarray
    eigenvalues: np.ndarray
    variance_share: np.ndarray
    non_euclidean_share: float


class Embedding(NamedTuple):
    """A finished layout, ready to plot.

    Attributes:
        coords: ``(n, 2)`` embedding.
        axis_labels: Axis names, carrying variance shares where they exist.
        note: One line on how to read this layout, including its parameters.
        method: Which layout produced it.
        mds: The full MDS result, when that is what ran.
    """

    coords: np.ndarray
    axis_labels: Tuple[str, str]
    note: str
    method: str
    mds: Optional[MDSResult] = None




def classical_mds(distances: np.ndarray, n_components: int = 2) -> MDSResult:
    """Embed a distance matrix by classical MDS, the PCA equivalent for distances.

    Args:
        distances: Symmetric distance matrix.
        n_components: Number of axes to keep.

    Returns:
        Coordinates plus the eigenvalue diagnostics.
    """
    d = symmetrise(distances)
    n = len(d)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (d**2) @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    positive = eigenvalues[eigenvalues > 0]
    keep = min(n_components, len(positive))
    coords = np.zeros((n, n_components))
    coords[:, :keep] = eigenvectors[:, :keep] * np.sqrt(eigenvalues[:keep])

    # Eigenvector signs are arbitrary; pin them so the picture does not flip.
    for axis in range(n_components):
        if coords[np.argmax(np.abs(coords[:, axis])), axis] < 0:
            coords[:, axis] *= -1

    total = positive.sum()
    share = eigenvalues[:n_components].clip(min=0) / total if total else np.zeros(n_components)
    negative = -eigenvalues[eigenvalues < 0].sum()
    return MDSResult(coords, eigenvalues, share, float(negative / total) if total else 0.0)


def max_perplexity(n_samples: int) -> float:
    """Largest perplexity worth offering for ``n_samples`` points.

    Args:
        n_samples: Number of structures being embedded.

    Returns:
        An upper bound of at least 2.
    """
    return max(2.0, (n_samples - 1) / 3.0)


def embed_distances(
    distances: np.ndarray,
    method: str = UMAP,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: float = 15.0,
    random_state: int = RANDOM_STATE,
) -> Embedding:
    """Lay out a precomputed distance matrix in two dimensions.

    Args:
        distances: Square distance matrix.
        method: One of :data:`METHODS`.
        n_neighbors: UMAP neighbourhood size.
        min_dist: UMAP minimum separation.
        perplexity: t-SNE perplexity, capped by :func:`max_perplexity`.
        random_state: Seed for the stochastic methods.

    Returns:
        The layout, its axis labels and a line on how to read it.

    Raises:
        ValueError: If ``method`` is unknown, or t-SNE is asked for with fewer than four structures.
    """
    d = symmetrise(distances)
    n = len(d)

    if method == PCA_MDS:
        mds = classical_mds(d)
        share = mds.variance_share
        labels = (
            f"PC1 ({share[0]:.0%} of variance)",
            f"PC2 ({share[1]:.0%} of variance)",
        )
        note = f"These two axes hold {share[:2].sum():.0%} of the distance structure"
        note += "; the rest cannot be seen in 2D." if share[:2].sum() < 0.9 else "."
        if mds.non_euclidean_share > 0.05:
            note += (
                f" Non-Euclidean residue: {mds.non_euclidean_share:.0%} of the"
                " positive eigenvalue mass."
            )
        return Embedding(mds.coords, labels, note, method, mds)

    if method == TSNE:
        if n < 4:
            raise ValueError("t-SNE needs at least 4 structures")
        from sklearn.manifold import TSNE as _TSNE

        used = float(min(perplexity, max(2.0, n - 1.5)))
        coords = _TSNE(
            n_components=2,
            metric="precomputed",
            init="random",  # init="pca" is rejected for precomputed metrics
            perplexity=used,
            random_state=random_state,
        ).fit_transform(d)
        note = (
            f"perplexity {used:.1f}. Only which points group together carries"
            " meaning; the distance between groups does not."
        )
        return Embedding(np.asarray(coords, dtype=float), ("t-SNE 1", "t-SNE 2"), note, method)

    if method == UMAP:
        import umap as _umap

        coords = _umap.UMAP(
            n_components=2,
            metric="precomputed",
            n_neighbors=int(min(max(2, n_neighbors), max(2, n - 1))),
            min_dist=float(min_dist),
            random_state=random_state,
        ).fit_transform(d)
        note = (
            f"n_neighbors {int(n_neighbors)}, min_dist {min_dist:g}. Groups are"
            " meaningful, the gaps between them are not."
        )
        return Embedding(np.asarray(coords, dtype=float), ("UMAP 1", "UMAP 2"), note, method)

    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
