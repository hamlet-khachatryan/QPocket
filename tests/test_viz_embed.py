"""Tests for qpocket.viz.embed and the vendored viewer asset.

Layouts of a distance matrix are easy to get subtly wrong: a flipped eigenvector
makes the same data look different between runs, and a parameter scikit-learn
rejects only shows up at call time. These tests pin the invariants that make two
runs of the notebook comparable.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from qpocket.viz.assets import (
    THREEDMOL_CDN,
    THREEDMOL_SHA256,
    THREEDMOL_VERSION,
    threedmol_path,
    threedmol_script,
    threedmol_source,
)
from qpocket.viz.embed import (
    METHODS,
    PCA_MDS,
    TSNE,
    UMAP,
    classical_mds,
    embed_distances,
    max_perplexity,
    symmetrise,
)


def distances_from_points(points):
    points = np.asarray(points, dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(-1))


@pytest.fixture
def two_groups():
    """Two well-separated groups of points, and their group labels."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.3, size=(8, 3))
    b = rng.normal(10.0, 0.3, size=(8, 3))
    labels = np.array([0] * 8 + [1] * 8)
    return distances_from_points(np.vstack([a, b])), labels


def group_separation(coords, labels):
    """Mean within-group distance over mean between-group distance."""
    d = distances_from_points(coords)
    iu = np.triu_indices(len(coords), 1)
    same = labels[iu[0]] == labels[iu[1]]
    return d[iu][same].mean() / d[iu][~same].mean()


# ---------------------------------------------------------------------------
# symmetrise
# ---------------------------------------------------------------------------


def test_symmetrise_averages_the_two_triangles():
    d = np.array([[0.0, 1.0], [3.0, 0.0]])
    out = symmetrise(d)
    assert out[0, 1] == out[1, 0] == pytest.approx(2.0)


def test_symmetrise_zeroes_the_diagonal():
    d = np.array([[0.5, 1.0], [1.0, 0.5]])
    assert np.diag(symmetrise(d)) == pytest.approx([0.0, 0.0])


def test_symmetrise_rejects_non_square_input():
    with pytest.raises(ValueError, match="square"):
        symmetrise(np.zeros((2, 3)))


def test_symmetrise_does_not_mutate_its_argument():
    d = np.array([[0.0, 1.0], [3.0, 0.0]])
    symmetrise(d)
    assert d[1, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# classical MDS
# ---------------------------------------------------------------------------


def test_mds_recovers_planar_geometry():
    """Points already in a plane should come back with their distances intact."""
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
    result = classical_mds(distances_from_points(points))
    assert distances_from_points(result.coords) == pytest.approx(distances_from_points(points), abs=1e-6)
    assert result.variance_share.sum() == pytest.approx(1.0, abs=1e-6)
    assert result.non_euclidean_share == pytest.approx(0.0, abs=1e-9)


def test_mds_variance_share_reports_lost_structure():
    """Genuinely 3D data cannot be fully shown on two axes."""
    points = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    result = classical_mds(distances_from_points(points))
    assert 0.0 < result.variance_share.sum() < 0.95


def test_mds_flags_non_euclidean_distances():
    """A metric that breaks the triangle inequality gets negative eigenvalues."""
    d = np.array(
        [
            [0.0, 1.0, 1.0, 9.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [9.0, 1.0, 1.0, 0.0],
        ]
    )
    assert classical_mds(d).non_euclidean_share > 0.05


def test_mds_signs_are_pinned_so_the_picture_is_stable():
    points = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 1.0], [2.0, -2.0]])
    d = distances_from_points(np.pad(points, ((0, 0), (0, 1))))
    first = classical_mds(d).coords
    second = classical_mds(d).coords
    assert first == pytest.approx(second)
    for axis in range(2):
        assert first[np.argmax(np.abs(first[:, axis])), axis] > 0


def test_mds_keeps_groups_together(two_groups):
    d, labels = two_groups
    assert group_separation(classical_mds(d).coords, labels) < 0.5


def test_mds_eigenvalues_are_descending():
    rng = np.random.default_rng(1)
    d = distances_from_points(rng.normal(size=(10, 4)))
    values = classical_mds(d).eigenvalues
    assert np.all(np.diff(values) <= 1e-9)


# ---------------------------------------------------------------------------
# embed_distances
# ---------------------------------------------------------------------------


def test_methods_are_the_three_offered():
    assert METHODS == (PCA_MDS, UMAP, TSNE)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_finite_two_dimensional_coordinates(method, two_groups):
    d, _ = two_groups
    result = embed_distances(d, method, n_neighbors=5, perplexity=4.0)
    assert result.coords.shape == (len(d), 2)
    assert np.isfinite(result.coords).all()
    assert result.method == method
    assert result.note and len(result.axis_labels) == 2


@pytest.mark.parametrize("method", METHODS)
def test_every_method_is_reproducible(method, two_groups):
    d, _ = two_groups
    first = embed_distances(d, method, n_neighbors=5, perplexity=4.0).coords
    second = embed_distances(d, method, n_neighbors=5, perplexity=4.0).coords
    assert first == pytest.approx(second)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_separates_two_clear_groups(method, two_groups):
    d, labels = two_groups
    coords = embed_distances(d, method, n_neighbors=5, perplexity=4.0).coords
    assert group_separation(coords, labels) < 0.75


def test_pca_axis_labels_carry_the_variance_share(two_groups):
    d, _ = two_groups
    result = embed_distances(d, PCA_MDS)
    assert "PC1" in result.axis_labels[0] and "variance" in result.axis_labels[0]
    assert result.mds is not None


def test_only_pca_reports_an_mds_result(two_groups):
    d, _ = two_groups
    assert embed_distances(d, UMAP, n_neighbors=5).mds is None
    assert embed_distances(d, TSNE, perplexity=4.0).mds is None


def test_tsne_falls_back_below_four_structures():
    """It used to raise; a three-structure run is ordinary, so it degrades instead."""
    result = embed_distances(np.zeros((3, 3)), TSNE)
    assert result.method == PCA_MDS
    assert "too few" in result.note


def test_tsne_perplexity_is_capped_below_the_sample_count(two_groups):
    """scikit-learn rejects perplexity >= n_samples outright."""
    d, _ = two_groups
    result = embed_distances(d, TSNE, perplexity=999.0)
    assert np.isfinite(result.coords).all()
    assert "perplexity" in result.note


def test_umap_neighbours_are_capped_below_the_sample_count(two_groups):
    d, _ = two_groups
    result = embed_distances(d, UMAP, n_neighbors=999)
    assert np.isfinite(result.coords).all()


def test_unknown_method_is_reported():
    with pytest.raises(ValueError, match="unknown method"):
        embed_distances(np.zeros((5, 5)), "PCoA-ish")


@pytest.mark.parametrize("n,expected", [(2, 2.0), (4, 2.0), (10, 3.0), (31, 10.0)])
def test_max_perplexity_stays_usable(n, expected):
    assert max_perplexity(n) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# vendored 3Dmol asset
# ---------------------------------------------------------------------------


def test_vendored_build_is_present_and_unmodified():
    path = threedmol_path()
    assert path.exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == THREEDMOL_SHA256


def test_cdn_url_pins_the_same_version():
    assert THREEDMOL_VERSION in THREEDMOL_CDN


def test_vendored_build_defines_the_apis_the_viewer_calls():
    """Each of these was verified against this build; a bump needs rechecking."""
    source = threedmol_source()
    for symbol in [
        "createViewerGrid",
        "createViewer",
        "addModel",
        "setStyle",
        "setHoverable",
        "addLabel",
        "removeLabel",
        "zoomTo",
        "cartoon",
        "greenCarbon",
        "CustomLinear",
        "noComputeSecondaryStructure",
    ]:
        assert symbol in source, symbol


def test_offline_script_inlines_the_source():
    tag = threedmol_script(offline=True)
    assert tag.startswith("<script>") and tag.endswith("</script>")
    assert len(tag) > 400_000
    # Exactly one closing tag, or the page would be truncated early.
    assert tag.count("</script>") == 1


def test_online_script_points_at_the_pinned_cdn_build():
    tag = threedmol_script(offline=False)
    assert tag == f'<script src="{THREEDMOL_CDN}"></script>'


def test_a_tiny_run_falls_back_instead_of_crashing():
    """Three structures broke UMAP's spectral start deep inside scipy."""
    import numpy as np

    from qpocket.viz.embed import PCA_MDS, TSNE, UMAP, embed_distances

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    for method in (UMAP, TSNE):
        result = embed_distances(d, method=method)
        assert result.coords.shape == (3, 2)
        assert result.method == PCA_MDS
        assert "too few" in result.note


def test_four_structures_still_use_the_method_asked_for():
    import numpy as np

    from qpocket.viz.embed import UMAP, embed_distances

    rng = np.random.default_rng(0)
    points = rng.random((4, 3))
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(d, 0.0)
    assert embed_distances(d, method=UMAP).method == UMAP
