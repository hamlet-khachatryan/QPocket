"""Clustering methods behind the registry give exactly today's labels."""
from types import SimpleNamespace

import numpy as np
import pytest

from qpocket.core import clustering as cl
from qpocket.core.ports import CLUSTERING


def _distances(n=12, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.vstack([rng.normal(0, 0.3, (n // 2, 2)), rng.normal(4, 0.3, (n - n // 2, 2))])
    return np.linalg.norm(pts[:, None] - pts[None], axis=-1)


def _stage(**kw):
    base = dict(method="hierarchical", linkage="average", k=None,
                criterion="count", threshold=None, damping=0.9, max_iter=300)
    base.update(kw)
    return SimpleNamespace(**base)


CASES = [
    _stage(criterion="relative", threshold=0.5), _stage(k=3), _stage(criterion="threshold", threshold=1.0),
    _stage(method="kmedoids", k=2), _stage(method="affinity"),
]


@pytest.mark.parametrize("stage", CASES, ids=lambda s: f"{s.method}-{s.k}-{s.criterion}")
def test_each_plugin_matches_the_core_dispatch(stage):
    d = _distances()
    expected = cl._stage_labels(d, stage.method, stage.linkage, stage.k,
                                stage.criterion, stage.threshold, stage.damping, stage.max_iter)
    got = CLUSTERING.get(stage.method).fn(d, stage)
    assert np.array_equal(got.labels, expected.labels)


def test_a_threshold_cut_needs_a_hierarchical_stage():
    with pytest.raises(ValueError, match="hierarchical"):
        CLUSTERING.get("kmedoids").fn(_distances(), _stage(method="kmedoids", criterion="threshold",
                                                           threshold=1.0))


@pytest.mark.parametrize("method", ["hierarchical", "kmedoids"])
def test_a_count_stage_without_k_names_the_fix(method):
    with pytest.raises(ValueError, match="set k or use criterion 'threshold' or 'relative'"):
        CLUSTERING.get(method).fn(_distances(), _stage(method=method))


def test_a_relative_cut_needs_a_hierarchical_stage():
    with pytest.raises(ValueError, match="hierarchical"):
        CLUSTERING.get("kmedoids").fn(_distances(), _stage(method="kmedoids", criterion="relative",
                                                           threshold=0.5))


def test_affinity_ignores_the_criterion():
    d = _distances()
    relative = CLUSTERING.get("affinity").fn(d, _stage(method="affinity", criterion="relative", threshold=0.5))
    assert np.array_equal(relative.labels, CLUSTERING.get("affinity").fn(d, _stage(method="affinity")).labels)


def test_two_stage_through_the_registry_matches_the_builtin():
    from qpocket.pipelines.clustering import stage_labels

    d1, d2 = _distances(seed=1), _distances(seed=2)
    a = cl.two_stage_labels(d1, d2, min_cluster_size=3)
    b = cl.two_stage_labels(d1, d2, min_cluster_size=3, stage_labels=stage_labels)
    assert np.array_equal(a.labels, b.labels)


def test_every_config_method_is_registered():
    import typing

    from qpocket.config import StageClustering

    methods = typing.get_args(StageClustering.__annotations__["method"])
    assert set(methods) <= set(CLUSTERING.available())
