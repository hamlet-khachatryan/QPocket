"""The feature registry reproduces today's columns and can be narrowed."""
import numpy as np

from qpocket.core import features as ft
from qpocket.core.ports import FEATURES
from qpocket.types import FeatureContext, StructureData


def _ctx(n=6):
    rng = np.random.default_rng(0)
    data = StructureData(rng.random((20, 3)) * 8, rng.normal(size=20), rng.random(20),
                         np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((1, 3)))
    return FeatureContext(points=rng.random((n, 3)) * 8, pdb_path=None, pqr_path=None,
                          structure_data=data, options={})


def _call(plugin, ctx):
    return plugin.fn(ctx.points, ctx, plugin.params())


def test_providers_reproduce_pocket_features():
    ctx = _ctx()
    apbs = np.zeros(len(ctx.points))
    reference = ft.pocket_features(ctx.points, ctx.structure_data, apbs)
    produced = {}
    for plugin in FEATURES.resolve(ft.DEFAULT_PROVIDERS):
        produced.update(_call(plugin, ctx._replace(options={"apbs": apbs})))
    assert set(produced) == set(reference)
    for column, values in reference.items():
        assert np.array_equal(produced[column], values)


