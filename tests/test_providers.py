"""The feature registry reproduces today's columns and can be narrowed."""
import numpy as np
import pytest

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


def test_default_set_matches_the_historic_columns():
    assert ft.columns_for(FEATURES.resolve(ft.DEFAULT_PROVIDERS)) == ft.FEATURE_COLUMNS


def test_narrowing_drops_columns_and_requirements():
    chosen = FEATURES.resolve(["hbond", "aromatic"])
    assert ft.columns_for(chosen) == ("h_donor", "h_acceptor", "aromatic")
    assert "pqr" not in {r for p in chosen for r in p.requires}


def test_unknown_provider_names_the_options():
    with pytest.raises(ValueError, match="aromatic"):
        FEATURES.resolve(["apbs", "nope"])


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


def test_empty_structure_data_supports_geometry_only_runs():
    empty = StructureData.empty()
    out = ft.pocket_features(np.zeros((3, 3)), empty, np.zeros(3), ["hbond", "aromatic"])
    assert list(out) == ["h_donor", "h_acceptor", "aromatic"]
    assert all(values.shape == (3,) for values in out.values())


def test_a_new_provider_needs_only_registration_and_the_config():
    """The template is the worked example for a QM or ML feature."""
    assert "template" in FEATURES.available() and "template" not in ft.DEFAULT_PROVIDERS
    chosen = FEATURES.resolve(["hbond", "template"])
    assert ft.columns_for(chosen)[-1] == "template"
    values = _call(FEATURES.get("template"), _ctx(4))
    assert values["template"].shape == (4,) and np.all(values["template"] == 1.0)


def test_every_feature_plugin_declares_its_columns():
    for name in FEATURES.available():
        plugin = FEATURES.get(name)
        assert plugin.columns, name
        produced = _call(plugin, _ctx(3)._replace(options={"apbs": np.zeros(3)}))
        assert set(produced) == set(plugin.columns), name
