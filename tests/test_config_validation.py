"""Tests for validate_config() in config.py."""
from __future__ import annotations

import pytest

from qpocket.config import (
    DistanceConfig,
    FragmentScreenParams,
    ReferenceParams,
    validate_config,
)
from tests.conftest import make_config
from qpocket.config import evolve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref(**kw) -> ReferenceParams:
    defaults = dict(
        pdb_id="1ABC",
        chain_id="A",
        pocket_definition="Interactions",
    )
    defaults.update(kw)
    return ReferenceParams(**defaults)


def _dm(**kw) -> DistanceConfig:
    defaults = dict(
        geo_graph_neighbor_radius=2.0,
        parallel_jobs=-1,
        global_k=100,
        sigma=10.0,
        max_points=5000,
        shape_weight=0.85,
        chem_scale=12.5,
    )
    defaults.update(kw)
    return DistanceConfig(**defaults)


def _single(**kw) -> DistanceConfig:
    """A DistanceConfig whose single-stage clustering carries ``kw``.

    ``damping`` and ``max_iter`` moved out of the top level in 0.2: they only
    ever applied to affinity propagation, so they now live on the stage that
    selects it.
    """
    from qpocket.config import ClusteringParams, StageClustering

    dm = _dm()
    dm = evolve(dm, "clustering", ClusteringParams(single=StageClustering(method="affinity", **kw)))
    return dm


# ---------------------------------------------------------------------------
# Valid configurations
# ---------------------------------------------------------------------------

def test_valid_pdb_query_config():
    cfg = make_config(mode="pdb_query")
    validate_config(cfg)  # must not raise


def test_valid_fragment_screen_config():
    cfg = make_config(mode="fragment_screen")
    validate_config(cfg)  # must not raise


# ---------------------------------------------------------------------------
# ReferenceParams — pdb_id
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ReferenceParams — chain_id
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# QueryParams
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FragmentScreenParams
# ---------------------------------------------------------------------------


def test_fragment_screen_reference_index_out_of_bounds():
    fs = FragmentScreenParams(
        input_type="paths",
        structure_paths=["a.pdb", "b.pdb"],
        reference_index=5,
    )
    cfg = make_config(mode="fragment_screen", fragment_screen=fs)
    with pytest.raises(ValueError, match="reference_index"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# IOConfig
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DistanceConfig
# ---------------------------------------------------------------------------


def _stage(**kw):
    from qpocket.config import ClusteringParams, StageClustering

    return evolve(_dm(), "clustering", ClusteringParams(shape=StageClustering(**kw)))


# ---------------------------------------------------------------------------
# Buffers and cryoprotectants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cavity-defined sites
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Percentile cuts
# ---------------------------------------------------------------------------


