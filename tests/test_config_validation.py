"""Tests for validate_config() in config.py."""
from __future__ import annotations

import pytest

from qpocket.config import (
    DistanceConfig,
    FragmentScreenParams,
    IOConfig,
    QueryParams,
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

def test_missing_pdb_id():
    cfg = make_config(reference=_ref(pdb_id=None))
    with pytest.raises(ValueError, match="pdb_id"):
        validate_config(cfg)


def test_pdb_id_too_short():
    cfg = make_config(reference=_ref(pdb_id="1AB"))
    with pytest.raises(ValueError, match="pdb_id"):
        validate_config(cfg)


def test_pdb_id_too_long():
    cfg = make_config(reference=_ref(pdb_id="1ABCD"))
    with pytest.raises(ValueError, match="pdb_id"):
        validate_config(cfg)


def test_pdb_id_special_chars():
    cfg = make_config(reference=_ref(pdb_id="1A-C"))
    with pytest.raises(ValueError, match="pdb_id"):
        validate_config(cfg)


def test_pdb_id_exactly_4_alphanumeric_ok():
    cfg = make_config(reference=_ref(pdb_id="4XYZ"))
    validate_config(cfg)


# ---------------------------------------------------------------------------
# ReferenceParams — chain_id
# ---------------------------------------------------------------------------

def test_chain_id_lowercase():
    cfg = make_config(reference=_ref(chain_id="a"))
    with pytest.raises(ValueError, match="chain_id"):
        validate_config(cfg)


def test_chain_id_three_chars():
    cfg = make_config(reference=_ref(chain_id="ABC"))
    with pytest.raises(ValueError, match="chain_id"):
        validate_config(cfg)


def test_chain_id_two_uppercase_ok():
    cfg = make_config(reference=_ref(chain_id="AB"))
    validate_config(cfg)


# ---------------------------------------------------------------------------
# QueryParams
# ---------------------------------------------------------------------------

def test_sequence_identity_zero():
    cfg = make_config(pdb_query=QueryParams(sequence_identity=0.0))
    with pytest.raises(ValueError, match="sequence_identity"):
        validate_config(cfg)


def test_sequence_identity_above_one():
    cfg = make_config(pdb_query=QueryParams(sequence_identity=1.01))
    with pytest.raises(ValueError, match="sequence_identity"):
        validate_config(cfg)


def test_sequence_identity_half_ok():
    cfg = make_config(pdb_query=QueryParams(sequence_identity=0.5))
    validate_config(cfg)


def test_sequence_identity_one_ok():
    cfg = make_config(pdb_query=QueryParams(sequence_identity=1.0))
    validate_config(cfg)


def test_resolution_limit_zero():
    cfg = make_config(pdb_query=QueryParams(resolution_limit=0.0))
    with pytest.raises(ValueError, match="resolution_limit"):
        validate_config(cfg)


def test_resolution_limit_negative():
    cfg = make_config(pdb_query=QueryParams(resolution_limit=-1.0))
    with pytest.raises(ValueError, match="resolution_limit"):
        validate_config(cfg)


def test_min_length_zero():
    cfg = make_config(pdb_query=QueryParams(min_length=0))
    with pytest.raises(ValueError, match="min_length"):
        validate_config(cfg)


def test_max_length_equal_to_min_length():
    cfg = make_config(pdb_query=QueryParams(min_length=50, max_length=50))
    with pytest.raises(ValueError, match="max_length"):
        validate_config(cfg)


def test_max_length_less_than_min_length():
    cfg = make_config(pdb_query=QueryParams(min_length=100, max_length=50))
    with pytest.raises(ValueError, match="max_length"):
        validate_config(cfg)


def test_max_length_greater_ok():
    cfg = make_config(pdb_query=QueryParams(min_length=50, max_length=200))
    validate_config(cfg)


# ---------------------------------------------------------------------------
# FragmentScreenParams
# ---------------------------------------------------------------------------

def test_fragment_screen_csv_without_csv_path():
    fs = FragmentScreenParams(input_type="csv", csv_path=None)
    cfg = make_config(mode="fragment_screen", fragment_screen=fs)
    with pytest.raises(ValueError, match="csv_path"):
        validate_config(cfg)


def test_fragment_screen_csv_empty_csv_path():
    fs = FragmentScreenParams(input_type="csv", csv_path="")
    cfg = make_config(mode="fragment_screen", fragment_screen=fs)
    with pytest.raises(ValueError, match="csv_path"):
        validate_config(cfg)


def test_fragment_screen_paths_empty_list():
    fs = FragmentScreenParams(input_type="paths", structure_paths=[])
    cfg = make_config(mode="fragment_screen", fragment_screen=fs)
    with pytest.raises(ValueError, match="structure_paths"):
        validate_config(cfg)


def test_fragment_screen_paths_none():
    fs = FragmentScreenParams(input_type="paths", structure_paths=None)
    cfg = make_config(mode="fragment_screen", fragment_screen=fs)
    with pytest.raises(ValueError, match="structure_paths"):
        validate_config(cfg)


def test_fragment_screen_reference_index_out_of_bounds():
    fs = FragmentScreenParams(
        input_type="paths",
        structure_paths=["a.pdb", "b.pdb"],
        reference_index=5,
    )
    cfg = make_config(mode="fragment_screen", fragment_screen=fs)
    with pytest.raises(ValueError, match="reference_index"):
        validate_config(cfg)


def test_fragment_screen_valid_paths():
    fs = FragmentScreenParams(
        input_type="paths",
        structure_paths=["a.pdb", "b.pdb"],
        reference_index=1,
    )
    cfg = make_config(mode="fragment_screen", fragment_screen=fs)
    validate_config(cfg)


# ---------------------------------------------------------------------------
# IOConfig
# ---------------------------------------------------------------------------

def test_io_invalid_log_format():
    cfg = make_config(io=IOConfig(log_format="xml"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="log_format"):
        validate_config(cfg)


def test_io_invalid_compression():
    cfg = make_config(io=IOConfig(compression="lz4"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="compression"):
        validate_config(cfg)


def test_io_valid_json_gzip():
    cfg = make_config(io=IOConfig(log_format="json", compression="gzip"))
    validate_config(cfg)


def test_io_valid_text_none():
    cfg = make_config(io=IOConfig(log_format="text", compression="none"))
    validate_config(cfg)


# ---------------------------------------------------------------------------
# DistanceConfig
# ---------------------------------------------------------------------------

def test_alpha_too_high():
    cfg = make_config(distance=_dm(shape_weight=1.5))
    with pytest.raises(ValueError, match="shape_weight"):
        validate_config(cfg)


def test_alpha_negative():
    cfg = make_config(distance=_dm(shape_weight=-0.1))
    with pytest.raises(ValueError, match="shape_weight"):
        validate_config(cfg)


def test_alpha_boundary_zero_ok():
    cfg = make_config(distance=_dm(shape_weight=0.0))
    validate_config(cfg)


def test_alpha_boundary_one_ok():
    cfg = make_config(distance=_dm(shape_weight=1.0))
    validate_config(cfg)


def test_parallel_jobs_minus_two():
    cfg = make_config(distance=_dm(parallel_jobs=-2))
    with pytest.raises(ValueError, match="parallel_jobs"):
        validate_config(cfg)


def test_parallel_jobs_minus_one_ok():
    cfg = make_config(distance=_dm(parallel_jobs=-1))
    validate_config(cfg)


def test_parallel_jobs_positive_ok():
    cfg = make_config(distance=_dm(parallel_jobs=4))
    validate_config(cfg)


def test_geo_graph_neighbor_radius_zero():
    cfg = make_config(distance=_dm(geo_graph_neighbor_radius=0.0))
    with pytest.raises(ValueError, match="geo_graph_neighbor_radius"):
        validate_config(cfg)


def test_sigma_zero():
    cfg = make_config(distance=_dm(sigma=0.0))
    with pytest.raises(ValueError, match="sigma"):
        validate_config(cfg)


def test_beta_zero():
    cfg = make_config(distance=_dm(chem_scale=0.0))
    with pytest.raises(ValueError, match="chem_scale"):
        validate_config(cfg)


def test_global_k_zero():
    cfg = make_config(distance=_dm(global_k=0))
    with pytest.raises(ValueError, match="global_k"):
        validate_config(cfg)


def test_max_points_four():
    cfg = make_config(distance=_dm(max_points=4))
    with pytest.raises(ValueError, match="max_points"):
        validate_config(cfg)


def test_max_points_five_ok():
    cfg = make_config(distance=_dm(max_points=5))
    validate_config(cfg)


def test_affinity_damping_too_low():
    cfg = make_config(distance=_single(damping=0.4))
    with pytest.raises(ValueError, match="damping"):
        validate_config(cfg)


def test_affinity_damping_equals_one():
    cfg = make_config(distance=_single(damping=1.0))
    with pytest.raises(ValueError, match="damping"):
        validate_config(cfg)


def test_affinity_damping_valid():
    cfg = make_config(distance=_single(damping=0.7))
    validate_config(cfg)


def test_affinity_max_iter_zero():
    cfg = make_config(distance=_single(max_iter=0))
    with pytest.raises(ValueError, match="max_iter"):
        validate_config(cfg)


def test_affinity_max_iter_positive_ok():
    cfg = make_config(distance=_single(max_iter=100))
    validate_config(cfg)


def test_compute_defaults_and_validation():
    cfg = make_config()
    assert cfg.compute.workers == 0 and cfg.compute.backend == "process"
    cfg = evolve(cfg, "compute.backend", "cluster")
    with pytest.raises(ValueError, match="compute.backend"):
        validate_config(cfg)
    cfg = evolve(cfg, "compute.backend", "process")
    cfg = evolve(cfg, "compute.workers", -2)
    with pytest.raises(ValueError, match="compute.workers"):
        validate_config(cfg)


@pytest.mark.parametrize("key,value", [("box_padding", -1.0), ("min_heavy_atoms", 0)])
def test_sites_validation(key, value):
    from qpocket.config import QPocketConfig
    base = evolve(QPocketConfig(), "reference.pdb_id", "1ABC")
    with pytest.raises(ValueError, match=f"sites.{key}"):
        validate_config(evolve(base, f"sites.{key}", value))


def _stage(**kw):
    from qpocket.config import ClusteringParams, StageClustering

    return evolve(_dm(), "clustering", ClusteringParams(shape=StageClustering(**kw)))


@pytest.mark.parametrize("kw", [dict(percentile=0.0), dict(percentile=100.0), dict(method="kmedoids")])
def test_bad_stage_cuts_are_refused(kw):
    with pytest.raises(ValueError, match="distance.clustering.shape"):
        validate_config(make_config(distance=_stage(**kw)))


@pytest.mark.parametrize("kw", [dict(percentile=15.0), dict(percentile=0.5), dict(method="kmedoids", k=3),
                                dict(method="affinity")])
def test_good_stage_cuts_pass(kw):
    validate_config(make_config(distance=_stage(**kw)))


# ---------------------------------------------------------------------------
# Buffers and cryoprotectants
# ---------------------------------------------------------------------------

def test_buffer_name_as_cofactor_rejected():
    cfg = evolve(make_config(mode="pdb_query"), "cleaning.allowed_cofactors", "ZN|GOL")
    with pytest.raises(ValueError, match="sites.keep_ligands"):
        validate_config(cfg)


def test_keep_ligands_defaults_empty():
    assert list(make_config(mode="pdb_query").sites.keep_ligands) == []


def test_keep_ligands_must_be_a_list():
    cfg = evolve(make_config(mode="pdb_query"), "sites.keep_ligands", "CIT")
    with pytest.raises(ValueError, match="sites.keep_ligands"):
        validate_config(cfg)


# ---------------------------------------------------------------------------
# Cavity-defined sites
# ---------------------------------------------------------------------------

def test_site_defaults():
    s = make_config(mode="pdb_query").sites
    assert (s.min_cavity_points, s.cavity_overlap, s.min_structure_fraction,
            s.ligand_fraction, s.box_padding) == (50, 0.5, 0.05, 0.5, 0.0)
    assert not hasattr(s, "link_cutoff")


@pytest.mark.parametrize("key,value", [("cavity_overlap", 0.0), ("cavity_overlap", 1.5),
                                       ("ligand_fraction", 0.0), ("ligand_fraction", 1.1),
                                       ("min_structure_fraction", -0.1), ("min_structure_fraction", 1.1),
                                       ("min_cavity_points", 0), ("box_padding", -1.0)])
def test_site_parameter_ranges(key, value):
    with pytest.raises(ValueError, match=f"sites.{key}"):
        validate_config(evolve(make_config(mode="pdb_query"), f"sites.{key}", value))


# ---------------------------------------------------------------------------
# Percentile cuts
# ---------------------------------------------------------------------------

def test_cut_defaults_are_percentiles():
    from qpocket.config import ClusteringParams

    c = ClusteringParams()
    assert (c.shape.percentile, c.chem.percentile, c.single.percentile) == (15.0, 20.0, 15.0)
    assert not hasattr(c.shape, "criterion") and not hasattr(c.shape, "threshold")


@pytest.mark.parametrize("value", [0.0, 100.0, -5.0])
def test_percentile_range(value):
    cfg = evolve(make_config(mode="pdb_query"), "distance.clustering.shape.percentile", value)
    with pytest.raises(ValueError, match="distance.clustering.shape.percentile"):
        validate_config(cfg)


def test_kmedoids_needs_k():
    cfg = evolve(make_config(mode="pdb_query"), "distance.clustering.shape.method", "kmedoids")
    with pytest.raises(ValueError, match="distance.clustering.shape.k"):
        validate_config(cfg)
