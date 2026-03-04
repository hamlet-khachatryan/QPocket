"""Tests for validate_config() in config.py."""
from __future__ import annotations

import pytest

from qpocket_pipeline.config import (
    BoundingBoxParams,
    DistanceMatrix,
    FragmentScreenParams,
    IOConfig,
    QueryParams,
    QPocketConfig,
    ReferenceParams,
    validate_config,
)
from tests.conftest import make_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref(**kw) -> ReferenceParams:
    defaults = dict(
        pdb_id="1ABC",
        chain_id="A",
        ligand_id="LIG",
        pocket_definition="Interactions",
        pocket_mode="cavity_detection",
    )
    defaults.update(kw)
    return ReferenceParams(**defaults)


def _dm(**kw) -> DistanceMatrix:
    defaults = dict(
        geo_graph_neighbor_radius=2.0,
        paralell_jobs=-1,
        global_k=100,
        sigma=10.0,
        max_points=5000,
        alpha=0.85,
        beta=12.5,
        affinity_damping=0.9,
        affinity_max_iter=300,
    )
    defaults.update(kw)
    return DistanceMatrix(**defaults)


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
# ReferenceParams — bounding_box mode
# ---------------------------------------------------------------------------

def test_bounding_box_mode_missing_center_and_dims():
    ref = _ref(pocket_mode="bounding_box", bounding_box=BoundingBoxParams(center=None, dimensions=None))
    cfg = make_config(reference=ref)
    with pytest.raises(ValueError, match="bounding_box"):
        validate_config(cfg)


def test_bounding_box_mode_missing_dims():
    ref = _ref(
        pocket_mode="bounding_box",
        bounding_box=BoundingBoxParams(center=[1.0, 2.0, 3.0], dimensions=None),
    )
    cfg = make_config(reference=ref)
    with pytest.raises(ValueError, match="bounding_box"):
        validate_config(cfg)


def test_bounding_box_wrong_center_length():
    ref = _ref(
        pocket_mode="bounding_box",
        bounding_box=BoundingBoxParams(center=[1.0, 2.0], dimensions=[5.0, 5.0, 5.0]),
    )
    cfg = make_config(reference=ref)
    with pytest.raises(ValueError, match="center"):
        validate_config(cfg)


def test_bounding_box_negative_dimension():
    ref = _ref(
        pocket_mode="bounding_box",
        bounding_box=BoundingBoxParams(center=[1.0, 2.0, 3.0], dimensions=[5.0, -1.0, 5.0]),
    )
    cfg = make_config(reference=ref)
    with pytest.raises(ValueError, match="dimensions"):
        validate_config(cfg)


def test_bounding_box_valid():
    ref = _ref(
        pocket_mode="bounding_box",
        pocket_definition="Solvent",
        ligand_id=None,
        bounding_box=BoundingBoxParams(center=[0.0, 0.0, 0.0], dimensions=[10.0, 10.0, 10.0]),
    )
    cfg = make_config(reference=ref)
    validate_config(cfg)


# ---------------------------------------------------------------------------
# ReferenceParams — ligand_id requirement
# ---------------------------------------------------------------------------

def test_interactions_cavity_no_ligand_id():
    ref = _ref(pocket_definition="Interactions", pocket_mode="cavity_detection", ligand_id=None)
    cfg = make_config(reference=ref)
    with pytest.raises(ValueError, match="ligand_id"):
        validate_config(cfg)


def test_solvent_pocket_no_ligand_id_ok():
    ref = _ref(pocket_definition="Solvent", pocket_mode="cavity_detection", ligand_id=None)
    cfg = make_config(reference=ref)
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


def test_fragment_screen_interaction_padding_zero():
    fs = FragmentScreenParams(
        input_type="csv", csv_path="/p.csv", interaction_padding=0.0
    )
    cfg = make_config(mode="fragment_screen", fragment_screen=fs)
    with pytest.raises(ValueError, match="interaction_padding"):
        validate_config(cfg)


def test_fragment_screen_valid_paths():
    fs = FragmentScreenParams(
        input_type="paths",
        structure_paths=["a.pdb", "b.pdb"],
        reference_index=1,
        interaction_padding=5.0,
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
# DistanceMatrix
# ---------------------------------------------------------------------------

def test_alpha_too_high():
    cfg = make_config(spectral_distance_matrix=_dm(alpha=1.5))
    with pytest.raises(ValueError, match="alpha"):
        validate_config(cfg)


def test_alpha_negative():
    cfg = make_config(spectral_distance_matrix=_dm(alpha=-0.1))
    with pytest.raises(ValueError, match="alpha"):
        validate_config(cfg)


def test_alpha_boundary_zero_ok():
    cfg = make_config(spectral_distance_matrix=_dm(alpha=0.0))
    validate_config(cfg)


def test_alpha_boundary_one_ok():
    cfg = make_config(spectral_distance_matrix=_dm(alpha=1.0))
    validate_config(cfg)


def test_paralell_jobs_minus_two():
    cfg = make_config(spectral_distance_matrix=_dm(paralell_jobs=-2))
    with pytest.raises(ValueError, match="paralell_jobs"):
        validate_config(cfg)


def test_paralell_jobs_minus_one_ok():
    cfg = make_config(spectral_distance_matrix=_dm(paralell_jobs=-1))
    validate_config(cfg)


def test_paralell_jobs_positive_ok():
    cfg = make_config(spectral_distance_matrix=_dm(paralell_jobs=4))
    validate_config(cfg)


def test_geo_graph_neighbor_radius_zero():
    cfg = make_config(spectral_distance_matrix=_dm(geo_graph_neighbor_radius=0.0))
    with pytest.raises(ValueError, match="geo_graph_neighbor_radius"):
        validate_config(cfg)


def test_sigma_zero():
    cfg = make_config(spectral_distance_matrix=_dm(sigma=0.0))
    with pytest.raises(ValueError, match="sigma"):
        validate_config(cfg)


def test_beta_zero():
    cfg = make_config(spectral_distance_matrix=_dm(beta=0.0))
    with pytest.raises(ValueError, match="beta"):
        validate_config(cfg)


def test_global_k_zero():
    cfg = make_config(spectral_distance_matrix=_dm(global_k=0))
    with pytest.raises(ValueError, match="global_k"):
        validate_config(cfg)


def test_max_points_four():
    cfg = make_config(spectral_distance_matrix=_dm(max_points=4))
    with pytest.raises(ValueError, match="max_points"):
        validate_config(cfg)


def test_max_points_five_ok():
    cfg = make_config(spectral_distance_matrix=_dm(max_points=5))
    validate_config(cfg)


def test_affinity_damping_too_low():
    cfg = make_config(spectral_distance_matrix=_dm(affinity_damping=0.4))
    with pytest.raises(ValueError, match="affinity_damping"):
        validate_config(cfg)


def test_affinity_damping_equals_one():
    cfg = make_config(spectral_distance_matrix=_dm(affinity_damping=1.0))
    with pytest.raises(ValueError, match="affinity_damping"):
        validate_config(cfg)


def test_affinity_damping_valid():
    cfg = make_config(spectral_distance_matrix=_dm(affinity_damping=0.7))
    validate_config(cfg)


def test_affinity_max_iter_zero():
    cfg = make_config(spectral_distance_matrix=_dm(affinity_max_iter=0))
    with pytest.raises(ValueError, match="affinity_max_iter"):
        validate_config(cfg)


def test_affinity_max_iter_positive_ok():
    cfg = make_config(spectral_distance_matrix=_dm(affinity_max_iter=100))
    validate_config(cfg)