"""Tests for qpocket.viz.interactions.

The archives carry error sentinels alongside real residues, and the notebook
divides by the ligand-bound structures only, so most of these tests are about
which structures and which entries are counted rather than about plotting.
"""

from __future__ import annotations

import h5py
import pytest

from qpocket.viz.interactions import (
    OTHER,
    SENTINELS,
    load_run_interactions,
    residue_axis,
    strip_sentinels,
)


def write_archive(path, entries):
    """Write a dict of structure ID to entries the way the pipeline does."""
    with h5py.File(path, "w") as handle:
        for pid, rows in entries.items():
            handle.create_dataset(pid, data=[r.encode("ascii") for r in sorted(rows)])


def make_run(tmp_path, observed, potential):
    """A run directory holding only the two interaction archives."""
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    write_archive(meta / "observed_interactions.h5", observed)
    write_archive(meta / "potential_interactions.h5", potential)
    return tmp_path


def test_sentinels_are_dropped_and_the_structure_survives():
    stripped = strip_sentinels({"1ABC": ["no_ligand_found"], "2DEF": ["TYR_10_A|Hydrophobic"]})
    assert stripped["1ABC"] == ()
    assert stripped["2DEF"] == ("TYR_10_A|Hydrophobic",)


@pytest.mark.parametrize("sentinel", sorted(SENTINELS))
def test_every_sentinel_is_recognised(sentinel):
    assert strip_sentinels({"1ABC": [sentinel]})["1ABC"] == ()


def test_a_residue_is_never_mistaken_for_a_sentinel():
    # Guards against a substring test: this residue key contains no sentinel,
    # but a naive "in" check over the whole entry could still misfire.
    entries = strip_sentinels({"1ABC": ["ASN_740_A|H-bond Donor"]})
    assert entries["1ABC"] == ("ASN_740_A|H-bond Donor",)


def test_axis_puts_the_most_contacted_residue_first():
    observed = {
        "1ABC": ("TYR_10_A|Hydrophobic",),
        "2DEF": ("TYR_10_A|Hydrophobic", "ASN_20_A|H-bond Donor"),
    }
    potential = {"1ABC": ("TYR_10_A", "ASN_20_A", "LEU_30_A"), "2DEF": ("TYR_10_A", "ASN_20_A")}
    assert residue_axis(observed, potential) == ("TYR_10_A", "ASN_20_A", "LEU_30_A")


def test_axis_is_stable_when_counts_tie():
    # Two residues contacted equally often fall back to the residue name, so the
    # axis cannot depend on dict or set iteration order.
    observed = {"1ABC": ("BBB_2_A|Hydrophobic", "AAA_1_A|Hydrophobic")}
    assert residue_axis(observed, {}) == ("AAA_1_A", "BBB_2_A")


def test_loading_a_run_separates_holo_from_apo(tmp_path):
    run = make_run(
        tmp_path,
        observed={"1ABC": ["TYR_10_A|Hydrophobic"], "2DEF": ["no_ligand_found"]},
        potential={"1ABC": ["TYR_10_A", "LEU_30_A"], "2DEF": ["TYR_10_A", "LEU_30_A"]},
    )
    run_data = load_run_interactions(run)
    assert run_data.holo == ("1ABC",)
    assert set(run_data.potential) == {"1ABC", "2DEF"}
    assert run_data.engaged == frozenset({"TYR_10_A"})
    assert run_data.residues == ("TYR_10_A", "LEU_30_A")


def test_an_unreserved_type_folds_into_other(tmp_path):
    run = make_run(
        tmp_path,
        observed={"1ABC": ["TYR_10_A|Hydrophobic", "ASN_20_A|Metal Complex"]},
        potential={"1ABC": ["TYR_10_A", "ASN_20_A"]},
    )
    run_data = load_run_interactions(run)
    assert run_data.types == ("Hydrophobic", OTHER)
    assert run_data.folded == ("Metal Complex",)


def test_a_run_without_archives_loads_as_none(tmp_path):
    (tmp_path / "metadata").mkdir()
    assert load_run_interactions(tmp_path) is None
