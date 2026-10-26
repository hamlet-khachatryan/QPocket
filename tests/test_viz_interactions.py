"""Tests for qpocket.viz.interactions.

The archives carry error sentinels alongside real residues, and the notebook
divides by the ligand-bound structures only, so most of these tests are about
which structures and which entries are counted rather than about plotting.
"""

from __future__ import annotations

import h5py
import pytest

from qpocket.viz.interactions import (
    ITYPE_COLORS,
    OBSERVED_COLOR,
    OTHER,
    POTENTIAL_COLOR,
    SENTINELS,
    carbon_scheme,
    classify_residues,
    cluster_frequencies,
    filter_residues,
    load_run_interactions,
    observed_figure,
    potential_figure,
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


def five_member_run(tmp_path):
    """Five structures, three of them ligand-bound, with one shared residue."""
    return make_run(
        tmp_path,
        observed={
            "1AAA": ["TYR_10_A|Hydrophobic"],
            "2BBB": ["TYR_10_A|Hydrophobic"],
            "3CCC": ["ASN_20_A|H-bond Donor"],
            "4DDD": ["no_ligand_found"],
            "5EEE": ["no_ligand_found"],
        },
        potential={pid: ["TYR_10_A", "ASN_20_A", "LEU_30_A"]
                   for pid in ["1AAA", "2BBB", "3CCC", "4DDD", "5EEE"]},
    )


def test_observed_percent_divides_by_the_ligand_bound_members(tmp_path):
    # TYR is contacted in 2 of the 3 holo structures. Counting the two apo ones
    # would give 40%, which no structure in the run supports.
    run = load_run_interactions(five_member_run(tmp_path))
    freqs = cluster_frequencies(run, list(run.potential))
    assert freqs.n_members == 5
    assert freqs.n_holo == 3
    assert freqs.contacted["TYR_10_A"] == pytest.approx(200 / 3)
    assert freqs.contacted["ASN_20_A"] == pytest.approx(100 / 3)


def test_lines_the_pocket_divides_by_every_member(tmp_path):
    # An apo structure has no interactions but its pocket still has a lining.
    run = load_run_interactions(five_member_run(tmp_path))
    freqs = cluster_frequencies(run, list(run.potential))
    assert freqs.potential["LEU_30_A"] == pytest.approx(100.0)
    assert freqs.contacted["LEU_30_A"] == pytest.approx(0.0)


def test_a_cluster_sees_only_its_own_members(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    freqs = cluster_frequencies(run, ["1AAA", "2BBB"], label="cluster 0")
    assert freqs.label == "cluster 0"
    assert freqs.n_holo == 2
    assert freqs.contacted["TYR_10_A"] == pytest.approx(100.0)
    assert freqs.contacted["ASN_20_A"] == pytest.approx(0.0)


def test_the_axis_survives_a_cluster_that_touches_nothing(tmp_path):
    # Rows must not vanish when a cluster is all apo: an empty row is the signal.
    run = load_run_interactions(five_member_run(tmp_path))
    freqs = cluster_frequencies(run, ["4DDD", "5EEE"])
    assert freqs.residues == run.residues
    assert freqs.n_holo == 0
    assert freqs.contacted.sum() == pytest.approx(0.0)


def test_switching_cluster_never_reorders_the_axis(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    for members in (["1AAA", "2BBB"], ["3CCC"], list(run.potential)):
        assert cluster_frequencies(run, members).residues == run.residues


def test_observed_columns_are_exactly_the_runs_types(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    freqs = cluster_frequencies(run, list(run.potential))
    assert tuple(freqs.observed.columns) == run.types


def test_an_unreserved_type_lands_in_the_other_column(tmp_path):
    run = load_run_interactions(make_run(
        tmp_path,
        observed={"1AAA": ["TYR_10_A|Metal Complex"]},
        potential={"1AAA": ["TYR_10_A"]},
    ))
    freqs = cluster_frequencies(run, ["1AAA"])
    assert freqs.observed.loc["TYR_10_A", OTHER] == pytest.approx(100.0)


def test_only_unused_keeps_the_residues_never_contacted(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    trimmed = filter_residues(cluster_frequencies(run, list(run.potential)), only_unused=True)
    assert trimmed.residues == ("LEU_30_A",)


def test_filtering_preserves_the_axis_order(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    freqs = cluster_frequencies(run, list(run.potential))
    trimmed = filter_residues(freqs, min_percent=1.0)
    assert list(trimmed.residues) == [r for r in freqs.residues if r in set(trimmed.residues)]
    assert list(trimmed.observed.index) == list(trimmed.residues)
    assert trimmed.n_holo == freqs.n_holo


def test_observed_figure_stacks_one_trace_per_type(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    fig = observed_figure(cluster_frequencies(run, list(run.potential)))
    assert fig.layout.barmode == "stack"
    names = {trace.name for trace in fig.data}
    assert names == {"Hydrophobic", "H-bond Donor"}


def test_observed_figure_uses_the_reserved_hue(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    fig = observed_figure(cluster_frequencies(run, list(run.potential)))
    for trace in fig.data:
        assert trace.marker.color == ITYPE_COLORS[trace.name]


def test_both_figures_share_one_category_order(tmp_path):
    # Rows must line up across the pair, so a residue reads straight across.
    run = load_run_interactions(five_member_run(tmp_path))
    freqs = cluster_frequencies(run, list(run.potential))
    left, right = observed_figure(freqs), potential_figure(freqs)
    assert left.layout.yaxis.categoryarray == right.layout.yaxis.categoryarray
    # Plotly draws the first category at the bottom, so the axis is reversed to
    # put the most contacted residue at the top.
    assert left.layout.yaxis.categoryarray == tuple(reversed(freqs.residues))


def test_a_type_absent_from_the_selection_gets_no_trace(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    fig = observed_figure(cluster_frequencies(run, ["1AAA", "2BBB"], label="cluster 0"))
    assert {trace.name for trace in fig.data} == {"Hydrophobic"}


def test_potential_figure_marks_the_contacted_share(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    fig = potential_figure(cluster_frequencies(run, list(run.potential)))
    assert fig.layout.barmode == "group"
    assert [trace.name for trace in fig.data] == ["lines the pocket", "contacted"]
    lining = dict(zip(fig.data[0].y, fig.data[0].x))
    assert lining["LEU_30_A"] == pytest.approx(100.0)


def test_both_denominators_appear_in_the_title(tmp_path):
    # The two panels divide by different things; the chart has to say so.
    run = load_run_interactions(five_member_run(tmp_path))
    freqs = cluster_frequencies(run, list(run.potential))
    assert "3" in observed_figure(freqs).layout.title.text
    assert "5" in potential_figure(freqs).layout.title.text


def test_an_empty_axis_still_makes_a_figure(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    empty = filter_residues(cluster_frequencies(run, list(run.potential)), min_percent=1e6)
    assert empty.residues == ()
    assert observed_figure(empty).data == ()
    assert potential_figure(empty) is not None


def test_carbon_scheme_recolours_carbon_and_leaves_cpk_alone():
    scheme = carbon_scheme("#eb6834")
    assert scheme["prop"] == "elem"
    assert scheme["map"]["C"] == "#eb6834"
    assert scheme["map"]["N"] == "#3050F8"
    assert scheme["map"]["O"] == "#FF0D0D"
    assert scheme["map"]["S"] == "#FFFF30"


def test_carbon_scheme_does_not_share_its_map_between_calls():
    first = carbon_scheme("#eb6834")
    carbon_scheme("#7b5cd6")
    assert first["map"]["C"] == "#eb6834"


def test_the_two_class_colours_differ_from_each_other_and_from_cpk():
    cpk = {carbon_scheme("#000000")["map"][e] for e in ("N", "O", "S")}
    assert OBSERVED_COLOR != POTENTIAL_COLOR
    assert not {OBSERVED_COLOR, POTENTIAL_COLOR} & cpk


def test_classify_splits_the_picked_residues_by_class(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    groups = classify_residues(run, ["TYR_10_A", "LEU_30_A"])
    assert groups == {"observed": ("TYR_10_A",), "potential": ("LEU_30_A",)}


def test_classify_ignores_a_residue_that_was_not_picked(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    assert classify_residues(run, []) == {}


def test_classify_keeps_the_axis_order(tmp_path):
    run = load_run_interactions(five_member_run(tmp_path))
    groups = classify_residues(run, ["ASN_20_A", "TYR_10_A"])
    assert groups["observed"] == ("TYR_10_A", "ASN_20_A")
