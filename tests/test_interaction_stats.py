"""Testing whether two clusters engage different residues.

The claim under test is that "the bars look different" is not evidence. Each
fixture below plants a known answer — a residue only one cluster touches, a
residue both touch equally, a residue both touch in different ways — and the
tests pin which of the three tests is supposed to fire for each. The case that
justifies keeping two tests is
:func:`test_a_residue_that_only_changes_type_is_invisible_to_the_presence_test`.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from qpocket.core.interaction_stats import (
    MIN_TESTABLE,
    build_profile,
    engaged_set,
    exclusivity,
    interaction_type_shift,
    pairwise_exclusivity,
    residue_enrichment,
)


def write_h5(path, data):
    """Write ``{pdb_id: [entry, ...]}`` the way the pipeline's writer does."""
    with h5py.File(path, "w") as hf:
        for pid, entries in data.items():
            arr = np.array([e.encode("ascii") for e in entries])
            hf.create_dataset(pid, data=arr)
    return path


CLUSTER_A = ["A1", "A2", "A3", "A4"]
CLUSTER_B = ["B1", "B2", "B3", "B4"]


@pytest.fixture
def split_profile(tmp_path):
    """ARG_1_A only in A, ASP_2_A only in B, TYR_3_A in both, LYS_4_A in neither."""
    data = {}
    for pid in CLUSTER_A:
        data[pid] = ["ARG_1_A|Hydrophobic", "TYR_3_A|Hydrophobic"]
    for pid in CLUSTER_B:
        data[pid] = ["ASP_2_A|Hydrophobic", "TYR_3_A|Hydrophobic"]
    # A fifth structure in neither cluster carries the residue nobody tests.
    data["C1"] = ["LYS_4_A|Hydrophobic"]
    return build_profile(write_h5(tmp_path / "obs.h5", data))


@pytest.fixture
def type_shift_profile(tmp_path):
    """TYR_3_A is engaged by everyone, but as a donor in A and hydrophobic in B."""
    data = {}
    for pid in CLUSTER_A:
        data[pid] = ["TYR_3_A|H-bond Donor"] * 3
    for pid in CLUSTER_B:
        data[pid] = ["TYR_3_A|Hydrophobic"] * 3
    return build_profile(write_h5(tmp_path / "shift.h5", data))


def by_residue(results):
    return {r.residue: r for r in results}


# --- reading the profile --------------------------------------------------


def test_profile_axes_are_sorted_so_two_reads_align(split_profile):
    assert list(split_profile.residues) == sorted(split_profile.residues)


def test_a_potential_file_records_no_interaction_types(tmp_path):
    """Potential entries are bare residues, with no ``|type`` suffix."""
    path = write_h5(tmp_path / "pot.h5", {"A1": ["ARG_1_A"], "A2": ["ARG_1_A"]})
    assert build_profile(path).type_counts is None


def test_a_potential_profile_still_records_engagement(tmp_path):
    path = write_h5(tmp_path / "pot.h5", {"A1": ["ARG_1_A"], "A2": ["ASP_2_A"]})
    assert build_profile(path).presence.sum() == 2


def test_restricting_to_a_cluster_drops_the_other_structures(tmp_path):
    data = {"A1": ["ARG_1_A"], "B1": ["ASP_2_A"]}
    profile = build_profile(write_h5(tmp_path / "p.h5", data), pdb_ids=["A1"])
    assert profile.structures == ("A1",)


def test_a_missing_file_is_refused_rather_than_read_as_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="interaction file not found"):
        build_profile(tmp_path / "absent.h5")


# --- presence: which residues differ --------------------------------------


def test_a_residue_exclusive_to_one_cluster_is_significant(split_profile):
    results = by_residue(residue_enrichment(split_profile, CLUSTER_A, CLUSTER_B))
    assert results["ARG_1_A"].significant


def test_a_residue_engaged_by_both_equally_is_not_significant(split_profile):
    results = by_residue(residue_enrichment(split_profile, CLUSTER_A, CLUSTER_B))
    assert not results["TYR_3_A"].significant


def test_an_exclusive_residue_reports_the_full_frequency_difference(split_profile):
    results = by_residue(residue_enrichment(split_profile, CLUSTER_A, CLUSTER_B))
    assert results["ARG_1_A"].difference == pytest.approx(1.0)


def test_the_difference_sign_says_which_cluster_is_favoured(split_profile):
    results = by_residue(residue_enrichment(split_profile, CLUSTER_A, CLUSTER_B))
    assert results["ASP_2_A"].difference == pytest.approx(-1.0)


def test_a_residue_neither_cluster_engages_is_not_tested_at_all(split_profile):
    """Carrying it would inflate the correction and weaken every real finding."""
    results = by_residue(residue_enrichment(split_profile, CLUSTER_A, CLUSTER_B))
    assert "LYS_4_A" not in results


def test_correction_never_lowers_a_p_value(split_profile):
    for result in residue_enrichment(split_profile, CLUSTER_A, CLUSTER_B):
        assert result.q_value >= result.p_value - 1e-12


def test_results_come_back_most_significant_first(split_profile):
    q_values = [r.q_value for r in residue_enrichment(split_profile, CLUSTER_A, CLUSTER_B)]
    assert q_values == sorted(q_values)


# --- power ----------------------------------------------------------------


def test_a_two_member_cluster_is_flagged_underpowered(tmp_path):
    """Two versus two cannot reach significance; that must not read as 'no difference'."""
    data = {"A1": ["ARG_1_A|Hydrophobic"], "A2": ["ARG_1_A|Hydrophobic"],
            "B1": ["ASP_2_A|Hydrophobic"], "B2": ["ASP_2_A|Hydrophobic"]}
    profile = build_profile(write_h5(tmp_path / "tiny.h5", data))
    results = residue_enrichment(profile, ["A1", "A2"], ["B1", "B2"])
    assert all(r.underpowered for r in results)


def test_an_underpowered_result_is_never_reported_significant(tmp_path):
    data = {"A1": ["ARG_1_A|Hydrophobic"], "A2": ["ARG_1_A|Hydrophobic"],
            "B1": ["ASP_2_A|Hydrophobic"], "B2": ["ASP_2_A|Hydrophobic"]}
    profile = build_profile(write_h5(tmp_path / "tiny.h5", data))
    results = residue_enrichment(profile, ["A1", "A2"], ["B1", "B2"])
    assert not any(r.significant for r in results)


def test_min_testable_is_the_documented_floor(split_profile):
    """Three is the smallest cluster the module claims to test."""
    results = residue_enrichment(split_profile, CLUSTER_A[:MIN_TESTABLE], CLUSTER_B)
    assert not any(r.underpowered for r in results)


def test_an_empty_cluster_yields_no_results(split_profile):
    assert residue_enrichment(split_profile, CLUSTER_A, []) == []


# --- interaction type -----------------------------------------------------


def test_a_residue_that_only_changes_type_is_invisible_to_the_presence_test(
    type_shift_profile,
):
    """Both clusters engage TYR_3_A in every structure, so Fisher sees nothing."""
    results = by_residue(residue_enrichment(type_shift_profile, CLUSTER_A, CLUSTER_B))
    assert not results["TYR_3_A"].significant


def test_a_residue_that_only_changes_type_is_caught_by_the_chi_square(
    type_shift_profile,
):
    """The same residue, same fixture — this is why both tests exist."""
    results = by_residue(interaction_type_shift(type_shift_profile, CLUSTER_A, CLUSTER_B))
    assert results["TYR_3_A"].significant


def test_a_residue_used_the_same_way_shows_no_type_shift(tmp_path):
    data = {pid: ["TYR_3_A|Hydrophobic"] * 3 for pid in CLUSTER_A + CLUSTER_B}
    profile = build_profile(write_h5(tmp_path / "same.h5", data))
    results = interaction_type_shift(profile, CLUSTER_A, CLUSTER_B)
    assert not any(r.significant for r in results)


def test_a_single_interaction_type_cannot_shift(tmp_path):
    """One type means a one-column table, which is not a test."""
    data = {pid: ["TYR_3_A|Hydrophobic"] * 3 for pid in CLUSTER_A + CLUSTER_B}
    profile = build_profile(write_h5(tmp_path / "same.h5", data))
    results = by_residue(interaction_type_shift(profile, CLUSTER_A, CLUSTER_B))
    assert results["TYR_3_A"].underpowered


def test_a_residue_only_one_cluster_engages_is_left_to_the_presence_test(split_profile):
    """Testing it here as well would double-count the same difference."""
    results = by_residue(interaction_type_shift(split_profile, CLUSTER_A, CLUSTER_B))
    assert "ARG_1_A" not in results


def test_a_potential_profile_has_no_types_to_shift(tmp_path):
    data = {pid: ["ARG_1_A"] for pid in CLUSTER_A + CLUSTER_B}
    profile = build_profile(write_h5(tmp_path / "pot.h5", data))
    assert interaction_type_shift(profile, CLUSTER_A, CLUSTER_B) == []


# --- exclusivity ----------------------------------------------------------


def test_disjoint_repertoires_score_zero(tmp_path):
    data = {pid: ["ARG_1_A|Hydrophobic"] for pid in CLUSTER_A}
    data.update({pid: ["ASP_2_A|Hydrophobic"] for pid in CLUSTER_B})
    profile = build_profile(write_h5(tmp_path / "disjoint.h5", data))
    assert exclusivity(profile, CLUSTER_A, CLUSTER_B).jaccard == 0.0


def test_identical_repertoires_score_one(tmp_path):
    data = {pid: ["ARG_1_A|Hydrophobic"] for pid in CLUSTER_A + CLUSTER_B}
    profile = build_profile(write_h5(tmp_path / "same.h5", data))
    assert exclusivity(profile, CLUSTER_A, CLUSTER_B).jaccard == 1.0


def test_a_half_shared_repertoire_scores_one_third(split_profile):
    """A engages {ARG, TYR}, B engages {ASP, TYR}: one shared of three."""
    assert exclusivity(split_profile, CLUSTER_A, CLUSTER_B).jaccard == pytest.approx(1 / 3)


def test_exclusivity_names_the_residues_on_each_side(split_profile):
    result = exclusivity(split_profile, CLUSTER_A, CLUSTER_B)
    assert (result.a_only, result.shared, result.b_only) == (
        ("ARG_1_A",), ("TYR_3_A",), ("ASP_2_A",),
    )


def test_a_residue_below_the_engagement_fraction_is_not_in_the_set(tmp_path):
    """One structure of four is 25%, under the 50% default."""
    data = {pid: [] for pid in CLUSTER_A}
    data["A1"] = ["ARG_1_A|Hydrophobic"]
    profile = build_profile(write_h5(tmp_path / "rare.h5", data))
    assert engaged_set(profile, CLUSTER_A) == ()


def test_raising_the_engagement_fraction_shrinks_the_set(split_profile):
    assert engaged_set(split_profile, CLUSTER_A, engaged_frac=1.01) == ()


def test_pairwise_covers_every_unordered_pair(split_profile):
    clusters = {0: CLUSTER_A, 1: CLUSTER_B, 2: ["C1"]}
    assert set(pairwise_exclusivity(split_profile, clusters)) == {(0, 1), (0, 2), (1, 2)}
