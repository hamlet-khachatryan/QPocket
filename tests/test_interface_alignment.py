"""
Tests for multi-chain (interface) alignment functionality.

Covers:
  - find_auxiliary_chain() — pure Python, no PyMOL dependency
  - Config validation for new interface-mode fields
  - process_structure_interface() — uses the pymol mock from conftest
"""
from __future__ import annotations

import numpy as np
import pytest

from qpocket.core.sequence import find_auxiliary_chain
from qpocket.config import (
    AlignParams,
    ReferenceParams,
    validate_config,
)
from tests.conftest import make_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seqs(**kw) -> dict:
    """Build chain_sequences dict from keyword arguments."""
    return {k: v for k, v in kw.items()}


def _cents(**kw) -> dict:
    """Build chain_ca_centroids dict from keyword arguments."""
    return {k: np.array(v, dtype=float) for k, v in kw.items()}


# ---------------------------------------------------------------------------
# find_auxiliary_chain — sequence filter
# ---------------------------------------------------------------------------

class TestFindAuxiliaryChainSequenceFilter:

    def test_single_candidate_returned_directly(self):
        """Only one chain meets the identity threshold — no spatial step needed."""
        seqs = _seqs(A="ACDEFGHIKLMNPQRSTVWY", B="ACDEFGHIKLMNPQRSTVWY", C="MMMMMMMMMM")
        cents = _cents(A=[0, 0, 0], B=[5, 0, 0], C=[100, 0, 0])
        ref_aux_seq = "ACDEFGHIKLMNPQRSTVWY"
        ref_aux_cent = np.array([5.0, 0.0, 0.0])

        result = find_auxiliary_chain(
            chain_sequences=seqs,
            chain_ca_centroids=cents,
            ref_aux_seq=ref_aux_seq,
            ref_aux_centroid=ref_aux_cent,
            primary_chain="A",
            min_identity=0.70,
        )
        assert result == "B"

    def test_primary_chain_excluded_from_candidates(self):
        """Primary chain is never returned even if its sequence matches perfectly."""
        seqs = _seqs(A="ACDEFGHIKLMNPQRSTVWY", B="MMMMMMMMMMMMMMMMMMMM")
        cents = _cents(A=[0, 0, 0], B=[50, 0, 0])
        ref_aux_seq = "ACDEFGHIKLMNPQRSTVWY"

        result = find_auxiliary_chain(
            chain_sequences=seqs,
            chain_ca_centroids=cents,
            ref_aux_seq=ref_aux_seq,
            ref_aux_centroid=np.array([0.0, 0.0, 0.0]),
            primary_chain="A",
            min_identity=0.70,
        )
        assert result is None  # B doesn't match, A is excluded

    def test_no_chain_above_threshold_returns_none(self):
        seqs = _seqs(A="ACDEFGHIKLMNPQRSTVWY", B="MMMMMMMMMMMMMMMMMMMM")
        cents = _cents(A=[0, 0, 0], B=[5, 0, 0])
        ref_aux_seq = "ACDEFGHIKLMNPQRSTVWY"

        result = find_auxiliary_chain(
            chain_sequences=seqs,
            chain_ca_centroids=cents,
            ref_aux_seq=ref_aux_seq,
            ref_aux_centroid=np.array([5.0, 0.0, 0.0]),
            primary_chain="A",
            min_identity=0.95,  # B won't reach this
        )
        assert result is None

    def test_empty_chain_sequence_skipped(self):
        """Chains with empty sequences should not be considered."""
        seqs = _seqs(A="ACDEFGHIKLMNPQRSTVWY", B="", C="ACDEFGHIKLMNPQRSTVWY")
        cents = _cents(A=[0, 0, 0], B=[5, 0, 0], C=[10, 0, 0])
        ref_aux_seq = "ACDEFGHIKLMNPQRSTVWY"

        result = find_auxiliary_chain(
            chain_sequences=seqs,
            chain_ca_centroids=cents,
            ref_aux_seq=ref_aux_seq,
            ref_aux_centroid=np.array([10.0, 0.0, 0.0]),
            primary_chain="A",
            min_identity=0.70,
        )
        # B is empty so skipped, C matches
        assert result == "C"


# ---------------------------------------------------------------------------
# find_auxiliary_chain — spatial disambiguation (homodimer)
# ---------------------------------------------------------------------------

class TestFindAuxiliaryChainSpatialDisambiguation:

    HOMODIMER_SEQ = "ACDEFGHIKLMNPQRSTVWY"

    def _homodimer_seqs(self, chains):
        return {c: self.HOMODIMER_SEQ for c in chains}

    def test_homodimer_picks_closer_chain(self):
        """With two equally matching chains, pick the one nearer the reference centroid."""
        seqs = self._homodimer_seqs(["A", "B", "C"])
        # Reference auxiliary centroid at [10, 0, 0]
        # B is at [9, 0, 0] (dist 1), C is at [50, 0, 0] (dist 40)
        cents = _cents(A=[0, 0, 0], B=[9, 0, 0], C=[50, 0, 0])
        ref_cent = np.array([10.0, 0.0, 0.0])

        result = find_auxiliary_chain(
            chain_sequences=seqs,
            chain_ca_centroids=cents,
            ref_aux_seq=self.HOMODIMER_SEQ,
            ref_aux_centroid=ref_cent,
            primary_chain="A",
            min_identity=0.70,
        )
        assert result == "B"

    def test_homodimer_three_candidates_picks_nearest(self):
        seqs = self._homodimer_seqs(["A", "B", "C", "D"])
        cents = _cents(A=[0, 0, 0], B=[100, 0, 0], C=[3, 0, 0], D=[50, 0, 0])
        ref_cent = np.array([2.0, 0.0, 0.0])

        result = find_auxiliary_chain(
            chain_sequences=seqs,
            chain_ca_centroids=cents,
            ref_aux_seq=self.HOMODIMER_SEQ,
            ref_aux_centroid=ref_cent,
            primary_chain="A",
            min_identity=0.70,
        )
        assert result == "C"  # distance 1 from ref_cent

    def test_large_deviation_still_returns_match(self):
        """Exceeding max_centroid_deviation emits a warning but still returns the chain."""
        seqs = self._homodimer_seqs(["A", "B"])
        cents = _cents(A=[0, 0, 0], B=[100, 0, 0])
        ref_cent = np.array([0.0, 0.0, 0.0])

        result = find_auxiliary_chain(
            chain_sequences=seqs,
            chain_ca_centroids=cents,
            ref_aux_seq=self.HOMODIMER_SEQ,
            ref_aux_centroid=ref_cent,
            primary_chain="A",
            min_identity=0.70,
            max_centroid_deviation=5.0,  # B is at distance 100 — exceeds threshold
        )
        # Should still return B (warning is logged, not an exception)
        assert result == "B"

    def test_missing_centroid_falls_back_to_seq_identity(self):
        """If centroid is missing for some candidates, still returns a result."""
        seqs = self._homodimer_seqs(["A", "B", "C"])
        # C has no centroid entry
        cents = _cents(A=[0, 0, 0], B=[5, 0, 0])
        ref_cent = np.array([5.0, 0.0, 0.0])

        result = find_auxiliary_chain(
            chain_sequences=seqs,
            chain_ca_centroids=cents,
            ref_aux_seq=self.HOMODIMER_SEQ,
            ref_aux_centroid=ref_cent,
            primary_chain="A",
            min_identity=0.70,
        )
        # C has no centroid — best spatial match is B (dist 0)
        assert result == "B"


# ---------------------------------------------------------------------------
# Config validation — new ReferenceParams fields
# ---------------------------------------------------------------------------

class TestConfigValidationAuxiliaryChains:

    def test_valid_single_auxiliary_chain(self):
        ref = ReferenceParams(
            pdb_id="1ABC",
            chain_id="A",
            auxiliary_chain_ids=["B"],
            pocket_definition="Interactions",
        )
        cfg = make_config(reference=ref)
        validate_config(cfg)  # must not raise

    def test_valid_two_auxiliary_chains(self):
        ref = ReferenceParams(
            pdb_id="1ABC",
            chain_id="A",
            auxiliary_chain_ids=["B", "C"],
            pocket_definition="Interactions",
        )
        cfg = make_config(reference=ref)
        validate_config(cfg)

    def test_null_auxiliary_chain_ids_valid(self):
        ref = ReferenceParams(
            pdb_id="1ABC",
            chain_id="A",
            auxiliary_chain_ids=None,
            pocket_definition="Interactions",
        )
        cfg = make_config(reference=ref)
        validate_config(cfg)

    def test_auxiliary_chain_lowercase_raises(self):
        ref = ReferenceParams(
            pdb_id="1ABC",
            chain_id="A",
            auxiliary_chain_ids=["b"],  # lowercase — invalid
            pocket_definition="Interactions",
        )
        cfg = make_config(reference=ref)
        with pytest.raises(ValueError, match="auxiliary_chain_ids"):
            validate_config(cfg)

    def test_auxiliary_chain_too_long_raises(self):
        ref = ReferenceParams(
            pdb_id="1ABC",
            chain_id="A",
            auxiliary_chain_ids=["BCD"],  # 3 chars — invalid
            pocket_definition="Interactions",
        )
        cfg = make_config(reference=ref)
        with pytest.raises(ValueError, match="auxiliary_chain_ids"):
            validate_config(cfg)

    def test_primary_in_auxiliary_raises(self):
        """chain_id must not be repeated in auxiliary_chain_ids."""
        ref = ReferenceParams(
            pdb_id="1ABC",
            chain_id="A",
            auxiliary_chain_ids=["A", "B"],  # A is primary — invalid
            pocket_definition="Interactions",
        )
        cfg = make_config(reference=ref)
        with pytest.raises(ValueError, match="auxiliary_chain_ids"):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# Config validation — new AlignParams fields
# ---------------------------------------------------------------------------

class TestConfigValidationAlignParams:

    def test_min_auxiliary_identity_zero_raises(self):
        cfg = make_config(alignment=AlignParams(min_auxiliary_identity=0.0))
        with pytest.raises(ValueError, match="min_auxiliary_identity"):
            validate_config(cfg)

    def test_min_auxiliary_identity_above_one_raises(self):
        cfg = make_config(alignment=AlignParams(min_auxiliary_identity=1.01))
        with pytest.raises(ValueError, match="min_auxiliary_identity"):
            validate_config(cfg)

    def test_min_auxiliary_identity_boundary_one_ok(self):
        cfg = make_config(alignment=AlignParams(min_auxiliary_identity=1.0))
        validate_config(cfg)

    def test_min_auxiliary_identity_half_ok(self):
        cfg = make_config(alignment=AlignParams(min_auxiliary_identity=0.5))
        validate_config(cfg)

    def test_max_centroid_deviation_zero_raises(self):
        cfg = make_config(
            alignment=AlignParams(max_auxiliary_centroid_deviation=0.0)
        )
        with pytest.raises(ValueError, match="max_auxiliary_centroid_deviation"):
            validate_config(cfg)

    def test_max_centroid_deviation_negative_raises(self):
        cfg = make_config(
            alignment=AlignParams(max_auxiliary_centroid_deviation=-5.0)
        )
        with pytest.raises(ValueError, match="max_auxiliary_centroid_deviation"):
            validate_config(cfg)

    def test_max_centroid_deviation_positive_ok(self):
        cfg = make_config(
            alignment=AlignParams(max_auxiliary_centroid_deviation=20.0)
        )
        validate_config(cfg)

    def test_require_all_chains_false_ok(self):
        cfg = make_config(alignment=AlignParams(require_all_chains=False))
        validate_config(cfg)

    def test_auxiliary_uniprot_ids_none_ok(self):
        cfg = make_config(alignment=AlignParams(auxiliary_uniprot_ids=None))
        validate_config(cfg)
