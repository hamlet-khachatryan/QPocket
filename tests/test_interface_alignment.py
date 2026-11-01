"""
Tests for multi-chain (interface) alignment functionality.

Covers:
  - find_auxiliary_chain() — pure Python, no PyMOL dependency
  - Config validation for new interface-mode fields
  - process_structure_interface() — uses the pymol mock from conftest
"""
from __future__ import annotations

import numpy as np

from qpocket.core.sequence import find_auxiliary_chain


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


# ---------------------------------------------------------------------------
# Config validation — new ReferenceParams fields
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Config validation — new AlignParams fields
# ---------------------------------------------------------------------------

