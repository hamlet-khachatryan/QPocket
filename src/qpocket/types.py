"""Immutable values passed between core, io and pipelines."""
from typing import NamedTuple

import numpy as np


class Scaler(NamedTuple):
    """Per-feature standardisation fitted over a set of pockets.

    Attributes:
        mean: Mean of each feature column.
        scale: Standard deviation of each column (1.0 where constant).
    """

    mean: np.ndarray
    scale: np.ndarray


class Fingerprint(NamedTuple):
    """Spectral fingerprint of one pocket.

    Attributes:
        eigenvalues: Laplacian eigenvalues, padded or truncated to global_k.
        spectrum: Flattened graph-Fourier power spectrum of the features.
    """

    eigenvalues: np.ndarray
    spectrum: np.ndarray


class StructureData(NamedTuple):
    """Protein atoms and pharmacophore sites used to featurise a pocket.

    Attributes:
        coords: (M, 3) atom coordinates from the PQR.
        charges: (M,) partial charges.
        hydrophobicity: (M,) per-atom hydrophobicity.
        donors: (K, 3) H-bond donor atoms.
        acceptors: (K, 3) H-bond acceptor atoms.
        aromatics: (K, 3) aromatic ring centres.
    """

    coords: np.ndarray
    charges: np.ndarray
    hydrophobicity: np.ndarray
    donors: np.ndarray
    acceptors: np.ndarray
    aromatics: np.ndarray
