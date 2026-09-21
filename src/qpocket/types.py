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
