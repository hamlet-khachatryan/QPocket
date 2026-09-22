"""The feature provider protocol and registry (pure)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from qpocket.types import StructureData


class FeatureContext(NamedTuple):
    """Everything a provider may read about the structure being featurised.

    Attributes:
        points: (N, 3) pocket points.
        pdb_path: Cleaned protein PDB, when one exists.
        pqr_path: Protonated PQR, when a provider needed it.
        structure_data: Atoms, charges, hydrophobicity and pharmacophore sites.
        options: This run's options, including values the pipeline precomputed.
    """

    points: np.ndarray
    pdb_path: Optional[Path]
    pqr_path: Optional[Path]
    structure_data: StructureData
    options: Dict[str, object]


@runtime_checkable
class FeatureProvider(Protocol):
    """One or more columns of featurized.csv.

    Attributes:
        name: Name used in features.enabled.
        columns: Columns this provider produces.
        requires: Inputs it needs: "pqr", "structure", or nothing.
    """

    name: str
    columns: Tuple[str, ...]
    requires: Tuple[str, ...]

    def compute(self, points: np.ndarray, ctx: FeatureContext) -> Dict[str, np.ndarray]:
        """Values for this provider's columns."""


_REGISTRY: Dict[str, FeatureProvider] = {}


def register(provider: FeatureProvider) -> FeatureProvider:
    """Make a provider selectable by name.

    Args:
        provider: The provider.

    Returns:
        The same provider, so it can be used as a decorator.
    """
    _REGISTRY[provider.name] = provider
    return provider


def available() -> Tuple[str, ...]:
    """Every registered provider name.

    Returns:
        Names in registration order.
    """
    return tuple(_REGISTRY)


def get(name: str) -> FeatureProvider:
    """One provider by name.

    Args:
        name: Registered name.

    Returns:
        The provider.

    Raises:
        ValueError: If no provider has that name.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown feature provider {name!r}; available: {', '.join(available())}")
    return _REGISTRY[name]


def resolve(names: Sequence[str]) -> List[FeatureProvider]:
    """Look up several providers.

    Args:
        names: Registered names.

    Returns:
        The providers, in the given order.

    Raises:
        ValueError: If any name is unknown.
    """
    return [get(name) for name in names]


def columns_for(providers: Sequence[FeatureProvider]) -> Tuple[str, ...]:
    """Columns of a provider set, in featurized.csv order.

    Args:
        providers: Chosen providers.

    Returns:
        Built-in columns first, in their historic order, then any new ones.
    """
    from qpocket.core.features import FEATURE_COLUMNS

    produced = [column for provider in providers for column in provider.columns]
    known = [c for c in FEATURE_COLUMNS if c in produced]
    extra = [c for c in produced if c not in FEATURE_COLUMNS]
    return tuple(known + extra)
