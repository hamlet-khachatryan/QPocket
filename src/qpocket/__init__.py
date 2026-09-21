"""Q-Pocket: pocket conformational space, as a library.

    import qpocket as qp
    result = qp.compare([qp.PocketSpec("7KG6", ligand="WDG"), qp.PocketSpec("4B1J", ligand="A1R")])
"""
from importlib import import_module

# Loaded on first use, so `import qpocket` (and the CLI) never needs APBS or PLIP.
_EXPORTS = {
    "fingerprint": ("qpocket.core.spectral", "fingerprint"),
    "fit_scaler": ("qpocket.core.spectral", "fit_scaler"),
    "geodesic_matrix": ("qpocket.core.spectral", "geodesic_matrix"),
    "pocket_distance": ("qpocket.core.spectral", "pocket_distance"),
    "load_scaler": ("qpocket.io.cache", "load_scaler"),
    "save_scaler": ("qpocket.io.cache", "save_scaler"),
    "featurize": ("qpocket.pipelines.compare", "featurize"),
    "compare_features": ("qpocket.pipelines.compare", "compare_features"),
    "compare": ("qpocket.pipelines.compare", "run"),
    "ComparisonResult": ("qpocket.types", "ComparisonResult"),
    "Fingerprint": ("qpocket.types", "Fingerprint"),
    "PocketSpec": ("qpocket.types", "PocketSpec"),
    "Scaler": ("qpocket.types", "Scaler"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'qpocket' has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value
