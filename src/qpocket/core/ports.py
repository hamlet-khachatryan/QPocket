"""Plug-in points: one registry per port; built-ins load on first lookup (pure)."""
from importlib import import_module

from qpocket.core.registry import Registry


def _load_builtins() -> None:
    # By name, so core never imports a plug-in module at import time.
    import_module("qpocket.plugins")


SOURCES = Registry("structure source", _load_builtins)
CHARGES = Registry("charge model", _load_builtins)
FEATURES = Registry("feature channel", _load_builtins)
CONTACTS = Registry("contact detector", _load_builtins)
CLUSTERING = Registry("cluster method", _load_builtins)
DECOYS = Registry("decoy strategy", _load_builtins)
DOCKING = Registry("docking engine", _load_builtins)
RECEPTORS = Registry("receptor preparation", _load_builtins)
