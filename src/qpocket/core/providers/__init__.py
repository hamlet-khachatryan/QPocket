"""Feature providers: importing this package registers the built-ins."""
from qpocket.core.providers import builtin, template  # noqa: F401  (register on import)
from qpocket.core.providers.base import (
    FeatureContext,
    FeatureProvider,
    available,
    columns_for,
    get,
    register,
    resolve,
)

__all__ = ["FeatureContext", "FeatureProvider", "available", "columns_for", "get",
           "register", "resolve"]
