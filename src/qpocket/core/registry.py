"""A named collection of plug-ins for one port (pure)."""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Generic, NamedTuple, Optional, Sequence, Tuple, TypeVar

F = TypeVar("F", bound=Callable)


@dataclasses.dataclass(frozen=True)
class NoParams:
    """Parameters of a plug-in that takes none."""


class Resources(NamedTuple):
    """What one call of a plug-in needs from the machine.

    Attributes:
        cpus: Cores.
        mem_gb: Memory in GB.
        gpu: GPUs.
        walltime_min: Wall time per item, in minutes.
        heavy: Worth a SLURM array under compute.backend "auto".
        cache: Results are cached by input hash.
    """

    cpus: int = 1
    mem_gb: float = 2.0
    gpu: int = 0
    walltime_min: int = 30
    heavy: bool = False
    cache: bool = False


class Plugin(NamedTuple):
    """One registered implementation of a port.

    Attributes:
        name: Name used in config.
        fn: The implementation.
        params: Frozen dataclass of its parameters.
        resources: What one call needs.
        tools: Executables it runs, for doctor.
        columns: Columns it produces (feature channels only).
        requires: Stages it needs, e.g. "charges".
    """

    name: str
    fn: Callable
    params: type
    resources: Resources
    tools: Tuple[str, ...]
    columns: Tuple[str, ...]
    requires: Tuple[str, ...]


class Registry(Generic[F]):
    """Plug-ins for one port, looked up by name.

    Args:
        kind: What the plug-ins are, for error messages.
        loader: Called once before the first lookup, to import the built-ins.
    """

    def __init__(self, kind: str, loader: Optional[Callable[[], None]] = None):
        self.kind = kind
        self._loader = loader
        self._plugins: Dict[str, Plugin] = {}

    def register(self, name: str, *, params: type = NoParams, resources: Resources = Resources(),
                 tools: Sequence[str] = (), columns: Sequence[str] = (),
                 requires: Sequence[str] = ()) -> Callable[[F], F]:
        """Decorator adding a function under ``name``.

        Args:
            name: Name used in config.
            params: Frozen dataclass of its parameters.
            resources: What one call needs.
            tools: Executables it runs.
            columns: Columns it produces.
            requires: Stages it needs.

        Returns:
            A decorator that returns the function unchanged.

        Raises:
            ValueError: If ``name`` is already registered.
            TypeError: If ``params`` is not a frozen dataclass.
        """
        if not (dataclasses.is_dataclass(params) and params.__dataclass_params__.frozen):
            raise TypeError(f"{self.kind} {name!r}: params must be a frozen dataclass")

        def add(fn: F) -> F:
            if name in self._plugins:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._plugins[name] = Plugin(name, fn, params, resources, tuple(tools),
                                         tuple(columns), tuple(requires))
            return fn

        return add

    def _load(self) -> None:
        if self._loader is not None:
            loader, self._loader = self._loader, None
            loader()

    def available(self) -> Tuple[str, ...]:
        """Registered names, in registration order.

        Returns:
            The names.
        """
        self._load()
        return tuple(self._plugins)

    def get(self, name: str) -> Plugin:
        """One plug-in by name.

        Args:
            name: Registered name.

        Returns:
            The plug-in.

        Raises:
            ValueError: If no plug-in has that name; the message lists the available ones.
        """
        self._load()
        if name not in self._plugins:
            raise ValueError(f"unknown {self.kind} {name!r}; available: {', '.join(self._plugins)}")
        return self._plugins[name]

    def resolve(self, names: Sequence[str]) -> Tuple[Plugin, ...]:
        """Several plug-ins, in the given order.

        Args:
            names: Registered names.

        Returns:
            The plug-ins.
        """
        return tuple(self.get(name) for name in names)
