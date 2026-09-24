"""Eliot log destinations for a run."""
from __future__ import annotations

import atexit
from pathlib import Path

from typing import TYPE_CHECKING

from eliot import add_destinations, to_file

if TYPE_CHECKING:
    from qpocket.config import QPocketConfig


def configure(cfg: QPocketConfig, log_path: Path) -> None:
    """Sets up Eliot structured logging to file (and optionally stdout).

    Args:
        cfg: QPocket configuration (reads ``cfg.io.log_format``).
        log_path: Path to the log output file.
    """
    log_file = open(str(log_path), "w", encoding="utf-8")
    atexit.register(log_file.close)

    to_file(log_file)

    if cfg.io.log_format == "text":
        # eliot 1.15 replaced pretty_print, which printed for us, with
        # pretty_format, which returns the string and leaves printing to us.
        from eliot.prettyprint import pretty_format

        add_destinations(lambda message: print(pretty_format(message), end=""))
