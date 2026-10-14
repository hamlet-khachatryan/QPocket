"""Docking binaries: the ones shipped with QPocket, and where else to look."""
import hashlib
import json
import os
import platform
import shutil
import stat
from pathlib import Path
from typing import Dict, Optional

from eliot import log_message

from qpocket.io.tools import ToolNotFoundError

REAL_VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "bin"
VENDOR_DIR = REAL_VENDOR_DIR
_verified: Dict[Path, bool] = {}


def platform_key() -> str:
    """This machine's vendor folder name.

    Returns:
        e.g. "linux-x86_64" or "macos-arm64".
    """
    system = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system(),
                                                                           platform.system().lower())
    machine = {"aarch64": "arm64", "AMD64": "x86_64"}.get(platform.machine(), platform.machine())
    return f"{system}-{machine}"


def vendored(binary: str) -> Optional[Path]:
    """The shipped binary for this platform, checked against its recorded sha256.

    Args:
        binary: Executable name, e.g. "qvina2".

    Returns:
        Its path, or None when none is shipped for this platform or the checksum fails.
    """
    manifest_path = VENDOR_DIR / "manifest.json"
    if not manifest_path.exists():
        return None
    spec = (json.loads(manifest_path.read_text(encoding="utf-8")).get(binary, {}).get("platforms", {})
            .get(platform_key()))
    if not spec:
        return None
    # A plain binary is one part; a dynamically linked one also lists the libraries it loads.
    parts = spec.get("parts") or [{"path": spec["file"], "sha256": spec.get("sha256")}]
    if not all(part.get("sha256") for part in parts):
        return None
    folder = VENDOR_DIR / platform_key()
    if not all(_checked(folder / part["path"], part["sha256"]) for part in parts):
        return None
    path = folder / spec["file"]
    mode = path.stat().st_mode
    if not mode & stat.S_IXUSR:
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _checked(path: Path, sha256: str) -> bool:
    if not path.exists():
        return False
    if path not in _verified:
        _verified[path] = hashlib.sha256(path.read_bytes()).hexdigest() == sha256
        if not _verified[path]:
            log_message(message_type="error", stage="vendored_engine", path=str(path),
                        error="sha256 mismatch; refusing to run it")
    return _verified[path]


def resolve_executable(binary: str, explicit: Optional[str] = None) -> str:
    """Where to run a docking binary from.

    Args:
        binary: Executable name.
        explicit: docking.engine_path, when set.

    Returns:
        The path or command to run.

    Raises:
        ToolNotFoundError: If none of the four places has it.
    """
    env = f"QPOCKET_{binary.upper().replace('-', '_')}"
    found = explicit or os.environ.get(env) or vendored(binary) or shutil.which(binary)
    if not found:
        raise ToolNotFoundError(
            f"{binary} not found: set docking.engine_path, or {env}, or add it to the vendor folder "
            f"({VENDOR_DIR / platform_key()}), or put it on PATH")
    return str(found)
