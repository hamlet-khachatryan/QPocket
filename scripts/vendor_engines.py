"""Download, verify and record the docking binaries QPocket ships (src/qpocket/vendor/bin)."""
import argparse
import base64
import hashlib
import json
import stat
import sys
import urllib.request
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "src/qpocket/vendor/bin"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"{url}: HTTP {response.status}")
        return response.read()


def main(record: bool) -> int:
    """Verify every recorded binary; with record, download missing ones and store their sha256.

    Args:
        record: Download and record instead of only verifying.

    Returns:
        Process exit code.
    """
    manifest_path = VENDOR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bad = 0
    for binary, entry in manifest.items():
        if record and entry.get("license_api"):
            payload = json.loads(_get(entry["license_api"]))
            text = base64.b64decode(payload["content"]).decode()
            (VENDOR / "LICENSES").mkdir(exist_ok=True)
            (VENDOR / "LICENSES" / f"{binary}.txt").write_text(text, encoding="utf-8")
        for key, spec in entry["platforms"].items():
            if not spec:
                continue
            path = VENDOR / key / spec["file"]
            if record and spec.get("url") and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_get(spec["url"]))
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                print(f"downloaded {binary} {key} ({path.stat().st_size} bytes)")
            if not path.exists():
                print(f"MISSING {binary} {key}: {path}")
                bad += 1
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if record:
                spec["sha256"] = digest
            elif digest != spec.get("sha256"):
                print(f"CHECKSUM MISMATCH {binary} {key}")
                bad += 1
    if record:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("ok" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="download missing binaries and record checksums")
    sys.exit(main(parser.parse_args().record))
