"""Docking binaries resolve from config, environment, the repo, then PATH."""
import hashlib
import json
import os
import stat

import pytest

from qpocket.io import engines
from qpocket.io.tools import ToolNotFoundError


@pytest.fixture
def fake_vendor(tmp_path, monkeypatch):
    key = engines.platform_key()
    binary = tmp_path / key / "qvina2"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"#!/bin/sh\necho qvina\n")
    manifest = {"qvina2": {"version": "2.1", "license": "Apache-2.0", "source": "test",
                           "platforms": {key: {"file": "qvina2", "url": None,
                                               "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(engines, "VENDOR_DIR", tmp_path)
    engines._verified.clear()
    monkeypatch.delenv("QPOCKET_QVINA2", raising=False)
    return binary


def test_the_vendored_binary_is_verified_and_executable(fake_vendor):
    path = engines.vendored("qvina2")
    assert path == fake_vendor and os.stat(path).st_mode & stat.S_IXUSR


def test_a_tampered_binary_is_refused(fake_vendor):
    fake_vendor.write_bytes(b"changed")
    assert engines.vendored("qvina2") is None


def test_resolution_order(fake_vendor, monkeypatch, tmp_path):
    assert engines.resolve_executable("qvina2", "/opt/my/qvina") == "/opt/my/qvina"
    monkeypatch.setenv("QPOCKET_QVINA2", "/env/qvina")
    assert engines.resolve_executable("qvina2") == "/env/qvina"
    monkeypatch.delenv("QPOCKET_QVINA2")
    assert engines.resolve_executable("qvina2") == str(fake_vendor)


def test_a_missing_binary_names_every_place_looked(fake_vendor, monkeypatch):
    monkeypatch.setattr(engines.shutil, "which", lambda name: None)
    with pytest.raises(ToolNotFoundError, match="engine_path.*QPOCKET_NOPE.*vendor.*PATH"):
        engines.resolve_executable("nope")


def test_the_real_manifest_is_consistent():
    manifest = json.loads((engines.REAL_VENDOR_DIR / "manifest.json").read_text())
    assert set(manifest) >= {"vina", "qvina2"}
    for binary, entry in manifest.items():
        for key, spec in entry["platforms"].items():
            if spec and spec.get("sha256"):
                assert (engines.REAL_VENDOR_DIR / key / spec["file"]).exists(), (binary, key)
