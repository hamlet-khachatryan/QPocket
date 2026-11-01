"""Docking binaries resolve from config, environment, the repo, then PATH."""
import hashlib
import json
import os
import stat

import pytest

from qpocket.io import engines


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


@pytest.fixture
def fake_multipart(tmp_path, monkeypatch):
    """An engine shipped as an executable plus the libraries it loads."""
    key = engines.platform_key()
    exe = tmp_path / key / "qvina2" / "bin" / "qvina2"
    lib = tmp_path / key / "qvina2" / "lib" / "libboost_thread.dylib"
    for path, data in ((exe, b"#!/bin/sh\necho qvina\n"), (lib, b"library")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    manifest = {"qvina2": {"version": "2.1", "license": "Apache-2.0", "source": "test", "platforms": {key: {
        "file": "qvina2/bin/qvina2",
        "parts": [{"path": "qvina2/bin/qvina2", "url": None, "member": "bin/qvina2", "sha256": digest(exe)},
                  {"path": "qvina2/lib/libboost_thread.dylib", "url": None,
                   "member": "lib/libboost_thread.dylib", "sha256": digest(lib)}]}}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(engines, "VENDOR_DIR", tmp_path)
    engines._verified.clear()
    return exe, lib


