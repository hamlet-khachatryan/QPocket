# Bundled docking binaries

| Binary | Version | Licence | linux-x86_64 | linux-arm64 | macos-arm64 | macos-x86_64 |
|---|---|---|---|---|---|---|
| `vina` (AutoDock Vina) | 1.2.7 | Apache-2.0 | yes | yes | yes | yes |
| `qvina2` (QuickVina 2.1) | 2.1 | Apache-2.0 | yes | — | yes | yes |

Upstream publishes QuickVina 2 for x86-64 Linux only. The macOS builds come from conda-forge
(`qvina` 2.1.0 build 8) and link Boost dynamically, so each one ships as `qvina2/bin/qvina2` plus the
four Boost 1.84 libraries it loads from `qvina2/lib/` (Boost Software License, `LICENSES/boost.txt`).
libc++ comes from macOS itself. Nothing needs conda at run time.

On ARM Linux, QPocket looks for `qvina2` in `docking.engine_path`, then `$QPOCKET_QVINA2`, then `PATH`.
To ship your own build, add the platform entry to `manifest.json` (a single `file` and `url`, or
`parts`, each with a `url` and an optional conda-package `member`) and run
`python scripts/vendor_engines.py --record`.

`manifest.json` records each file's sha256; QPocket refuses to run a binary when any of its files' checksums differ.
Verify everything with `python scripts/vendor_engines.py`.
