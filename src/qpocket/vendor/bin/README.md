# Bundled docking binaries

| Binary | Version | Licence | linux-x86_64 | linux-arm64 | macos-arm64 | macos-x86_64 |
|---|---|---|---|---|---|---|
| `vina` (AutoDock Vina) | 1.2.7 | Apache-2.0 | yes | yes | yes | yes |
| `qvina2` (QuickVina 2.1) | 2.1 | Apache-2.0 | yes | — | — | — |

Upstream publishes no QuickVina 2 build for ARM Linux or macOS. On those machines QPocket looks for
`qvina2` in `docking.engine_path`, then `$QPOCKET_QVINA2`, then `PATH`. To ship your own build, drop it
in `<platform>/qvina2`, add the platform entry to `manifest.json`, and run
`python scripts/vendor_engines.py --record`.

`manifest.json` records each file's sha256; QPocket refuses to run a binary whose checksum differs.
Verify everything with `python scripts/vendor_engines.py`.
