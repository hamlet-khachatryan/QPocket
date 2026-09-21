# Installation

## Check the machine first

Two dependencies ship only as platform wheels with no source build to fall back
on, so whether Q-Pocket installs at all is decided by the machine, not by pip.
Run the preflight **on the machine that will do the work** — on a cluster, that
means a compute node, not the login node:

```bash
git clone https://github.com/denovosciences/QPocket.git
cd QPocket
python3 scripts/qpocket_doctor.py
```

It needs nothing but the standard library, so it runs before anything is
installed, and it prints the exact install command for that machine.

### What can install

| | Python 3.10 | 3.11 | 3.12 |
|---|---|---|---|
| **Ubuntu 22.04+, Debian 12** (glibc ≥ 2.35) | ✓ | ✓ | ✓ |
| **Rocky / RHEL / Alma 9** (glibc 2.34) | ✗ | ✓ | ✗ |
| **Rocky / RHEL / CentOS 8** (glibc 2.28) | ✗ | ✗ | ✗ |
| **Any aarch64 / ARM** | ✗ | ✗ | ✗ |
| **macOS** (arm64, x86_64) | ✗ via pip — see below | | |

This table was produced by resolving the full dependency set for each platform,
not from the package pages. Resolving is not the same as working, which is why
`qpocket doctor` (below) must pass on the target machine: the macOS
`pymol-open-source` 3.2.0a0 wheel installs cleanly and then fails to import,
because it was linked against a library path on its builder's machine. On macOS
use a conda environment with `conda install -c conda-forge pymol-open-source`. The constraints behind it:

| package | Linux wheels |
|---|---|
| `pymol-open-source` 3.2.0a0 | x86_64, glibc ≥ 2.35, CPython 3.9–3.13 |
| `pymol-open-source` 3.1.0a0 | x86_64, glibc ≥ 2.34, **CPython 3.11 only** |
| `apbs-binary` 3.4.1 | x86_64, glibc ≥ 2.28 — **no aarch64 build** |

On Rocky 9, use Python 3.11 — often `module load python/3.11` on a cluster —
and pip will select the 3.1.0a0 PyMOL wheel on its own.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

A very old system pip can fail building from `pyproject.toml` with
`'Distribution' has no attribute '_finalize_feature_opts'`; upgrading pip first
avoids it.

Optional extras:

```bash
pip install -e ".[viz]"   # the marimo explorer notebook
pip install -e ".[dev]"   # tests and documentation
```

PLIP and CavitOmiX are vendored under `src/` and install with the package;
there is nothing extra to fetch.

## Verify

```bash
qpocket doctor
```

This goes further than the preflight: it imports every dependency and then
**runs** pdb2pqr and APBS on a tiny test system. That last part matters on
Linux, where the `apbs-binary` wheel imports fine even if its binary cannot load
its shared libraries — only running it tells you.

```
  ok     platform               Linux x86_64, glibc 2.35
  ok     import pymol           
  ok     import apbs_binary     featurisation
  ok     pdb2pqr runs           wrote 888 bytes in 1.0s
  ok     apbs runs              wrote 468 KiB grid in 1.0s

This node can run the pipeline.
```

On a cluster, run it where the jobs will run:

```bash
srun -p <partition> --pty .venv/bin/qpocket doctor
```

## When the machine is not supported

The preflight names what is missing. The usual fixes:

- **PyMOL has no wheel** (Rocky 8, or Rocky 9 on a Python other than 3.11): use
  Python 3.11 on Rocky 9; otherwise there is no pip route. PyMOL is imported
  when the package loads, so every `qpocket` command needs it, not only the
  alignment stage.
- **APBS has no wheel** (glibc < 2.28, or ARM): APBS must come from a site module
  or a separate build, on `PATH`.

Either is a reason to consider a container image built on a supported base,
which sidesteps the host's glibc entirely.
