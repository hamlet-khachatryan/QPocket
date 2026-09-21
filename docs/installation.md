# Installation

## Check the machine first

One dependency, `apbs-binary`, ships only as platform wheels with no source build
to fall back on, so whether Q-Pocket installs is decided by the machine, not by
pip. Run the preflight **on the machine that will do the work** — on a cluster,
that means a compute node, not the login node:

```bash
python3 scripts/qpocket_doctor.py
```

It needs nothing but the standard library and runs on any Python from 2.7 and
3.4 up, so it can tell you when the system `python3` is too old instead of
failing on it.

### What can install

| | Python 3.9 – 3.12 |
|---|---|
| **Ubuntu 22.04+, Debian 12** | ✓ |
| **Rocky / RHEL / Alma 8 and 9** | ✓ |
| **macOS** (arm64, x86_64) | ✓ |
| **CentOS / RHEL 7** (glibc 2.17) | ✗ — no APBS wheel |
| **Any aarch64 / ARM** | ✗ — no APBS wheel |

This was produced by resolving the full dependency set for each platform. The
one constraint behind it is `apbs-binary`: x86_64 only, glibc ≥ 2.28.

Q-Pocket needs Python 3.9 or newer. The system `python3` on RHEL 7 and 8 is 3.6;
on a cluster, `module avail python` usually offers a newer one.

Structure alignment and cleaning used to need PyMOL, whose wheels require glibc
≥ 2.34 and ruled out RHEL 8 entirely. They now use Biopython. On the 20 PARG
structures every aligned atom lands within 0.0016 Å of where PyMOL put it.

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

The preflight names what is missing. For APBS on glibc < 2.28 or ARM, it has to
come from a site module or a separate build on `PATH`, or Q-Pocket has to run in
a container built on a supported base image.
