# Installation

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.9 | Tested on 3.9 – 3.12 |
| PyMOL | Open-source or incentive version (not on PyPI — see below) |
| APBS | System binary wrapped by `apbs-binary` pip package |
| pdb2pqr | Installed via pip |

---

## 1. Install PyMOL

QPocket uses PyMOL for structure alignment and cleaning.  PyMOL is **not** available on PyPI; install it via **conda** (recommended) or from a pre-built wheel:

```bash
# Conda (recommended)
conda install -c conda-forge pymol-open-source

# Or — open-source wheel from GitHub releases
pip install pymol-open-source  # if a wheel exists for your platform
```

## 2. Install QPocket

### From source (editable)

```bash
git clone https://github.com/denovosciences/QPocket.git
cd QPocket
pip install -e ".[dev]"
```

### From PyPI (once published)

```bash
pip install QPocket
```

### With optional extras

```bash
# Development + docs
pip install "QPocket[dev]"

# Fragment screening decoy generation
pip install "QPocket[screen]"
```

---

## 3. Verify installation

```bash
qpocket --version
qpocket --help
```

---

## 4. External binaries

### APBS

`apbs-binary` (a Python wrapper that bundles the APBS static binary) is declared as a core dependency and will be installed automatically by pip.  Verify:

```bash
python -c "from apbs_binary import popen_apbs; print('APBS OK')"
```

### pdb2pqr

Also installed automatically.  Verify:

```bash
pdb2pqr --version
```

---

## 5. PLIP

PLIP (Protein–Ligand Interaction Profiler) is bundled inside `src/plip/` as a git submodule.  No separate installation is required — it is importable once QPocket is installed in editable mode.

```bash
# If you cloned without --recursive, initialise the submodule:
git submodule update --init --recursive
```