# Cross-docking preparation fails clearly, and works — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `qpocket screen prepare` names the real reason when receptors cannot be prepared, `qpocket doctor` reports the `screen` extra, and S1-G127 of EV-A71 2A prepares on the laptop.

**Architecture:** `prepare_receptor` keeps returning `Optional[Path]` but records why it failed into an optional `failures` dict; `_prepare_in` collects them, raises `ReceptorPreparationError` when none succeeded and logs a warning listing failures otherwise. `qpocket.doctor.screen_extra()` probes `pdbfixer` and `openmm`; the doctor CLI prints them as warn rows like the docking engines.

**Tech Stack:** Python 3.11, click, eliot, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-27-explorer-pages-design.md` (Phase 0)

## Global Constraints

- The missing-import hint reads exactly: `install the screen extra: uv sync --extra screen`.
- Partial failure: the rest proceed; a warning lists the failed representatives and reasons.
- Doctor: the screen extra is an optional group (warn, never FAIL), like docking engines. It goes in the installed `qpocket doctor` (`src/qpocket/cli/doctor.py`), not the pre-install `scripts/qpocket_doctor.py` (*my call*: an extra can only be checked once installed).
- No dataset-specific code in the pipeline.

## Review Focus

- A failure that is not an ImportError (e.g. pdb2pqr crash) still appears with its own message in the all-failed error — Task 1 `test_every_receptor_failing_names_each_reason`.
- `receptor_from_pqr` returning False (no exception) counts as a failure with a reason, not a silent drop — Task 1 `test_a_receptor_conversion_failure_is_reported`.
- The previous snapshot survives an all-failed prepare — Task 1 `test_no_receptors_leaves_the_previous_snapshot`.
- 36 identical reasons must not make a 36-line error: reasons are grouped with counts — Task 1 `test_identical_reasons_are_grouped`.
- Doctor must not crash when an import raises something other than ImportError — Task 2 catches `Exception`.

---

### Task 1: receptor failures surface in prepare

**Files:**
- Modify: `src/qpocket/arms/screen/receptors.py:159-185`
- Modify: `src/qpocket/arms/screen/crossdock.py:169-200`
- Test: `tests/test_crossdock_prepare.py`

**Interfaces:**
- Produces, in `receptors.py`:
  - `class ReceptorPreparationError(RuntimeError)`
  - `prepare_receptor(cleaned_pdb, pocket, out_dir, rep_id, cfg, sequences=None, *, failures: Optional[Dict[str, str]] = None) -> Optional[Path]` — on failure sets `failures[rep_id] = "<ExceptionType>: <message>"`, or `"PDBQT conversion failed"` when `receptor_from_pqr` is False; still returns None and still logs.
  - `describe_failures(failures: Dict[str, str]) -> str` — groups identical reasons as `"<reason> (<n> representatives: R1, R2, …)"` (at most 5 ids, then `…`), reasons joined by `"; "`; a reason containing `No module named 'openmm'` or `No module named 'pdbfixer'` gets ` — install the screen extra: uv sync --extra screen`.
- `_prepare_in` passes a dict; when `receptors` is empty it raises `ReceptorPreparationError(f"no receptor could be prepared: {describe_failures(failures)}")` before `shared_box`; when some failed, `log_message(message_type="warning", stage="prepare_receptor", failed=describe_failures(failures))`.

- [ ] **Step 1: Write the failing tests** (reuse the `run` and `site` fixtures; a failing fake records into `failures` and returns None)

```python
def test_every_receptor_failing_names_each_reason(run, monkeypatch):
    # fake: R1 → failures["R1"] = "ImportError: No module named 'openmm'"; R2 → "RuntimeError: pdb2pqr exploded"
    with pytest.raises(ReceptorPreparationError) as e:
        ...  # the same prepare call the existing `run` tests make
    msg = str(e.value)
    assert "No module named 'openmm'" in msg and "uv sync --extra screen" in msg
    assert "pdb2pqr exploded" in msg and "no coordinates" not in msg

def test_some_receptors_failing_still_prepares_the_rest(run, monkeypatch):
    # R1 fails, R2 ok → box.json exists, one receptor in the plan, a warning event naming R1

def test_identical_reasons_are_grouped():
    assert describe_failures({"A": "X", "B": "X"}) == "X (2 representatives: A, B)"

def test_a_receptor_conversion_failure_is_reported(tmp_path, monkeypatch):
    # real prepare_receptor, cfg.receptor.repair False, receptors._protonate no-op, receptor_from_pqr → False
    failures = {}
    assert prepare_receptor(..., failures=failures) is None
    assert failures == {"R1": "PDBQT conversion failed"}

def test_no_receptors_leaves_the_previous_snapshot(site, monkeypatch):
    # like test_a_failed_prepare_leaves_the_previous_snapshot, with every receptor failing
```

- [ ] **Step 2: Run** `uv run pytest tests/test_crossdock_prepare.py -v` — Expected: the new tests FAIL (ImportError on `ReceptorPreparationError` / `describe_failures`).
- [ ] **Step 3: Implement** the Interfaces above. The existing test fakes take `sequences=None` only, so update them to accept `**kw` (or `failures=None`).
- [ ] **Step 4: Run** `uv run pytest tests/test_crossdock_prepare.py -v` — Expected: all PASS.
- [ ] **Step 5: Commit** `fix(screen): name why receptors failed instead of an empty-box error`

### Task 2: doctor reports the screen extra

**Files:**
- Modify: `src/qpocket/doctor.py` (new function), `src/qpocket/cli/doctor.py` (rows after the docking engines)
- Modify: `docs/installation.md:44-48`
- Test: `tests/test_doctor.py`

**Interfaces:**
- Produces: `screen_extra() -> List[Tuple[str, Optional[str]]]` in `qpocket/doctor.py`: `[("pdbfixer", None | "<Type>: <msg>"), ("openmm", …)]`, catching `Exception`; `importlib` imported at module level so tests can patch `doctor.importlib.import_module`.
- CLI prints `ok  import <module>  screen extra` or `warn  import <module>  <error> — uv sync --extra screen (only needed for `qpocket screen`)`; never appended to `failures`.

- [ ] **Step 1: Write the failing tests**

```python
def test_screen_extra_reports_a_missing_module(monkeypatch):
    from qpocket import doctor
    def fake(name):
        if name == "openmm":
            raise ImportError("No module named 'openmm'")
        return object()
    monkeypatch.setattr(doctor.importlib, "import_module", fake)
    assert dict(doctor.screen_extra()) == {"pdbfixer": None,
                                           "openmm": "ImportError: No module named 'openmm'"}

def test_doctor_warns_but_passes_without_the_screen_extra(monkeypatch):
    # CliRunner on qpocket.cli.doctor.doctor ["--no-smoke"]; patch doctor.screen_extra → openmm missing,
    # doctor.required_tools → {}, and qpocket.io.tools.apbs_command → ("test", "apbs")
    # assert "uv sync --extra screen" in result.output and "openmm" not in the "check(s) failed" line
```

- [ ] **Step 2: Run** `uv run pytest tests/test_doctor.py -v` — Expected: the new tests FAIL (`screen_extra` missing).
- [ ] **Step 3: Implement**; in `docs/installation.md` add `uv sync --extra screen  # also cross-docking (pdbfixer, openmm)` under the viz line.
- [ ] **Step 4: Run** `uv run pytest tests/test_doctor.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(doctor): report the screen extra; document it`

### Task 3: prepare S1-G127 on the laptop

No code; the spec's Phase 0 acceptance.

- [ ] **Step 1:** `uv sync --extra viz --extra screen` — Expected: `uv run python -c "import openmm, pdbfixer"` exits 0.
- [ ] **Step 2:** `uv run qpocket doctor --no-smoke` — Expected: `ok import pdbfixer`, `ok import openmm`.
- [ ] **Step 3:** `PATH=.venv/bin:$PATH uv run qpocket screen prepare --workdir ~/qpocket_runs/EV71_2A_cavity --site S1-G127` — Expected: finishes; the task folder holds `receptors/*.pdbqt` (count reported, ≤ 36), `ligands/*.pdbqt` and a `box.json` with finite bounds; any partial-failure warning is read and reported.
- [ ] **Step 4:** `PATH=.venv/bin:$PATH uv run pytest -q` — Expected: green; any failure reported by name.
