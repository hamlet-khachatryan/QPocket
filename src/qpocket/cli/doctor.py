"""Checking that this machine can run the pipeline."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import click


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), default=None,
              help="Also check that this configuration loads.")
@click.option("--smoke/--no-smoke", default=True,
              help="Run APBS on a tiny synthetic system (a few seconds).")
def doctor(config: Optional[str], smoke: bool):
    """Check that this machine can actually run the pipeline.

    Reports the platform, every import, the external tools, and -- unless
    --no-smoke -- runs APBS for real, which is the check that matters on Linux:
    apbs-binary only patches its library path on macOS, so a wheel that imports
    is not proof of a binary that runs.

    Run it on a compute node, not the login node; they are often different.

    Example:
    \b
    qpocket doctor
    srun -p compute qpocket doctor      # where the work will actually happen
    """
    import importlib
    import platform
    import shutil

    failures: List[str] = []

    def line(status: str, label: str, detail: str = "") -> None:
        click.echo(f"  {status:<6} {label:<22} {detail}")

    def ok(label, detail=""):
        line("ok", label, detail)

    def bad(label, detail=""):
        line("FAIL", label, detail)
        failures.append(label)

    click.echo("Q-Pocket doctor\n")

    # --- platform ---------------------------------------------------------
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from qpocket_doctor import fmt, glibc_version  # type: ignore

        libc = f", glibc {fmt(glibc_version())}" if platform.system() == "Linux" else ""
    except Exception:
        libc = ""
    ok("platform", f"{platform.system()} {platform.machine()}{libc}")
    ok("python", f"{platform.python_version()} at {sys.executable}")

    # --- imports ----------------------------------------------------------
    for module, why in (
        ("numpy", ""), ("scipy", ""), ("pandas", ""), ("sklearn", ""),
        ("h5py", ""), ("joblib", ""), ("hydra", ""), ("omegaconf", ""),
        ("eliot", ""), ("Bio", "biopython"), ("biopandas", ""), ("rdkit", ""),
        ("prolif", ""), ("dimorphite_dl", ""), ("pdb2pqr", ""),
    ):
        try:
            importlib.import_module(module)
            ok(f"import {module}", why)
        except Exception as e:
            bad(f"import {module}", f"{type(e).__name__}: {e}")

    # --- APBS: the wheel, QPOCKET_APBS, or an apbs on PATH ---------------------
    from qpocket.io.tools import ToolNotFoundError, apbs_command

    try:
        source, detail = apbs_command()
        ok("apbs", f"{detail} ({source})")
    except ToolNotFoundError as e:
        bad("apbs", str(e))

    # --- external tools ---------------------------------------------------
    from qpocket import doctor as checks
    from qpocket.config import QPocketConfig, load_qpocket_config

    try:
        run_cfg = load_qpocket_config(config_path=config) if config else QPocketConfig()
    except Exception:
        run_cfg = QPocketConfig()  # the config check below says why it failed
    for tool, users in checks.required_tools(run_cfg).items():
        found = shutil.which(tool)
        if found:
            ok(f"{tool} on PATH", found)
        else:
            bad(f"{tool} on PATH", f"not found (needed by {', '.join(users)})")
    found = shutil.which("sbatch")
    if found:
        ok("sbatch on PATH", found)
    else:
        line("warn", "sbatch on PATH", "not found (only needed for `qpocket slurm`)")

    # --- docking engines (only `qpocket screen` needs them) ---------------
    from qpocket.core.ports import DOCKING
    from qpocket.io.engines import resolve_executable

    for name in DOCKING.available():
        engine = DOCKING.get(name).fn()
        try:
            ok(f"docking {name}", resolve_executable(engine.executable))
        except Exception as e:
            line("warn", f"docking {name}", str(e).split(":")[0] + " (only needed for `qpocket screen`)")

    # --- the screen extra (only `qpocket screen` needs it) ----------------
    for module, error in checks.screen_extra():
        if error is None:
            ok(f"import {module}", "screen extra")
        else:
            line("warn", f"import {module}",
                 f"{error} — uv sync --extra screen, plus --extra viz for the explorer (only needed for `qpocket screen`)")

    # --- APBS actually running -------------------------------------------
    if smoke:
        for label, probe in (
            ("pdb2pqr runs", checks.run_pdb2pqr_smoke_test),
            ("apbs runs", checks.run_apbs_smoke_test),
        ):
            try:
                seconds, detail = probe()
                ok(label, f"{detail} in {seconds:.1f}s")
            except Exception as e:
                bad(label, f"{type(e).__name__}: {e}")

    # --- the configuration ------------------------------------------------
    if config:
        try:
            from qpocket.config import load_qpocket_config

            load_qpocket_config(config_path=config)
            ok("config loads", config)
        except Exception as e:
            bad("config loads", f"{type(e).__name__}: {e}")

    click.echo()
    if failures:
        click.echo(f"{len(failures)} check(s) failed: {', '.join(failures)}", err=True)
        raise click.Abort()
    click.echo("This node can run the pipeline.")
