"""Run the fixed 7-structure PARG pipeline into a directory (offline, via the CLI)."""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

STRUCTURES = ["7KG6", "4B1H", "4A0D", "7KFP", "4B1J", "6O9Y", "6HMM"]
REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiments/PARG/7KG6/structures/original"


def golden_config(out: Path, workers: int = 0, features: list = None) -> dict:
    """Config for the golden run: fragment-screen mode, no network.

    Args:
        out: Working directory for the run.
        workers: compute.workers for the run.
        features: features.enabled, or None for the default set.

    Returns:
        Config mapping.
    """
    extra = {"compute": {"workers": workers}}
    if features:
        extra["features"] = {"enabled": list(features)}
    return {
        **extra,
        "mode": "fragment_screen",
        "reference": {"chain_id": "A", "pocket_definition": "Interactions",
                      "interaction_padding": 5.0},
        "paths": {"workdir": str(out)},
        "fragment_screen": {"input_type": "paths", "reference_path": str(SOURCE / "7KG6.cif"),
                            "structure_paths": [str(SOURCE / f"{p}.cif") for p in STRUCTURES]},
        "alignment": {"renumber_residues": False},
        "cleaning": {"allowed_cofactors": "CME"},
        "io": {"log_format": "json", "save_intermediate": False, "compression": "gzip"},
        "distance": {"parallel_jobs": 1},
    }


def main(out: Path, workers: int = 0, features: list = None) -> None:
    """Write the config and run ``qpocket run`` with it.

    Args:
        out: Working directory for the run.
        workers: compute.workers for the run.
        features: features.enabled, or None for the default set.
    """
    out.mkdir(parents=True, exist_ok=True)
    config = out / "golden.yaml"
    config.write_text(yaml.safe_dump(golden_config(out, workers, features)), encoding="utf-8")
    cli = Path(sys.executable).parent / ("qpocket.exe" if sys.platform == "win32" else "qpocket")
    subprocess.run([str(cli), "run", "--config", str(config)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--features", default=None,
                        help="Comma-separated providers, e.g. hbond,aromatic.")
    args = parser.parse_args()
    main(Path(args.out).resolve(), args.workers,
         args.features.split(",") if args.features else None)
