"""Run the fixed 7-structure PARG pipeline into a directory (offline, via the CLI)."""
import subprocess
import sys
from pathlib import Path

import yaml

STRUCTURES = ["7KG6", "4B1H", "4A0D", "7KFP", "4B1J", "6O9Y", "6HMM"]
REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiments/PARG/7KG6/structures/original"


def golden_config(out: Path) -> dict:
    """Config for the golden run: fragment-screen mode, no network.

    Args:
        out: Working directory for the run.

    Returns:
        Config mapping.
    """
    return {
        "mode": "fragment_screen",
        "reference": {"chain_id": "A", "ligand_id": "WDG", "pocket_definition": "Interactions",
                      "pocket_mode": "cavity_detection", "interaction_padding": 5.0},
        "paths": {"workdir": str(out)},
        "fragment_screen": {"input_type": "paths", "reference_path": str(SOURCE / "7KG6.cif"),
                            "structure_paths": [str(SOURCE / f"{p}.cif") for p in STRUCTURES]},
        "alignment": {"renumber_residues": False},
        "cleaning": {"allowed_cofactors": "CME"},
        "io": {"log_format": "json", "save_intermediate": False, "compression": "gzip"},
        "distance": {"parallel_jobs": 1},
    }


def main(out: Path) -> None:
    """Write the config and run ``qpocket run`` with it.

    Args:
        out: Working directory for the run.
    """
    out.mkdir(parents=True, exist_ok=True)
    config = out / "golden.yaml"
    config.write_text(yaml.safe_dump(golden_config(out)), encoding="utf-8")
    cli = Path(sys.executable).parent / ("qpocket.exe" if sys.platform == "win32" else "qpocket")
    subprocess.run([str(cli), "run", "--config", str(config)], check=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
