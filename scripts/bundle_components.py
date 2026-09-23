"""Download the component definitions that ship with the package."""
import sys
import urllib.request
from pathlib import Path

COMPONENTS = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
              "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
              "HOH", "MSE", "ZN", "MG", "CA", "NA", "K", "CL", "FE", "MN", "SO4", "GOL"]
OUT = Path(__file__).resolve().parents[1] / "src/qpocket/data/ccd"


def main() -> None:
    """Fetch every bundled component into the package data directory."""
    OUT.mkdir(parents=True, exist_ok=True)
    for comp in COMPONENTS:
        target = OUT / f"{comp}.cif"
        if target.exists():
            continue
        url = f"https://files.rcsb.org/ligands/download/{comp}.cif"
        with urllib.request.urlopen(url, timeout=30) as response:
            target.write_text(response.read().decode("utf-8"), encoding="utf-8")
        print("fetched", comp)


if __name__ == "__main__":
    sys.exit(main())
