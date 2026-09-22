"""Time the 7-structure PARG run at several worker counts."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(out_root: Path, counts=(1, 2, 4, 0)) -> None:
    """Run the golden pipeline at each worker count and print a Markdown table.

    Args:
        out_root: Scratch directory for the runs.
        counts: Worker settings to try (0 = half the cores).
    """
    print("| workers | seconds |\n|---|---|")
    for workers in counts:
        target = out_root / f"w{workers}"
        started = time.time()
        subprocess.run([sys.executable, str(REPO / "scripts/make_golden.py"), str(target),
                        "--workers", str(workers)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"| {workers} | {time.time() - started:.0f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--counts", default="1,2,4,0")
    args = parser.parse_args()
    main(Path(args.out).resolve(), [int(c) for c in args.counts.split(",")])
