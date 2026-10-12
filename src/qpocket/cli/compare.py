"""CLI: featurize one pocket, compare saved pockets, or compare straight from structures."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import numpy as np

_FINGERPRINT_OPTIONS = [
    click.option("--scaler", type=click.Path(exists=True), default=None,
                 help="scaler.npz, or a run directory holding metadata/scaler.npz, "
                      "to compare on a finished run's normalisation."),
    click.option("--global-k", default=100, show_default=True, help="Fingerprint length."),
    click.option("--sigma", default=10.0, show_default=True, help="Heat-kernel width."),
    click.option("--radius", default=2.0, show_default=True, help="Geodesic neighbour radius (A)."),
    click.option("-o", "--output", type=click.Path(), default=None, help="Write the combined matrix as CSV."),
]


def _fingerprint_options(fn):
    for option in reversed(_FINGERPRINT_OPTIONS):
        fn = option(fn)
    return fn


def _load_scaler(path: Optional[str]):
    if path is None:
        return None
    from qpocket.io.cache import load_scaler

    p = Path(path)
    if p.is_dir():
        found = sorted(p.glob("metadata/scaler.npz")) + sorted(p.glob("*/metadata/scaler.npz"))
        if not found:
            raise click.BadParameter(f"no metadata/scaler.npz under {p}", param_hint="--scaler")
        p = found[0]
    return load_scaler(p)


def _report(result, output: Optional[str]) -> None:
    frame = result.to_frame()
    click.echo("combined distance")
    click.echo(frame.round(4).to_string())
    click.echo("\nshape distance")
    click.echo(result.to_frame("shape").round(4).to_string())
    if output:
        frame.to_csv(output)
        click.echo(f"\nwrote {output}")


@click.command("featurize")
@click.argument("structure")
@click.option("--chain", default=None, help="Keep only this chain.")
@click.option("--ligand", default=None, help="Ligand residue name marking the pocket.")
@click.option("--residues", default=None, help="Comma-separated residues lining the pocket, e.g. TYR795,GLU756.")
@click.option("--padding", default=5.0, show_default=True, help="Pocket radius around --residues (A).")
@click.option("--features", "features_list", default=None,
              help="Comma-separated feature providers; defaults to all four.")
@click.option("-o", "--output", type=click.Path(), required=True, help="Output .npz (points, features).")
def featurize_command(structure, chain, ligand, residues, padding, features_list, output):
    """Find and featurise one pocket in STRUCTURE (a PDB ID or a file)."""
    from qpocket.core.features import DEFAULT_PROVIDERS
    from qpocket.core.features import columns_for
    from qpocket.core.ports import FEATURES
    from qpocket.arms.compare import featurize
    from qpocket.types import PocketSpec

    spec = PocketSpec(structure, chain, ligand,
                      tuple(r.strip().upper() for r in (residues or "").split(",") if r.strip()), padding)
    providers = ([f.strip() for f in features_list.split(",") if f.strip()]
                 if features_list else list(DEFAULT_PROVIDERS))
    try:
        columns = columns_for(FEATURES.resolve(providers))
        points, features = featurize(spec.validate(), features=providers)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc))
    np.savez(output, points=points, features=features, feature_names=np.array(columns),
             structure=spec.structure, pocket=spec.ligand or ",".join(spec.residues))
    click.echo(f"{len(points)} pocket points -> {output}")


@click.command("compare")
@click.argument("pockets", nargs=-1, required=True, type=click.Path(exists=True))
@_fingerprint_options
def compare_command(pockets, scaler, global_k, sigma, radius, output):
    """Compare pockets saved by `qpocket featurize` (two or more .npz files)."""
    from qpocket.arms.compare import compare_features

    if len(pockets) < 2:
        raise click.UsageError("give at least two pockets")
    loaded, ids, column_sets = [], [], set()
    for path in pockets:
        with np.load(path) as data:
            loaded.append((data["points"], data["features"]))
            column_sets.add(tuple(str(c) for c in data.get("feature_names", [])))
        ids.append(Path(path).stem)
    if len(column_sets) > 1:
        raise click.ClickException(
            "these pockets were featurised with different providers: "
            + "; ".join(", ".join(c) or "unknown" for c in sorted(column_sets))
        )
    try:
        result = compare_features(loaded, ids, _load_scaler(scaler),
                                  global_k=global_k, sigma=sigma, radius=radius)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    _report(result, output)


@click.command("compare-structures")
@click.argument("pockets", nargs=-1, required=True)
@click.option("--padding", default=5.0, show_default=True, help="Pocket radius around listed residues (A).")
@_fingerprint_options
def compare_structures_command(pockets, padding, scaler, global_k, sigma, radius, output):
    """Compare pockets given as STRUCTURE[@CHAIN]:LIGAND or STRUCTURE[@CHAIN]:RES,RES.

    STRUCTURE is a PDB ID (downloaded) or a file. Example:
    qpocket compare-structures 7KG6:WDG 4CPN@A:TYR795,GLU756
    """
    from qpocket.arms.compare import run
    from qpocket.types import PocketSpec

    if len(pockets) < 2:
        raise click.UsageError("give at least two pockets")
    try:
        specs = [PocketSpec.parse(p, padding=padding) for p in pockets]
        result = run(specs, scaler=_load_scaler(scaler), global_k=global_k, sigma=sigma, radius=radius)
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc))
    _report(result, output)
