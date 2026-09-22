"""Immutable values passed between core, io and pipelines."""
import re
from typing import NamedTuple, Optional, Tuple

import numpy as np


class Scaler(NamedTuple):
    """Per-feature standardisation fitted over a set of pockets.

    Attributes:
        mean: Mean of each feature column.
        scale: Standard deviation of each column (1.0 where constant).
    """

    mean: np.ndarray
    scale: np.ndarray


class Fingerprint(NamedTuple):
    """Spectral fingerprint of one pocket.

    Attributes:
        eigenvalues: Laplacian eigenvalues, padded or truncated to global_k.
        spectrum: Flattened graph-Fourier power spectrum of the features.
    """

    eigenvalues: np.ndarray
    spectrum: np.ndarray


class StructureData(NamedTuple):
    """Protein atoms and pharmacophore sites used to featurise a pocket.

    Attributes:
        coords: (M, 3) atom coordinates from the PQR.
        charges: (M,) partial charges.
        hydrophobicity: (M,) per-atom hydrophobicity.
        donors: (K, 3) H-bond donor atoms.
        acceptors: (K, 3) H-bond acceptor atoms.
        aromatics: (K, 3) aromatic ring centres.
    """

    coords: np.ndarray
    charges: np.ndarray
    hydrophobicity: np.ndarray
    donors: np.ndarray
    acceptors: np.ndarray
    aromatics: np.ndarray

    @classmethod
    def empty(cls) -> "StructureData":
        """Atom-free inputs, for runs whose providers need no PQR.

        Returns:
            All arrays zero-length.
        """
        return cls(np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                   np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))


class PocketSpec(NamedTuple):
    """Where a pocket is: a structure, an optional chain, and a ligand or residues.

    Attributes:
        structure: PDB ID (downloaded) or a local PDB/mmCIF path.
        chain: Keep only this chain, or None for all.
        ligand: Ligand residue name marking the pocket.
        residues: Residues lining the pocket, e.g. ("TYR795", "GLU756").
        padding: Distance (A) from those residues that counts as the pocket.
    """

    structure: str
    chain: Optional[str] = None
    ligand: Optional[str] = None
    residues: Tuple[str, ...] = ()
    padding: float = 5.0

    def validate(self) -> "PocketSpec":
        """Check that exactly one pocket definition is given.

        Returns:
            The spec itself.

        Raises:
            ValueError: If neither or both of ligand and residues are set.
        """
        if bool(self.ligand) == bool(self.residues):
            raise ValueError(f"{self.structure}: give a ligand or residues, not both or neither")
        return self

    @classmethod
    def parse(cls, text: str, padding: float = 5.0) -> "PocketSpec":
        """Parse STRUCTURE[@CHAIN]:POCKET, e.g. 7KG6:WDG or a.pdb@A:TYR795,GLU756.

        Args:
            text: Spec string; POCKET is a ligand code or comma-separated residues.
            padding: Pocket padding for residue definitions.

        Returns:
            The spec.

        Raises:
            ValueError: On a malformed string.
        """
        if ":" not in text:
            raise ValueError(f"{text!r}: expected STRUCTURE[@CHAIN]:POCKET")
        where, pocket = text.rsplit(":", 1)  # right split keeps Windows drive letters
        structure, _, chain = where.rpartition("@") if "@" in where else (where, "", "")
        items = tuple(p.strip().upper() for p in pocket.split(",") if p.strip())
        is_residues = len(items) > 1 or bool(re.fullmatch(r"[A-Z]{3}-?\d+[A-Z]?", items[0] if items else ""))
        spec = cls(structure=structure, chain=chain or None,
                   ligand=None if is_residues else (items[0] if items else None),
                   residues=items if is_residues else (), padding=padding)
        return spec.validate()


class ComparisonResult(NamedTuple):
    """Pairwise distances between compared pockets.

    Attributes:
        ids: Pocket labels, in matrix order.
        shape: Shape distance matrix.
        chem: Chemistry distance matrix.
        combined: Blended distance matrix.
        scaler: Feature scaling used for the fingerprints.
    """

    ids: Tuple[str, ...]
    shape: np.ndarray
    chem: np.ndarray
    combined: np.ndarray
    scaler: Scaler

    def to_frame(self, kind: str = "combined"):
        """One matrix as a labelled DataFrame.

        Args:
            kind: "combined", "shape" or "chem".

        Returns:
            Square pandas DataFrame.
        """
        import pandas as pd

        return pd.DataFrame(getattr(self, kind), index=list(self.ids), columns=list(self.ids))


class PQRAtoms(NamedTuple):
    """Atoms of a PQR file (pdb2pqr output, ions repaired).

    Attributes:
        coords: (M, 3) coordinates.
        charges: (M,) partial charges.
        radii: (M,) atomic radii.
        atom_names: Atom names.
        res_names: Residue names (AMBER protonation names).
        chain_ids: Chain IDs; empty for repaired ions.
        res_seqs: Residue numbers.
    """

    coords: np.ndarray
    charges: np.ndarray
    radii: np.ndarray
    atom_names: list
    res_names: list
    chain_ids: list
    res_seqs: list
