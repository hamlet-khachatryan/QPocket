"""Assign ligand protonation states for a given pH."""

from __future__ import annotations

from typing import List, Optional

from rdkit import Chem


def dominant_state(smiles: str, ph: float = 7.4) -> Optional[str]:
    """Dominant protonation state of a molecule at one pH.

    Args:
        smiles: SMILES of the neutral or as-drawn molecule.
        ph: pH at which to protonate.

    Returns:
        SMILES of the dominant state, or None when none could be produced.
    """
    from dimorphite_dl import protonate_smiles

    # precision=0 disables enumeration around each pKa, so exactly one state comes
    # back. With the default, several are returned and the first is not the
    # dominant one: glycine's first state carries a neutral amine.
    states: List[str] = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=0.0)
    return states[0] if states else None


def _neutral(mol: Chem.Mol) -> Chem.Mol:
    """Charge-free copy of a molecule, so two protonation states rank alike."""
    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    out = editable.GetMol()
    # Kekulisation is skipped: neutralising an aromatic cation leaves a ring RDKit
    # cannot kekulise, and canonical ranking only needs the graph.
    Chem.SanitizeMol(out, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES
                     ^ Chem.SANITIZE_KEKULIZE)
    return out


def assign_protonation(mol: Chem.Mol, ph: float = 7.4) -> Chem.Mol:
    """Copy of a ligand with formal charges for its dominant state at a pH.

    Args:
        mol: Heavy-atom ligand, as perceived from its component definition.
        ph: pH at which to protonate.

    Returns:
        A copy carrying the formal charges of the dominant state. The input is
        returned unchanged when no state could be assigned.
    """
    flat = Chem.RemoveHs(Chem.Mol(mol))
    state_smiles = dominant_state(Chem.MolToSmiles(flat), ph)
    if state_smiles is None:
        return mol
    state = Chem.MolFromSmiles(state_smiles)
    if state is None or state.GetNumAtoms() != flat.GetNumAtoms():
        return mol

    # Canonical ranking of the neutral graphs is a bijection between the two
    # molecules. Substructure matching is not: on a symmetric skeleton it can map
    # a charge onto the wrong atom of an equivalent pair.
    try:
        ours = list(Chem.CanonicalRankAtoms(_neutral(flat)))
        theirs = list(Chem.CanonicalRankAtoms(_neutral(state)))
    except Exception:
        return mol
    if sorted(ours) != sorted(theirs):
        return mol
    by_rank = {rank: index for index, rank in enumerate(theirs)}

    out = Chem.RWMol(flat)
    for index, rank in enumerate(ours):
        source = state.GetAtomWithIdx(by_rank[rank])
        # Aromatic sites are left alone. Their pKa sits near physiological pH (a
        # xanthine N-H is about 7.4), and charging one can cost the ring its
        # aromaticity, which silently removes every pi-stacking interaction.
        if source.GetFormalCharge() and not out.GetAtomWithIdx(index).GetIsAromatic():
            atom = out.GetAtomWithIdx(index)
            atom.SetFormalCharge(source.GetFormalCharge())
            atom.SetNoImplicit(False)
            # The hydrogen count travels with the charge: deprotonating an
            # aromatic N-H leaves valence 4 if its hydrogen is left behind.
            atom.SetNumExplicitHs(0)
    result = out.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception:
        return mol
    return result
