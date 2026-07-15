from rdkit import Chem

from xyz_std.h_ordering import _order_h_on_heavy_atom


def _parse_heavy_order_from_auxinfo(aux_info: str) -> list[int]:
    """
    Extract the heavy atom order from InChI AuxInfo /N: section.

    The /N: section lists heavy atom indices (1-based) in InChI canonical order.

    Args:
        aux_info: AuxInfo string from Chem.MolToInchiAndAuxInfo

    Returns:
        List of heavy atom indices (0-based) in InChI canonical order
    """
    for section in aux_info.split('/'):
        if section.startswith('N:'):
            return [int(idx) - 1 for idx in section[2:].split(',')]

    raise ValueError(f"N: layer not found in AuxInfo: {aux_info}")


def get_standard_atom_order(mol: Chem.Mol) -> list[int]:
    """
    Compute standard atom order from an RDKit Mol with explicit H and 3D conformer.

    Heavy atom ordering comes from InChI AuxInfo /N: section (canonical).
    Hydrogen ordering uses 3D-aware methods:
      - 2H prochiral sp3: deuterium + CIP (pro-R before pro-S)
      - 2H prochiral sp2: CIP rank + dihedral (pro-Z before pro-E)
      - 3+ H or equivalent 2H: geometric CCW angle projection

    Args:
        mol: RDKit Mol with explicit H, sanitized, and containing a 3D conformer.
             Typically created via xyz_to_rdkit_mol().

    Returns:
        Full 0-based atom order: [heavy atoms in InChI order, then H in 3D-aware order]
    """
    # Assign stereochemistry for CIP ranks (needed by deuterium and dihedral methods)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Get InChI and AuxInfo via RDKit built-in (no external inchi-1 needed)
    inchi, aux_info = Chem.MolToInchiAndAuxInfo(mol)
    if not inchi or not aux_info:
        raise RuntimeError("Failed to generate InChI/AuxInfo from mol")

    heavy_order = _parse_heavy_order_from_auxinfo(aux_info)
    heavy_set = set(heavy_order)

    # Assign each H to its parent heavy atom via bond connectivity
    h_attached_to: dict[int, int | None] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(neighbors) == 1 and neighbors[0] in heavy_set:
                h_attached_to[atom.GetIdx()] = neighbors[0]
            else:
                h_attached_to[atom.GetIdx()] = None

    # Order H on each heavy atom using 3D-aware methods
    hydrogen_order: list[int] = []
    for heavy_idx in heavy_order:
        hs_on_heavy = [h for h, parent in h_attached_to.items() if parent == heavy_idx]
        if len(hs_on_heavy) > 1:
            ordered = _order_h_on_heavy_atom(mol, heavy_idx, hs_on_heavy)
            hydrogen_order.extend(ordered)
        else:
            hydrogen_order.extend(hs_on_heavy)

    unassigned_hs = [h for h, p in h_attached_to.items() if p is None]
    unassigned_hs.sort()
    hydrogen_order.extend(unassigned_hs)

    full_order = heavy_order + hydrogen_order

    if len(full_order) != mol.GetNumAtoms():
        raise RuntimeError(
            f"Atom count mismatch: mol has {mol.GetNumAtoms()} atoms, "
            f"but generated order list has {len(full_order)}"
        )

    return full_order
