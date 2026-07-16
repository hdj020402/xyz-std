import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def _order_h_by_angle_projection(
    center_pos: np.ndarray,
    ref_pos: np.ndarray,
    h_pos_list: list[tuple[int, np.ndarray]]
) -> list[int]:
    """
    Order H atoms by CCW angle projection onto plane perpendicular to center->ref axis.

    Creates an orthonormal basis on the perpendicular plane via Gram-Schmidt,
    then sorts H atoms by their atan2 angle in that plane.

    Args:
        center_pos: Position of the center heavy atom
        ref_pos: Position of a reference non-H neighbor (defines projection axis)
        h_pos_list: List of (h_index, h_position) tuples

    Returns:
        Ordered list of H atom indices (sorted by CCW angle)
    """
    if len(h_pos_list) <= 1:
        return [idx for idx, _ in h_pos_list]

    z_axis = ref_pos - center_pos
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-10:
        return [idx for idx, _ in h_pos_list]
    z_axis = z_axis / z_norm

    # Gram-Schmidt: build orthonormal x/y axes on the perpendicular plane
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = arbitrary - np.dot(arbitrary, z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-10:
        return [idx for idx, _ in h_pos_list]
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)

    angles = []
    for h_idx, h_pos in h_pos_list:
        vec = h_pos - center_pos
        vec_proj = vec - np.dot(vec, z_axis) * z_axis
        angle = np.arctan2(np.dot(vec_proj, y_axis), np.dot(vec_proj, x_axis))
        angles.append((angle, h_idx))

    angles.sort()
    return [h_idx for _, h_idx in angles]


def _try_order_2h_sp3(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int] | None:
    """
    Order 2 H on sp3 center via deuterium substitution + CIP assignment.
    Replaces h1 with D, then checks if center becomes R or S.
    Returns [pro-R_idx, pro-S_idx] or None if CIP cannot determine.
    """
    mol_tmp = Chem.RWMol(Chem.Mol(mol))
    Chem.AssignAtomChiralTagsFromStructure(mol_tmp)
    mol_tmp.GetAtomWithIdx(h1_idx).SetIsotope(2)
    Chem.AssignStereochemistry(mol_tmp, cleanIt=True, force=True)
    cip = mol_tmp.GetAtomWithIdx(center_idx).GetPropsAsDict().get('_CIPCode')

    if cip == 'R':
        return [h1_idx, h2_idx]  # h1 is pro-R
    elif cip == 'S':
        return [h2_idx, h1_idx]  # h2 is pro-R
    return None


def _walk_allene_far_end(
    mol: Chem.Mol,
    center_idx: int,
    partner_idx: int
) -> tuple[int, int] | None:
    """Walk through sp carbons along the cumulene axis to the far terminal.

    Returns (far_atom_idx, prev_idx) or None if the walk fails.
    """
    cursor_idx = partner_idx
    prev_idx = center_idx
    cursor_atom = mol.GetAtomWithIdx(cursor_idx)

    while cursor_atom.GetHybridization() == Chem.HybridizationType.SP:
        next_atoms = []
        for bond in cursor_atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                other = bond.GetOtherAtomIdx(cursor_idx)
                if other != prev_idx:
                    next_atoms.append(other)
        if len(next_atoms) != 1:
            return None
        prev_idx = cursor_idx
        cursor_idx = next_atoms[0]
        cursor_atom = mol.GetAtomWithIdx(cursor_idx)

    return (cursor_idx, prev_idx)


def _try_order_2h_allene(
    mol: Chem.Mol,
    center_idx: int,
    partner_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int] | None:
    """Order 2 H on terminal =CH2 of allene/cumulene via axial chirality.

    Projects H1, H2 and the far-end highest-CIP substituent onto a plane
    perpendicular to the C=C=C axis, then computes signed angles to determine
    pro-R_a / pro-S_a ordering (CW arc from a to c → R_a).

    Returns [pro-R_a_idx, pro-S_a_idx] or None if H are equivalent.
    """
    far_info = _walk_allene_far_end(mol, center_idx, partner_idx)
    if far_info is None:
        return None
    far_idx, prev_idx = far_info

    far_subs = [
        n for n in mol.GetAtomWithIdx(far_idx).GetNeighbors()
        if n.GetIdx() != prev_idx
    ]
    if not far_subs:
        return None

    ranks = [n.GetPropsAsDict()['_CIPRank'] for n in far_subs]
    if len(far_subs) > 1 and len(set(ranks)) == 1:
        return None  # Equivalent substituents → H are equivalent

    far_c = max(far_subs, key=lambda n: n.GetPropsAsDict()['_CIPRank'])

    # Geometric projection onto plane perpendicular to C=C=C axis
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    partner_pos = np.array(conf.GetAtomPosition(partner_idx))
    h1_pos = np.array(conf.GetAtomPosition(h1_idx))
    h2_pos = np.array(conf.GetAtomPosition(h2_idx))
    far_c_pos = np.array(conf.GetAtomPosition(far_c.GetIdx()))

    axis = partner_pos - center_pos
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return None
    axis = axis / axis_norm

    def _project(vec):
        return vec - np.dot(vec, axis) * axis

    h1_proj = _project(h1_pos - center_pos)
    h2_proj = _project(h2_pos - center_pos)
    far_proj = _project(far_c_pos - center_pos)

    # Signed angle from H1 to far-c; axis as normal for CW/CCW determination.
    # CW (negative) → H1 as 'a' yields R_a; CCW (positive) → H1 as 'a' yields S_a.
    cross1 = np.dot(axis, np.cross(h1_proj, far_proj))
    angle1 = np.arctan2(cross1, np.dot(h1_proj, far_proj))

    if angle1 < 0:
        return [h1_idx, h2_idx]  # h1 is pro-R_a
    else:
        return [h2_idx, h1_idx]  # h2 is pro-R_a


def _try_order_2h_sp2(
    mol: Chem.Mol,
    center_idx: int,
    partner_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int] | None:
    """Order 2 H on sp2 =CH2 via CIP rank + dihedral angle.

    Uses the highest-CIP-ranked substituent on the partner atom as reference.
    |dihedral| < 90 deg means h1 is cis to ref (pro-Z).

    For allenes (partner is SP), delegates to axial chirality ordering.

    Returns [pro-Z_idx, pro-E_idx] or None if H are equivalent.
    """
    partner_atom = mol.GetAtomWithIdx(partner_idx)

    # Allene/cumulene: partner is sp → axial chirality ordering
    if partner_atom.GetHybridization() == Chem.HybridizationType.SP:
        return _try_order_2h_allene(mol, center_idx, partner_idx, h1_idx, h2_idx)

    # Normal sp2: partner's directly-bonded substituents
    partner_subs = [
        n for n in partner_atom.GetNeighbors()
        if n.GetIdx() != center_idx
    ]
    if not partner_subs:
        return None

    ranks = [n.GetPropsAsDict()['_CIPRank'] for n in partner_subs]
    if len(partner_subs) > 1 and len(set(ranks)) == 1:
        return None  # All substituents equivalent -> H are truly equivalent

    ref_idx = max(partner_subs, key=lambda n: n.GetPropsAsDict()['_CIPRank']).GetIdx()
    conf = mol.GetConformer()
    dihedral = rdMolTransforms.GetDihedralDeg(conf, h1_idx, center_idx, partner_idx, ref_idx)

    if abs(dihedral) < 90:
        return [h1_idx, h2_idx]  # h1 is pro-Z
    else:
        return [h2_idx, h1_idx]  # h2 is pro-Z


def _try_order_2h(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int] | None:
    """Dispatch: try sp2 (=CH2) first, then sp3 (prochiral center)."""
    center_atom = mol.GetAtomWithIdx(center_idx)

    # Check for double bond -> sp2 =CH2
    for bond in center_atom.GetBonds():
        if bond.GetBondTypeAsDouble() == 2.0:
            partner_idx = bond.GetOtherAtomIdx(center_idx)
            result = _try_order_2h_sp2(mol, center_idx, partner_idx, h1_idx, h2_idx)
            if result is not None:
                return result

    # Try sp3 prochiral
    return _try_order_2h_sp3(mol, center_idx, h1_idx, h2_idx)


def _order_h_on_heavy_atom(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """
    Order H atoms on a heavy atom using 3D-aware methods.

    Strategy:
      - 0-1 H: return as-is
      - 2 H: CIP-based (pro-R/pro-S for sp3, pro-Z/pro-E for sp2),
              fallback to geometric if CIP undetermined
      - 3+ H: geometric CCW angle projection

    Requires mol to have a conformer with 3D coordinates and both
    AssignAtomChiralTagsFromStructure and AssignStereochemistry already
    called (for CIP ranks in sp2 case).

    Args:
        mol: RDKit Mol with explicit H and a 3D conformer
        center_idx: Index of the heavy atom center
        h_indices: Indices of H atoms attached to center

    Returns:
        Deterministically ordered list of H atom indices
    """
    if len(h_indices) <= 1:
        return list(h_indices)

    if len(h_indices) == 2:
        result = _try_order_2h(mol, center_idx, h_indices[0], h_indices[1])
        if result is not None:
            return result

    # Geometric fallback (3+ H or CIP could not determine)
    conf = mol.GetConformer()
    center_atom = mol.GetAtomWithIdx(center_idx)
    non_h_neighbors = [n for n in center_atom.GetNeighbors() if n.GetAtomicNum() != 1]

    center_pos = np.array(conf.GetAtomPosition(center_idx))

    if not non_h_neighbors:
        # No non-H reference (e.g., CH4): pick one H as the reference axis,
        # place it first, then order the rest by CCW projection around center->ref_H.
        # This guarantees consistent handedness even without a heavy-atom reference.
        ref_h = min(h_indices)
        remaining = [h for h in h_indices if h != ref_h]
        if not remaining:
            return [ref_h]
        ref_pos = np.array(conf.GetAtomPosition(ref_h))
        h_pos_list = [(h, np.array(conf.GetAtomPosition(h))) for h in remaining]
        return [ref_h] + _order_h_by_angle_projection(center_pos, ref_pos, h_pos_list)

    # Pick reference: highest atomic number, then lowest index for stability
    ref_neighbor = max(non_h_neighbors, key=lambda n: (n.GetAtomicNum(), -n.GetIdx()))
    ref_pos = np.array(conf.GetAtomPosition(ref_neighbor.GetIdx()))
    h_pos_list = [(h, np.array(conf.GetAtomPosition(h))) for h in h_indices]

    return _order_h_by_angle_projection(center_pos, ref_pos, h_pos_list)
