import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def _build_perp_basis(
    z_axis: np.ndarray,
    x_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build orthonormal (x, y) basis on the plane perpendicular to z_axis.

    If x_direction is given, the x-axis is aligned toward its projection
    onto the perpendicular plane. Otherwise, an arbitrary seed vector is used.

    Returns (x_axis, y_axis) forming a right-handed basis with z_axis.
    """
    if x_direction is not None:
        x_vec = x_direction - np.dot(x_direction, z_axis) * z_axis
    else:
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_vec = arbitrary - np.dot(arbitrary, z_axis) * z_axis
    x_axis = x_vec / np.linalg.norm(x_vec)
    y_axis = np.cross(z_axis, x_axis)
    return x_axis, y_axis


def _projected_angle(
    v: np.ndarray,
    z_axis: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> float:
    """atan2 CCW angle of vector v projected onto plane perpendicular to z_axis."""
    v_proj = v - np.dot(v, z_axis) * z_axis
    return float(np.arctan2(np.dot(v_proj, y_axis), np.dot(v_proj, x_axis)))


def _is_ccw(
    mol: Chem.Mol,
    center_idx: int,
    z_axis: np.ndarray,
    atom_indices: tuple[int, ...],
) -> bool:
    """Check if atoms appear CCW when projected onto plane ⟂ z_axis.

    The atoms, in the given order, should trace a CCW path: atan2 angles
    are strictly increasing with exactly one wrap from +π back to -π.
    Returns True when looking along z_axis direction.
    """
    n = len(atom_indices)
    if n < 3:
        raise ValueError(f"_is_ccw requires at least 3 atoms, got {n}")

    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    x_axis, y_axis = _build_perp_basis(z_axis)

    angles = []
    for idx in atom_indices:
        v = np.array(conf.GetAtomPosition(idx)) - center_pos
        angles.append(_projected_angle(v, z_axis, x_axis, y_axis))

    # CCW: exactly one descent when traversing the cycle
    descents = sum(1 for i in range(n) if angles[i] > angles[(i + 1) % n])
    return descents == 1


def _order_h_by_angle_projection(
    center_pos: np.ndarray,
    z_axis: np.ndarray,
    h_pos_list: list[tuple[int, np.ndarray]],
    x_direction: np.ndarray | None = None,
) -> list[int]:
    """Order H atoms by CCW angle projection onto plane ⟂ z_axis.

    Args:
        center_pos: Position of the center heavy atom
        z_axis: Unit vector defining the projection axis
        h_pos_list: List of (h_index, h_position) tuples
        x_direction: Optional vector to align x-axis toward (for determinism
                     with 2 H where atan2 wrap-around could flip order)

    Returns:
        Ordered list of H atom indices (sorted by CCW angle)
    """
    if len(h_pos_list) <= 1:
        return [idx for idx, _ in h_pos_list]

    x_axis, y_axis = _build_perp_basis(z_axis, x_direction=x_direction)

    angles = []
    for h_idx, h_pos in h_pos_list:
        angle = _projected_angle(h_pos - center_pos, z_axis, x_axis, y_axis)
        angles.append((angle, h_idx))

    angles.sort()
    return [h_idx for _, h_idx in angles]


def _infer_lone_pair_position(
    center_pos: np.ndarray,
    bond_vecs: list[np.ndarray]
) -> np.ndarray:
    """Infer lone pair position for a pyramidal center.

    For a tetrahedral center, the sum of all four unit bond vectors ≈ 0.
    Given three explicit bond vectors, the lone pair direction is
    approximately opposite to their sum (each bond equally weighted
    by direction, not length).

    Args:
        center_pos: Position of the center atom
        bond_vecs: Vectors from center to each explicit neighbor

    Returns:
        Inferred lone pair position (at average bond length)
    """
    s = np.zeros(3)
    bond_lens = []
    for v in bond_vecs:
        bond_len = np.linalg.norm(v)
        bond_lens.append(bond_len)
        s = s + v / bond_len  # unit direction, equally weighted
    lp_dir = -s
    lp_dir = lp_dir / np.linalg.norm(lp_dir)
    avg_bond_len = float(np.mean(bond_lens))
    return center_pos + lp_dir * avg_bond_len


def _signed_tetrahedron_volume(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> float:
    """Signed volume of tetrahedron formed by 4 vertex vectors from center.

    Vectors a > b > c > d in CIP priority order. d is conceptually behind
    the plane of a, b, c (lowest priority).

    Returns:
        Negative → a,b,c left-handed → R
        Positive → a,b,c right-handed → S
    """
    a_rel = a - d
    b_rel = b - d
    c_rel = c - d
    return float(np.dot(a_rel, np.cross(b_rel, c_rel)))


def _try_order_2h_signed_volume(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int]:
    """Order 2 H on a prochiral center via signed volume of tetrahedron.

    Used as a fallback when RDKit cannot assign _CIPCode to the center
    (e.g., P, S, As, and other non-carbon stereogenic centers).

    For 4-coordinate centers, uses the four explicit substituents.
    For 3-coordinate pyramidal centers (phosphines, sulfonium ions, etc.),
    infers the lone pair position as the lowest-priority substituent.

    CIP priority: non-H substituents (by _CIPRank) > h1(D) > h2(H)
    [> lone_pair for 3-coordinate].

    Returns [pro-R_idx, pro-S_idx] or sorted if undetermined.
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    center_atom = mol.GetAtomWithIdx(center_idx)

    neighbors: list[Chem.Atom] = list(center_atom.GetNeighbors())
    n_explicit = len(neighbors)

    non_h = [n for n in neighbors if n.GetIdx() not in (h1_idx, h2_idx)]
    non_h.sort(key=lambda n: -_get_cip_rank(n))
    vecs = {n.GetIdx(): np.array(conf.GetAtomPosition(n.GetIdx())) - center_pos
            for n in neighbors}

    if n_explicit == 4:
        non_h_ranks = [_get_cip_rank(n) for n in non_h]
        if len(set(non_h_ranks)) == 1:
            return sorted([h1_idx, h2_idx])  # H equivalent

        vol = _signed_tetrahedron_volume(
            vecs[non_h[0].GetIdx()], vecs[non_h[1].GetIdx()],
            vecs[h1_idx], vecs[h2_idx])
        return [h1_idx, h2_idx] if vol < 0 else [h2_idx, h1_idx]

    elif n_explicit == 3:
        lp_pos = _infer_lone_pair_position(center_pos, list(vecs.values()))

        vol = _signed_tetrahedron_volume(
            vecs[non_h[0].GetIdx()], vecs[h1_idx],
            vecs[h2_idx], lp_pos - center_pos)
        return [h1_idx, h2_idx] if vol < 0 else [h2_idx, h1_idx]

    else:
        return sorted([h1_idx, h2_idx])


def _try_order_2h_sp3(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int]:
    """
    Order 2 H on sp3 center via deuterium substitution + CIP assignment.
    Replaces h1 with D, then checks if center becomes R or S.
    Falls back to signed-volume method for non-carbon centers (P, S, etc.).

    Returns [pro-R_idx, pro-S_idx] or sorted if undetermined.
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

    # RDKit cannot assign CIP (non-carbon centers: P, S, As, etc.)
    # Fall back to manual signed-volume determination
    return _try_order_2h_signed_volume(mol, center_idx, h1_idx, h2_idx)


def _walk_allene_far_end(
    mol: Chem.Mol,
    center_idx: int,
    partner_idx: int
) -> tuple[int, int, int] | None:
    """Walk through sp carbons along the cumulene axis to the far terminal.

    Returns (far_atom_idx, prev_idx, sp_count) or None if the walk fails.
    sp_count is the number of sp carbons traversed:
      - odd  → axial chirality (terminal planes perpendicular, R_a/S_a)
      - even → coplanar terminal planes (pro-Z/pro-E via dihedral)
    """
    cursor_idx = partner_idx
    prev_idx = center_idx
    cursor_atom = mol.GetAtomWithIdx(cursor_idx)
    sp_count = 0

    while cursor_atom.GetHybridization() == Chem.HybridizationType.SP:
        sp_count += 1
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

    return (cursor_idx, prev_idx, sp_count)


def _get_cip_rank(atom: Chem.Atom) -> int:
    """Get _CIPRank from an atom, with a clear error if missing."""
    props = atom.GetPropsAsDict()
    if '_CIPRank' not in props:
        raise RuntimeError(
            f"Atom {atom.GetIdx()} ({atom.GetSymbol()}) missing _CIPRank property. "
            f"Ensure Chem.AssignStereochemistry(mol, cleanIt=True, force=True) "
            f"has been called before H ordering."
        )
    return int(props['_CIPRank'])


def _try_order_2h_allene(
    mol: Chem.Mol,
    center_idx: int,
    partner_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int]:
    """Order 2 H on terminal =CH2 of allene/cumulene.

    Odd sp count (axial chirality, e.g. propadiene):
      Projects H1, H2 and the far-end highest-CIP substituent onto a plane
      perpendicular to the C=C=C axis, then computes signed angles.
      CW arc from a to c → R_a, CCW → S_a.
      Returns [pro-R_a_idx, pro-S_a_idx].

    Even sp count (coplanar, e.g. butatriene):
      Uses the far-end highest-CIP substituent as dihedral reference.
      |dihedral| < 90° → pro-Z.
      Returns [pro-Z_idx, pro-E_idx].

    Returns sorted if H are equivalent.
    """
    far_info = _walk_allene_far_end(mol, center_idx, partner_idx)
    if far_info is None:
        return sorted([h1_idx, h2_idx])
    far_idx, prev_idx, sp_count = far_info

    far_subs = [
        n for n in mol.GetAtomWithIdx(far_idx).GetNeighbors()
        if n.GetIdx() != prev_idx
    ]
    if not far_subs:
        return sorted([h1_idx, h2_idx])

    ranks = [_get_cip_rank(n) for n in far_subs]
    if len(far_subs) > 1 and len(set(ranks)) == 1:
        return sorted([h1_idx, h2_idx])  # Equivalent substituents → H are equivalent

    far_c = max(far_subs, key=lambda n: _get_cip_rank(n))

    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    partner_pos = np.array(conf.GetAtomPosition(partner_idx))
    h1_pos = np.array(conf.GetAtomPosition(h1_idx))
    h2_pos = np.array(conf.GetAtomPosition(h2_idx))
    far_c_pos = np.array(conf.GetAtomPosition(far_c.GetIdx()))

    if sp_count % 2 == 1:
        # Odd: axial chirality (planes perpendicular)
        axis = partner_pos - center_pos
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            return sorted([h1_idx, h2_idx])
        axis = axis / axis_norm

        def _project(vec):
            return vec - np.dot(vec, axis) * axis

        h1_proj = _project(h1_pos - center_pos)
        far_proj = _project(far_c_pos - center_pos)

        # Signed angle from H1 to far-c
        cross = np.dot(axis, np.cross(h1_proj, far_proj))
        angle = np.arctan2(cross, np.dot(h1_proj, far_proj))

        # CW (negative) → R_a, CCW (positive) → S_a
        if angle < 0:
            return [h1_idx, h2_idx]  # h1 is pro-R_a
        else:
            return [h2_idx, h1_idx]  # h2 is pro-R_a
    else:
        # Even: coplanar (pro-Z / pro-E via dihedral)
        dihedral = rdMolTransforms.GetDihedralDeg(
            conf, h1_idx, center_idx, partner_idx, far_c.GetIdx()
        )
        if abs(dihedral) < 90:
            return [h1_idx, h2_idx]  # h1 is pro-Z
        else:
            return [h2_idx, h1_idx]  # h2 is pro-Z


def _try_order_2h_sp2(
    mol: Chem.Mol,
    center_idx: int,
    partner_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int]:
    """Order 2 H on sp2 =CH2 via CIP rank + dihedral angle.

    Uses the highest-CIP-ranked substituent on the partner atom as reference.
    |dihedral| < 90 deg means h1 is cis to ref (pro-Z).

    For allenes (partner is SP), delegates to axial chirality ordering.

    Returns [pro-Z_idx, pro-E_idx] or sorted if H are equivalent.
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
        return sorted([h1_idx, h2_idx])

    ranks = [_get_cip_rank(n) for n in partner_subs]
    if len(partner_subs) > 1 and len(set(ranks)) == 1:
        return sorted([h1_idx, h2_idx])  # All substituents equivalent -> H are truly equivalent

    ref_idx = max(partner_subs, key=lambda n: _get_cip_rank(n)).GetIdx()
    conf = mol.GetConformer()
    dihedral = rdMolTransforms.GetDihedralDeg(conf, h1_idx, center_idx, partner_idx, ref_idx)

    if abs(dihedral) < 90:
        return [h1_idx, h2_idx]  # h1 is pro-Z
    else:
        return [h2_idx, h1_idx]  # h2 is pro-Z


def _get_z_plus_vec(
    mol: Chem.Mol,
    pair_indices: tuple[int, int]
) -> np.ndarray:
    """Determine the z⁺ direction (unit vector) of a trans/axial pair.

    z⁺ direction: z⁻ → z⁺ along the pair axis. z⁺ end is determined by
    CIP rank (higher → z⁺), _CanonicalOrder (larger → z⁺), or min index.
    """
    r0 = _get_cip_rank(mol.GetAtomWithIdx(pair_indices[0]))
    r1 = _get_cip_rank(mol.GetAtomWithIdx(pair_indices[1]))

    if r0 != r1:
        z_plus = pair_indices[0] if r0 > r1 else pair_indices[1]
    else:
        # Same CIP rank: use canonical heavy-atom order
        try:
            pos0 = mol.GetAtomWithIdx(pair_indices[0]).GetIntProp('_CanonicalOrder')
            pos1 = mol.GetAtomWithIdx(pair_indices[1]).GetIntProp('_CanonicalOrder')
            z_plus = pair_indices[0] if pos0 > pos1 else pair_indices[1]
        except KeyError:
            # Both are H (not in heavy_order): min original index
            z_plus = min(pair_indices)

    z_minus = pair_indices[0] if pair_indices[1] == z_plus else pair_indices[1]

    conf = mol.GetConformer()
    pos_plus = np.array(conf.GetAtomPosition(z_plus))
    pos_minus = np.array(conf.GetAtomPosition(z_minus))
    vec = pos_plus - pos_minus
    return vec / np.linalg.norm(vec)


def _order_h_geometric(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    z_axis: np.ndarray | None = None,
) -> list[int]:
    """Order H atoms by geometric CCW angle projection.

    Projects H atoms onto a plane perpendicular to a reference axis,
    then sorts by CCW atan2 angle.

    Args:
        mol: RDKit Mol with explicit H and a 3D conformer
        center_idx: Index of the heavy atom center
        h_indices: Indices of H atoms attached to center (length >= 2)
        z_axis: Optional explicit z-axis unit vector (from trans/axial pair).
                If None, auto-selects a reference direction:
                  - non-H neighbor (R-CH3, Case B)
                  - lone pair (NH3/PH3, 3H + lp)
                  - min-index H placed first (CH4, Case A)

    Returns:
        Deterministically ordered list of H atom indices
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))

    prefix: list[int] = []
    if z_axis is not None:
        h_list = h_indices
    else:
        center_atom = mol.GetAtomWithIdx(center_idx)
        all_nbrs = list(center_atom.GetNeighbors())
        non_h_nbrs = [n for n in all_nbrs if n.GetAtomicNum() != 1]

        if non_h_nbrs:
            ref_vec = np.array(conf.GetAtomPosition(non_h_nbrs[0].GetIdx())) - center_pos
            z_axis = ref_vec / np.linalg.norm(ref_vec)
            h_list = h_indices
        elif len(all_nbrs) == 3:
            bond_vecs = [np.array(conf.GetAtomPosition(n.GetIdx())) - center_pos
                         for n in all_nbrs]
            lp_vec = _infer_lone_pair_position(center_pos, bond_vecs) - center_pos
            z_axis = lp_vec / np.linalg.norm(lp_vec)
            h_list = h_indices
        else:
            ref_h = min(h_indices)
            prefix = [ref_h]
            h_list = [h for h in h_indices if h != ref_h]
            ref_vec = np.array(conf.GetAtomPosition(ref_h)) - center_pos
            z_axis = ref_vec / np.linalg.norm(ref_vec)

    h_pos_list = [(h, np.array(conf.GetAtomPosition(h))) for h in h_list]
    return prefix + _order_h_by_angle_projection(center_pos, z_axis, h_pos_list)


def _order_h_sp2(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H on an sp2 center: Z/E or allene axial chirality."""
    if len(h_indices) == 2:
        h1_idx, h2_idx = h_indices
        center_atom = mol.GetAtomWithIdx(center_idx)
        for bond in center_atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                partner_idx = bond.GetOtherAtomIdx(center_idx)
                return _try_order_2h_sp2(mol, center_idx, partner_idx, h1_idx, h2_idx)
        # No double bond found: sorted for determinism
        return sorted(h_indices)
    return _order_h_geometric(mol, center_idx, h_indices)


def _order_h_sp3(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H on an sp3 center: CIP pro-R/S (deuterium -> signed volume)."""
    if len(h_indices) == 2:
        h1_idx, h2_idx = h_indices
        return _try_order_2h_sp3(mol, center_idx, h1_idx, h2_idx)
    return _order_h_geometric(mol, center_idx, h_indices)


def _order_sp3d_axial_2h(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int,
    axial_indices: list[int],
    eq_indices: list[int]
) -> list[int]:
    """Order 2 axial H on SP3D via equatorial plane chirality.

    The two axial H are on opposite ends of the trigonal-bipyramidal axis.
    Whether they are equivalent depends on the three equatorial substituents:
    if all three have distinct CIP ranks, the equatorial plane is chiral
    and the two axial H are diastereotopic.

    Defines z⁺ as the direction along the axis from which the equatorial
    substituents (sorted by CIP rank descending) appear CCW.
    The axial H at the z⁺ end comes first.

    Returns [first_idx, second_idx] or sorted if H are equivalent.
    """
    conf = mol.GetConformer()

    # Check equivalence: need 3 distinct CIP ranks among eq substituents
    eq_ranks = [_get_cip_rank(mol.GetAtomWithIdx(i)) for i in eq_indices]
    if len(set(eq_ranks)) < len(eq_indices):
        return sorted([h1_idx, h2_idx])  # duplicate ranks → eq plane symmetric → H equivalent

    # Sort eq by CIP rank descending
    eq_sorted = sorted(eq_indices, key=lambda i: -_get_cip_rank(mol.GetAtomWithIdx(i)))
    a_idx, b_idx, c_idx = eq_sorted[0], eq_sorted[1], eq_sorted[2]

    # Axis direction: from h1 toward h2 (arbitrary initial choice)
    h1_pos = np.array(conf.GetAtomPosition(h1_idx))
    h2_pos = np.array(conf.GetAtomPosition(h2_idx))
    axis = h2_pos - h1_pos
    z_axis = axis / np.linalg.norm(axis)

    # a→b→c CCW when looking from h1 → h2 → h1 is at z⁺
    if _is_ccw(mol, center_idx, z_axis, (a_idx, b_idx, c_idx)):
        return [h1_idx, h2_idx]
    else:
        return [h2_idx, h1_idx]


def _order_sp3d_equatorial_2h(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int,
    axial_indices: list[int],
    eq_indices: list[int]
) -> list[int]:
    """Order 2 equatorial H on SP3D via axial CIP direction + atan2.

    The two equatorial H share the equatorial plane with one non-H
    substituent. The axis direction is defined by the two axial
    substituents: z⁺ points from the lower-CIP-rank axial to the
    higher-CIP-rank axial.

    On the plane ⟂ z⁺, the equatorial non-H substituent serves as
    the angle reference (0°). H atoms are ordered by CCW atan2 angle.

    Returns [first_idx, second_idx] or sorted if H are equivalent.
    """
    # Check equivalence: need distinct CIP ranks among axial substituents
    ax_ranks = [_get_cip_rank(mol.GetAtomWithIdx(i)) for i in axial_indices]
    if len(set(ax_ranks)) < 2:
        return sorted([h1_idx, h2_idx])  # equal ranks → no defined z⁺ → H equivalent

    # z_axis: ax_low → ax_high (z⁺ direction)
    ax_sorted = sorted(axial_indices, key=lambda i: -_get_cip_rank(mol.GetAtomWithIdx(i)))
    ax_high, ax_low = ax_sorted[0], ax_sorted[1]

    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))

    ax_high_pos = np.array(conf.GetAtomPosition(ax_high))
    ax_low_pos = np.array(conf.GetAtomPosition(ax_low))
    z_axis = ax_high_pos - ax_low_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    ref_idx = [i for i in eq_indices if i not in (h1_idx, h2_idx)][0]

    ref_vec = np.array(conf.GetAtomPosition(ref_idx)) - center_pos
    h_pos_list = [(h, np.array(conf.GetAtomPosition(h))) for h in (h1_idx, h2_idx)]
    return _order_h_by_angle_projection(
        center_pos, z_axis, h_pos_list, x_direction=ref_vec)


def _classify_sp3d_positions(
    mol: Chem.Mol,
    center_idx: int
) -> tuple[list[int], list[int]]:
    """Classify neighbors of an SP3D center as axial or equatorial.

    For trigonal bipyramidal (5-coordinate):
    - The two neighbors forming the largest angle (~180°) are axial.
    - The remaining three are equatorial.

    Returns (axial_indices, equatorial_indices).
    If classification fails (not 5 neighbors, or no clear axis),
    returns all neighbors as equatorial.
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    center_atom = mol.GetAtomWithIdx(center_idx)
    all_nbrs = list(center_atom.GetNeighbors())

    if len(all_nbrs) != 5:
        return ([], [n.GetIdx() for n in all_nbrs])

    vecs = {}
    for n in all_nbrs:
        vecs[n.GetIdx()] = np.array(conf.GetAtomPosition(n.GetIdx())) - center_pos

    nbr_indices = [n.GetIdx() for n in all_nbrs]

    # Find the pair with the largest angle: these are the axial neighbors
    max_angle = 0.0
    axial_pair = (nbr_indices[0], nbr_indices[1])
    for i in range(len(nbr_indices)):
        for j in range(i + 1, len(nbr_indices)):
            vi = vecs[nbr_indices[i]]
            vj = vecs[nbr_indices[j]]
            cos_angle = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle = np.arccos(cos_angle)
            if angle > max_angle:
                max_angle = angle
                axial_pair = (nbr_indices[i], nbr_indices[j])

    # Require the axial angle to be reasonably close to 180°
    if max_angle < np.radians(140.0):
        return ([], nbr_indices)

    axial_indices = list(axial_pair)
    equatorial_indices = [i for i in nbr_indices if i not in axial_indices]

    return (axial_indices, equatorial_indices)


def _order_h_sp3d(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H on an SP3D (trigonal bipyramidal) center.

    Separates axial from equatorial H. Within each group:
      - ax 2H: equatorial chirality (pro-R/S), equivalent → direct
      - eq 2H: axial CIP + atan2, equivalent → direct
      - eq 3H: axial CIP/_CanonicalOrder z⁺ + CCW geometric
      - eq 1H / ax 1H: direct

    Axial H are ordered before equatorial H in the result.
    """
    if len(h_indices) <= 1:
        return list(h_indices)

    axial_nbrs, eq_nbrs = _classify_sp3d_positions(mol, center_idx)

    # Classification failure: fall back to geometric
    if len(axial_nbrs) != 2:
        return _order_h_geometric(mol, center_idx, h_indices)

    axial_h = [h for h in h_indices if h in axial_nbrs]
    eq_h = [h for h in h_indices if h in eq_nbrs]

    result: list[int] = []

    # Order axial H group
    if len(axial_h) == 2:
        result.extend(_order_sp3d_axial_2h(
            mol, center_idx, axial_h[0], axial_h[1], axial_nbrs, eq_nbrs
        ))
    elif len(axial_h) == 1:
        result.extend(axial_h)

    # Order equatorial H group
    if len(eq_h) == 2:
        result.extend(_order_sp3d_equatorial_2h(
            mol, center_idx, eq_h[0], eq_h[1], axial_nbrs, eq_nbrs
        ))
    elif len(eq_h) == 3:
        # eq 3H: z⁺ from axial pair, CCW geometric
        z_axis = _get_z_plus_vec(mol, (axial_nbrs[0], axial_nbrs[1]))
        result.extend(_order_h_geometric(mol, center_idx, eq_h, z_axis=z_axis))
    elif len(eq_h) == 1:
        result.extend(eq_h)

    return result


def _find_sp3d2_trans_pairs(
    mol: Chem.Mol,
    center_idx: int
) -> list[tuple[int, int]]:
    """Find the three trans (180°) pairs in an octahedral center.

    Returns up to 3 (idx1, idx2) tuples, sorted by angle descending.
    Only pairs with angle > 150° are included.
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    center_atom = mol.GetAtomWithIdx(center_idx)
    all_nbrs = list(center_atom.GetNeighbors())

    vecs = {}
    for n in all_nbrs:
        vecs[n.GetIdx()] = np.array(conf.GetAtomPosition(n.GetIdx())) - center_pos

    pairs_with_angles = []
    nbr_indices = [n.GetIdx() for n in all_nbrs]
    for i in range(len(nbr_indices)):
        for j in range(i + 1, len(nbr_indices)):
            vi = vecs[nbr_indices[i]]
            vj = vecs[nbr_indices[j]]
            cos_angle = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle = np.arccos(cos_angle)
            if angle > np.radians(150.0):
                pairs_with_angles.append((angle, nbr_indices[i], nbr_indices[j]))

    pairs_with_angles.sort(key=lambda x: -x[0])

    used: set[int] = set()
    trans_pairs: list[tuple[int, int]] = []
    for _angle, i, j in pairs_with_angles:
        if i not in used and j not in used:
            trans_pairs.append((i, j))
            used.add(i)
            used.add(j)

    return trans_pairs


def _analyze_square_chirality(
    mol: Chem.Mol,
    center_idx: int,
    z_from: int,
    z_to: int,
    square_indices: list[int]
) -> bool | None:
    """Analyze chirality of a 4-point square on the plane ⟂ the z axis.

    z_from → z_to defines the axis direction. The four atoms in
    square_indices lie approximately in the perpendicular plane.

    Steps:
      1. Reject if the 4 CIP ranks form 2 groups of 2 (e.g., A₂B₂).
      2. Identify diagonal pairs (trans within the square).
         Reject if any diagonal has matching CIP ranks.
      3. Remove one atom (prefer duplicate rank, then lowest rank)
         to obtain 3 atoms with distinct CIP ranks.
      4. Triangle CW/CCW analysis on the 3 remaining atoms.

    Returns True if CCW when looking along z_from→z_to, False if CW,
    None if the square is symmetric (H equivalent).
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))

    # Collect CIP ranks
    sq_ranks = {idx: _get_cip_rank(mol.GetAtomWithIdx(idx)) for idx in square_indices}

    # --- Step 1: check for 2 groups of duplicates ---
    from collections import Counter
    rank_values = list(sq_ranks.values())
    rank_counts = Counter(rank_values)
    pairs_of_two = sum(1 for c in rank_counts.values() if c >= 2)
    if pairs_of_two >= 2:
        return None

    # --- Step 2: check diagonals ---
    # Find trans pairs among the 4 square atoms (these are the diagonals)
    sq_vecs = {}
    for idx in square_indices:
        sq_vecs[idx] = np.array(conf.GetAtomPosition(idx)) - center_pos

    diagonals = []
    sq_list = list(square_indices)
    for i in range(len(sq_list)):
        for j in range(i + 1, len(sq_list)):
            vi = sq_vecs[sq_list[i]]
            vj = sq_vecs[sq_list[j]]
            cos_angle = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle = np.arccos(cos_angle)
            if angle > np.radians(150.0):
                diagonals.append((sq_list[i], sq_list[j]))

    for d1, d2 in diagonals:
        if sq_ranks[d1] == sq_ranks[d2]:
            return None

    # --- Step 3: remove one atom to get 3 distinct ranks ---
    sorted_sq = sorted(square_indices, key=lambda i: -sq_ranks[i])

    kept = []
    seen_ranks: set[int] = set()
    for idx in sorted_sq:
        r = sq_ranks[idx]
        if r not in seen_ranks and len(kept) < 3:
            kept.append(idx)
            seen_ranks.add(r)
        elif len(kept) >= 3:
            break

    if len(kept) < 3:
        return None

    # --- Step 4: triangle CW/CCW analysis ---
    # Axis direction: z_from → z_to
    h_from_pos = np.array(conf.GetAtomPosition(z_from))
    h_to_pos = np.array(conf.GetAtomPosition(z_to))
    z_axis = h_to_pos - h_from_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    # kept[0], kept[1], kept[2] are in CIP descending order
    return _is_ccw(mol, center_idx, z_axis, (kept[0], kept[1], kept[2]))


def _order_sp3d2_trans_2h(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int,
    trans_pairs: list[tuple[int, int]]
) -> list[int]:
    """Order 2 trans H on an octahedral center via square chirality.

    The 4 cis substituents form a square. Analyzes its chirality:
    CW when looking from h1→h2 means h1 is at z⁺.
    Returns [z⁺_idx, z⁻_idx] or sorted if equivalent.
    """
    # Find the 4 cis substituents (all neighbors except h1 and h2)
    center_atom = mol.GetAtomWithIdx(center_idx)
    all_nbrs = {n.GetIdx() for n in center_atom.GetNeighbors()}
    h_set = {h1_idx, h2_idx}
    square_indices = list(all_nbrs - h_set)

    if len(square_indices) != 4:
        return sorted([h1_idx, h2_idx])

    ccw = _analyze_square_chirality(
        mol, center_idx, h1_idx, h2_idx, square_indices
    )

    if ccw is None:
        return sorted([h1_idx, h2_idx])
    elif ccw:
        return [h1_idx, h2_idx]  # CCW: h1 at z⁺
    else:
        return [h2_idx, h1_idx]  # CW: h2 at z⁺


def _order_sp3d2_cis_2h(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int,
    trans_pairs: list[tuple[int, int]]
) -> list[int]:
    """Order 2 cis H on an octahedral center.

    Each H belongs to a different trans pair. Order by the CIP rank
    of their trans partners. If the trans partners have equal CIP rank,
    analyze each H's cis square chirality.
    """
    # Find trans partner for each H
    t1 = None
    t2 = None
    for a, b in trans_pairs:
        if a == h1_idx:
            t1 = b
        elif b == h1_idx:
            t1 = a
        if a == h2_idx:
            t2 = b
        elif b == h2_idx:
            t2 = a

    if t1 is None or t2 is None:
        return sorted([h1_idx, h2_idx])

    r1 = _get_cip_rank(mol.GetAtomWithIdx(t1))
    r2 = _get_cip_rank(mol.GetAtomWithIdx(t2))

    if r1 != r2:
        # Order by trans partner CIP rank (higher rank H first)
        if r1 > r2:
            return [h1_idx, h2_idx]
        else:
            return [h2_idx, h1_idx]

    # T_a = T_b: use cis-square chirality
    center_atom = mol.GetAtomWithIdx(center_idx)
    all_nbrs = {n.GetIdx() for n in center_atom.GetNeighbors()}

    # H_a's cis square: all neighbors except H_a and T_a
    cis1 = list(all_nbrs - {h1_idx, t1})
    # H_b's cis square: all neighbors except H_b and T_b
    cis2 = list(all_nbrs - {h2_idx, t2})

    ccw1 = _analyze_square_chirality(mol, center_idx, t1, h1_idx, cis1)
    ccw2 = _analyze_square_chirality(mol, center_idx, t2, h2_idx, cis2)

    # The H whose cis square is CCW from T→H comes first
    if ccw1 is not None and ccw2 is not None:
        if ccw1 != ccw2:
            if ccw1:  # h1's square CCW → h1 first
                return [h1_idx, h2_idx]
            else:
                return [h2_idx, h1_idx]

    return sorted([h1_idx, h2_idx])


def _order_h_sp3d2(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H on an SP3D2 (octahedral) center.

    Dispatches by H count and trans relationship:
      - 2H trans: square chirality (CW/CCW on 4 cis substituents)
      - 2H cis:   trans partner CIP comparison → cis-square chirality
      - 3H:       classify as fac or mer, delegate to trans/cis sub-problems
      - 4H-5H:    classify non-H positions, delegate to sub-problems
      - 6H:       pick a trans pair as reference, order ax then eq

    Equivalent 2H are returned directly (no geometric needed).
    Geometric calls use explicit trans-pair z⁺ via _get_z_plus_vec when
    CIP ranks are equal.
    """
    n_h = len(h_indices)
    if n_h <= 1:
        return list(h_indices)

    trans_pairs = _find_sp3d2_trans_pairs(mol, center_idx)
    if len(trans_pairs) != 3:
        return _order_h_geometric(mol, center_idx, h_indices)

    # --- 2H ---
    if n_h == 2:
        h1_idx, h2_idx = h_indices[0], h_indices[1]
        is_trans = any(
            (h1_idx in pair and h2_idx in pair) for pair in trans_pairs
        )
        if is_trans:
            return _order_sp3d2_trans_2h(
                mol, center_idx, h1_idx, h2_idx, trans_pairs
            )
        else:
            return _order_sp3d2_cis_2h(
                mol, center_idx, h1_idx, h2_idx, trans_pairs
            )

    # --- 3H-6H: classify trans pairs ---
    h_set = set(h_indices)

    hh_pairs: list[tuple[int, int]] = []
    hx_pairs: list[tuple[int, int]] = []  # (H, non-H)
    xx_pairs: list[tuple[int, int]] = []
    for a, b in trans_pairs:
        a_is_h = a in h_set
        b_is_h = b in h_set
        if a_is_h and b_is_h:
            hh_pairs.append((a, b))
        elif a_is_h:
            hx_pairs.append((a, b))
        elif b_is_h:
            hx_pairs.append((b, a))
        else:
            xx_pairs.append((a, b))

    result: list[int] = []

    # --- 3H ---
    if n_h == 3:
        if len(hx_pairs) == 3:
            # fac: 3 H-X pairs
            ranks: dict[int, list[int]] = {}
            for _h, x in hx_pairs:
                r = _get_cip_rank(mol.GetAtomWithIdx(x))
                ranks.setdefault(r, []).append(_h)

            if len(ranks) == 3:
                # ABC: all different → order by trans CIP descending
                for r in sorted(ranks, reverse=True):
                    result.append(ranks[r][0])
            elif len(ranks) == 2:
                # AAB: unique H (trans=B) + 2H (trans=A) → 2H cis
                unique_rank = [r for r, hs in ranks.items()
                               if len(hs) == 1][0]
                dup_rank = [r for r, hs in ranks.items()
                            if len(hs) == 2][0]
                result.append(ranks[unique_rank][0])
                result.extend(_order_sp3d2_cis_2h(
                    mol, center_idx,
                    ranks[dup_rank][0], ranks[dup_rank][1], trans_pairs
                ))
            else:
                # AAA: all equivalent → min-idx H first, deuterate, cis-2H
                h_sorted = sorted(h_indices)
                first_h = h_sorted[0]
                h2, h3 = h_sorted[1], h_sorted[2]
                result.append(first_h)

                mol_tmp = Chem.RWMol(Chem.Mol(mol))
                Chem.AssignAtomChiralTagsFromStructure(mol_tmp)
                mol_tmp.GetAtomWithIdx(first_h).SetIsotope(2)
                Chem.AssignStereochemistry(mol_tmp, cleanIt=True, force=True)

                result.extend(_order_sp3d2_cis_2h(
                    mol_tmp, center_idx, h2, h3, trans_pairs
                ))
        else:
            # mer: 1 H-H + 1 H-X
            if hx_pairs:
                result.append(hx_pairs[0][0])
            for h_a, h_b in hh_pairs:
                result.extend(_order_sp3d2_trans_2h(
                    mol, center_idx, h_a, h_b, trans_pairs
                ))

    # --- 4H ---
    elif n_h == 4:
        if len(hx_pairs) == 0:
            # non-H trans: 2 H-H pairs + 1 non-H trans pair
            # Use trans non-H pair for z⁺ direction
            z_axis = _get_z_plus_vec(mol, xx_pairs[0])
            result = _order_h_geometric(mol, center_idx, h_indices, z_axis=z_axis)
        else:
            # non-H cis: 2 H-X + 1 H-H
            def _trans_rank(hx):
                return _get_cip_rank(mol.GetAtomWithIdx(hx[1]))

            ranks_hx = [_trans_rank(p) for p in hx_pairs]
            if len(set(ranks_hx)) >= 2:
                hx_pairs.sort(key=_trans_rank, reverse=True)
                for h, _x in hx_pairs:
                    result.append(h)
            else:
                # Equal rank → 2H equivalent: sorted for determinism
                result.extend(sorted(h for h, _ in hx_pairs))
            # H-H pair
            for h_a, h_b in hh_pairs:
                result.extend(_order_sp3d2_trans_2h(
                    mol, center_idx, h_a, h_b, trans_pairs
                ))

    # --- 5H ---
    elif n_h == 5:
        # 1 H-X + 2 H-H. H-X H first (unique trans partner).
        if hx_pairs:
            h_first, x = hx_pairs[0]
            result.append(h_first)
            eq_h = [h for h in h_indices if h != h_first]
            # Use trans H-X pair for z⁺ direction
            z_axis = _get_z_plus_vec(mol, (h_first, x))
            result.extend(_order_h_geometric(mol, center_idx, eq_h, z_axis=z_axis))

    # --- 6H ---
    else:
        # 3 H-H pairs. Pick one trans pair as ax (min-idx).
        sorted_hh = sorted(hh_pairs, key=lambda p: min(p))
        ax_a, ax_b = sorted_hh[0]
        # ax 2H equivalent: min idx first
        result.extend([min(ax_a, ax_b), max(ax_a, ax_b)])
        eq_h = [h for h in h_indices if h not in result]
        # Use ax trans pair for z⁺ direction
        z_axis = _get_z_plus_vec(mol, (ax_a, ax_b))
        result.extend(_order_h_geometric(mol, center_idx, eq_h, z_axis=z_axis))

    return result


def _order_h_on_heavy_atom(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H atoms on a heavy atom using 3D-aware methods.

    Dispatches by RDKit hybridization:
      - SP2: 2H via Z/E or allene axial chirality; equivalent → direct
      - SP3: 2H via CIP pro-R/S (deuterium → signed volume); equivalent → direct
      - SP3D: axial/equatorial separation; eq 3H via axial z⁺ + CCW
      - SP3D2: trans/cis analysis with deuteration for fac AAA
      - Other: geometric CCW fallback

    1 H is returned directly; 2H-equivalent cases skip geometric as they
    produce no prochiral pair after a single substitution.

    Requires mol to have a conformer with 3D coordinates and both
    AssignAtomChiralTagsFromStructure and AssignStereochemistry already
    called.

    Args:
        mol: RDKit Mol with explicit H and a 3D conformer
        center_idx: Index of the heavy atom center
        h_indices: Indices of H atoms attached to center

    Returns:
        Deterministically ordered list of H atom indices
    """
    if len(h_indices) <= 1:
        return list(h_indices)

    hyb = mol.GetAtomWithIdx(center_idx).GetHybridization()

    if hyb == Chem.HybridizationType.SP2:
        return _order_h_sp2(mol, center_idx, h_indices)
    elif hyb == Chem.HybridizationType.SP3:
        return _order_h_sp3(mol, center_idx, h_indices)
    elif hyb == Chem.HybridizationType.SP3D:
        return _order_h_sp3d(mol, center_idx, h_indices)
    elif hyb == Chem.HybridizationType.SP3D2:
        return _order_h_sp3d2(mol, center_idx, h_indices)
    else:
        return _order_h_geometric(mol, center_idx, h_indices)
