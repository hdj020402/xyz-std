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


def _infer_lone_pair_position(
    center_pos: np.ndarray,
    bond_vecs: list[np.ndarray]
) -> np.ndarray | None:
    """Infer lone pair position for a pyramidal center.

    For a tetrahedral center, the sum of all four bond vectors ≈ 0.
    Given three explicit bond vectors, the lone pair direction is
    approximately opposite to their sum.

    Args:
        center_pos: Position of the center atom
        bond_vecs: List of vectors from center to each explicit neighbor

    Returns:
        Inferred lone pair position (at average bond length) or None if degenerate
    """
    s = np.zeros(3)
    for v in bond_vecs:
        s = s + v
    lp_dir = -s
    norm = np.linalg.norm(lp_dir)
    if norm < 1e-10:
        return None
    avg_bond_len = np.mean([np.linalg.norm(v) for v in bond_vecs])
    return center_pos + lp_dir / norm * avg_bond_len


def _try_order_2h_signed_volume(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int] | None:
    """Order 2 H on a prochiral center via signed volume of tetrahedron.

    Used as a fallback when RDKit cannot assign _CIPCode to the center
    (e.g., P, S, As, and other non-carbon stereogenic centers).

    For 4-coordinate centers, uses the four explicit substituents.
    For 3-coordinate pyramidal centers (phosphines, sulfonium ions, etc.),
    infers the lone pair position as the lowest-priority substituent.

    CIP priority: non-H substituents (by _CIPRank) > D(h1) > H(h2)
    [> lone_pair for 3-coordinate].

    The signed volume sign determines R vs S for the deuterated center,
    which maps directly to pro-R / pro-S for the original center.

    Returns [pro-R_idx, pro-S_idx] or None if undetermined.
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    center_atom = mol.GetAtomWithIdx(center_idx)

    neighbors = list(center_atom.GetNeighbors())
    h_set = {h1_idx, h2_idx}
    non_h = [n for n in neighbors if n.GetIdx() not in h_set]
    n_explicit = len(neighbors)

    if n_explicit not in (3, 4):
        return None

    # Check equivalence: if non-H substituents have the same CIP rank,
    # the two H are truly equivalent → return None
    if len(non_h) >= 2:
        non_h_ranks = [
            n.GetPropsAsDict().get('_CIPRank') for n in non_h
        ]
        if None not in non_h_ranks and len(set(non_h_ranks)) == 1:
            return None

    # Collect bond vectors from center to each neighbor
    vecs: dict[int, np.ndarray] = {}
    for n in neighbors:
        vecs[n.GetIdx()] = np.array(conf.GetAtomPosition(n.GetIdx())) - center_pos

    # Build priority order: non-H (by _CIPRank desc) > h1(D) > h2(H)
    non_h_sorted = sorted(
        non_h,
        key=lambda n: -int(n.GetPropsAsDict().get('_CIPRank', 0))
    )
    priority_atoms = non_h_sorted + [
        mol.GetAtomWithIdx(h1_idx),
        mol.GetAtomWithIdx(h2_idx),
    ]

    # Set up four vectors a>b>c>d in CIP priority order
    if n_explicit == 4:
        if len(non_h_sorted) != 2:
            return None
        a = vecs[priority_atoms[0].GetIdx()]
        b = vecs[priority_atoms[1].GetIdx()]
        c = vecs[priority_atoms[2].GetIdx()]  # h1 (D)
        d = vecs[priority_atoms[3].GetIdx()]  # h2 (H)
    else:  # n_explicit == 3
        if len(non_h_sorted) != 1:
            return None
        a = vecs[priority_atoms[0].GetIdx()]  # non-H
        b = vecs[priority_atoms[1].GetIdx()]  # h1 (D)
        c = vecs[priority_atoms[2].GetIdx()]  # h2 (H)
        lp_pos = _infer_lone_pair_position(
            center_pos, [vecs[n.GetIdx()] for n in neighbors]
        )
        if lp_pos is None:
            return None
        d = lp_pos - center_pos  # lone pair (phantom, lowest priority)

    # Signed volume: (a-d)·((b-d)×(c-d))
    # Place lowest-priority substituent (d) conceptually behind the plane.
    # Positive → a,b,c right-handed → CCW → S
    # Negative → a,b,c left-handed  → CW  → R
    a_rel = a - d
    b_rel = b - d
    c_rel = c - d
    signed_vol = np.dot(a_rel, np.cross(b_rel, c_rel))

    if abs(signed_vol) < 1e-10:
        return None  # planar, cannot determine handedness

    if signed_vol < 0:
        return [h1_idx, h2_idx]  # R → h1 is pro-R
    else:
        return [h2_idx, h1_idx]  # S → h2 is pro-R


def _try_order_2h_sp3(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int
) -> list[int] | None:
    """
    Order 2 H on sp3 center via deuterium substitution + CIP assignment.
    Replaces h1 with D, then checks if center becomes R or S.
    Falls back to signed-volume method for non-carbon centers (P, S, etc.).

    Returns [pro-R_idx, pro-S_idx] or None if undetermined.
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
) -> list[int] | None:
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

    Returns None if H are equivalent.
    """
    far_info = _walk_allene_far_end(mol, center_idx, partner_idx)
    if far_info is None:
        return None
    far_idx, prev_idx, sp_count = far_info

    far_subs = [
        n for n in mol.GetAtomWithIdx(far_idx).GetNeighbors()
        if n.GetIdx() != prev_idx
    ]
    if not far_subs:
        return None

    ranks = [_get_cip_rank(n) for n in far_subs]
    if len(far_subs) > 1 and len(set(ranks)) == 1:
        return None  # Equivalent substituents → H are equivalent

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
            return None
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


def _get_z_plus_ref(
    mol: Chem.Mol,
    pair_indices: tuple[int, int]
) -> int:
    """Determine the atom at the z⁺ end of a trans/axial pair.

    z⁺ direction: low → high CIP rank. If CIP ranks are equal, falls back
    to _CanonicalOrder (InChI canonical position). If both atoms are H
    (not in heavy_order), uses min original index.
    """
    p0 = mol.GetAtomWithIdx(pair_indices[0]).GetPropsAsDict()
    p1 = mol.GetAtomWithIdx(pair_indices[1]).GetPropsAsDict()
    r0 = int(p0.get('_CIPRank', 0))
    r1 = int(p1.get('_CIPRank', 0))

    if r0 != r1:
        return pair_indices[0] if r0 > r1 else pair_indices[1]

    # Same CIP rank: use canonical heavy-atom order
    try:
        pos0 = int(p0['_CanonicalOrder'])
        pos1 = int(p1['_CanonicalOrder'])
        return pair_indices[0] if pos0 > pos1 else pair_indices[1]
    except KeyError:
        # Both are H (not in heavy_order): min original index
        return min(pair_indices)


def _order_h_geometric(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    ref_idx: int | None = None,
) -> list[int]:
    """Order H atoms by geometric CCW angle projection.

    Projects H atoms onto a plane perpendicular to a reference axis,
    then sorts by CCW atan2 angle.

    Args:
        mol: RDKit Mol with explicit H and a 3D conformer
        center_idx: Index of the heavy atom center
        h_indices: Indices of H atoms attached to center (length >= 2)
        ref_idx: Optional explicit reference atom index for the z-axis.
                 If None, auto-selects: non-H neighbor (sp3 Case B)
                 or min-index H placed first (sp3 Case A, e.g. CH4).

    Returns:
        Deterministically ordered list of H atom indices
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))

    if ref_idx is not None:
        # Explicit reference: project all H equally around the given axis
        ref_pos = np.array(conf.GetAtomPosition(ref_idx))
        h_pos_list = [(h, np.array(conf.GetAtomPosition(h))) for h in h_indices]
        return _order_h_by_angle_projection(center_pos, ref_pos, h_pos_list)

    center_atom = mol.GetAtomWithIdx(center_idx)
    non_h_neighbors = [n for n in center_atom.GetNeighbors() if n.GetAtomicNum() != 1]

    if non_h_neighbors:
        # Case B (sp3, e.g. R-CH3): single non-H neighbor as ref,
        # all H projected equally.
        ref_pos = np.array(conf.GetAtomPosition(non_h_neighbors[0].GetIdx()))
        h_pos_list = [(h, np.array(conf.GetAtomPosition(h))) for h in h_indices]
        return _order_h_by_angle_projection(center_pos, ref_pos, h_pos_list)

    # Case A (all H, e.g. CH4): pick min-index H as reference axis,
    # place it first, then CCW-order the rest.
    ref_h = min(h_indices)
    remaining = [h for h in h_indices if h != ref_h]
    if not remaining:
        return [ref_h]
    ref_pos = np.array(conf.GetAtomPosition(ref_h))
    h_pos_list = [(h, np.array(conf.GetAtomPosition(h))) for h in remaining]
    return [ref_h] + _order_h_by_angle_projection(center_pos, ref_pos, h_pos_list)


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
                result = _try_order_2h_sp2(mol, center_idx, partner_idx, h1_idx, h2_idx)
                if result is not None:
                    return result
        # 2H equivalent: sorted for determinism
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
        result = _try_order_2h_sp3(mol, center_idx, h1_idx, h2_idx)
        if result is not None:
            return result
        # 2H equivalent: sorted for determinism
        return sorted(h_indices)
    return _order_h_geometric(mol, center_idx, h_indices)


def _order_sp3d_axial_2h(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int,
    axial_indices: list[int],
    eq_indices: list[int]
) -> list[int] | None:
    """Order 2 axial H on SP3D via equatorial plane chirality.

    The two axial H are on opposite ends of the trigonal-bipyramidal axis.
    Whether they are equivalent depends on the three equatorial substituents:
    if all three have distinct CIP ranks, the equatorial plane is chiral
    and the two axial H are diastereotopic.

    Defines z⁺ as the direction along the axis from which the equatorial
    substituents (sorted by CIP rank descending) appear CCW.
    The axial H at the z⁺ end comes first.

    Returns [first_idx, second_idx] or None if H are equivalent.
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))

    # Check equivalence: need 3 distinct CIP ranks among eq substituents
    eq_ranks = []
    for eq_idx in eq_indices:
        props = mol.GetAtomWithIdx(eq_idx).GetPropsAsDict()
        if '_CIPRank' not in props:
            return None
        eq_ranks.append(int(props['_CIPRank']))
    if len(set(eq_ranks)) < len(eq_indices):
        return None  # duplicate ranks → eq plane symmetric → H equivalent

    # Sort eq by CIP rank descending
    eq_sorted = sorted(eq_indices, key=lambda i: -int(
        mol.GetAtomWithIdx(i).GetPropsAsDict()['_CIPRank']))
    a_idx, b_idx, c_idx = eq_sorted[0], eq_sorted[1], eq_sorted[2]

    # Axis direction: from h1 toward h2 (arbitrary initial choice)
    h1_pos = np.array(conf.GetAtomPosition(h1_idx))
    h2_pos = np.array(conf.GetAtomPosition(h2_idx))
    axis = h2_pos - h1_pos
    z_axis = axis / np.linalg.norm(axis)

    # Build orthonormal basis on plane ⟂ z_axis
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = arbitrary - np.dot(arbitrary, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Compute projected angles of eq substituents (CIP order: a→b→c)
    def _proj_angle(atom_idx):
        v = np.array(conf.GetAtomPosition(atom_idx)) - center_pos
        v_proj = v - np.dot(v, z_axis) * z_axis
        return np.arctan2(np.dot(v_proj, y_axis), np.dot(v_proj, x_axis))

    ang_a = _proj_angle(a_idx) % (2 * np.pi)
    ang_b = _proj_angle(b_idx) % (2 * np.pi)
    ang_c = _proj_angle(c_idx) % (2 * np.pi)

    # a→b→c is CCW if angles are in increasing cyclic order
    ccw = ((ang_a < ang_b < ang_c)
           or (ang_b < ang_c < ang_a)
           or (ang_c < ang_a < ang_b))

    # CCW when looking from h1 → h2 means h1 is at z⁺
    if ccw:
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
) -> list[int] | None:
    """Order 2 equatorial H on SP3D via axial CIP direction + atan2.

    The two equatorial H share the equatorial plane with one non-H
    substituent. The axis direction is defined by the two axial
    substituents: z⁺ points from the lower-CIP-rank axial to the
    higher-CIP-rank axial.

    On the plane ⟂ z⁺, the equatorial non-H substituent serves as
    the angle reference (0°). H atoms are ordered by CCW atan2 angle.

    Returns [first_idx, second_idx] or None if H are equivalent.
    """
    # Check equivalence: need distinct CIP ranks among axial substituents
    ax_ranks = []
    for ax_idx in axial_indices:
        props = mol.GetAtomWithIdx(ax_idx).GetPropsAsDict()
        if '_CIPRank' not in props:
            return None
        ax_ranks.append(int(props['_CIPRank']))
    if len(set(ax_ranks)) < 2:
        return None  # equal ranks → no defined z⁺ → H equivalent

    # z⁺: high CIP rank → low CIP rank
    ax_sorted = sorted(axial_indices, key=lambda i: -int(
        mol.GetAtomWithIdx(i).GetPropsAsDict()['_CIPRank']))
    ax_high, ax_low = ax_sorted[0], ax_sorted[1]

    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))

    # z_axis: ax_low → ax_high (z⁺ direction)
    ax_high_pos = np.array(conf.GetAtomPosition(ax_high))
    ax_low_pos = np.array(conf.GetAtomPosition(ax_low))
    z_axis = ax_high_pos - ax_low_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Find the non-H equatorial reference
    eq_non_h = [i for i in eq_indices if i not in (h1_idx, h2_idx)]
    if len(eq_non_h) != 1:
        return None
    ref_idx = eq_non_h[0]

    # Build orthonormal basis: x_axis toward eq reference
    ref_vec = np.array(conf.GetAtomPosition(ref_idx)) - center_pos
    x_vec = ref_vec - np.dot(ref_vec, z_axis) * z_axis
    x_norm = np.linalg.norm(x_vec)
    if x_norm < 1e-10:
        return None
    x_axis = x_vec / x_norm
    y_axis = np.cross(z_axis, x_axis)

    # Compute atan2 angles for H atoms
    angles = []
    for h_idx in (h1_idx, h2_idx):
        v = np.array(conf.GetAtomPosition(h_idx)) - center_pos
        v_proj = v - np.dot(v, z_axis) * z_axis
        angle = np.arctan2(np.dot(v_proj, y_axis), np.dot(v_proj, x_axis))
        angles.append((angle, h_idx))

    angles.sort()
    return [h_idx for _, h_idx in angles]


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
        ordered = _order_sp3d_axial_2h(
            mol, center_idx, axial_h[0], axial_h[1], axial_nbrs, eq_nbrs
        )
        if ordered is not None:
            result.extend(ordered)
        else:
            # 2H equivalent: sorted for determinism
            result.extend(sorted(axial_h))
    elif len(axial_h) == 1:
        result.extend(axial_h)

    # Order equatorial H group
    if len(eq_h) == 2:
        ordered = _order_sp3d_equatorial_2h(
            mol, center_idx, eq_h[0], eq_h[1], axial_nbrs, eq_nbrs
        )
        if ordered is not None:
            result.extend(ordered)
        else:
            # 2H equivalent: sorted for determinism
            result.extend(sorted(eq_h))
    elif len(eq_h) == 3:
        # eq 3H: z⁺ from axial CIP/_CanonicalOrder, CCW geometric
        ref = _get_z_plus_ref(mol, (axial_nbrs[0], axial_nbrs[1]))
        result.extend(_order_h_geometric(mol, center_idx, eq_h, ref_idx=ref))
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
    sq_ranks = {}
    for idx in square_indices:
        props = mol.GetAtomWithIdx(idx).GetPropsAsDict()
        if '_CIPRank' not in props:
            return None
        sq_ranks[idx] = int(props['_CIPRank'])

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

    # Gram-Schmidt basis on ⟂ plane
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = arbitrary - np.dot(arbitrary, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    def _proj_angle(atom_idx):
        v = np.array(conf.GetAtomPosition(atom_idx)) - center_pos
        v_proj = v - np.dot(v, z_axis) * z_axis
        return np.arctan2(np.dot(v_proj, y_axis), np.dot(v_proj, x_axis))

    # kept[0], kept[1], kept[2] are in CIP descending order
    ang0 = _proj_angle(kept[0]) % (2 * np.pi)
    ang1 = _proj_angle(kept[1]) % (2 * np.pi)
    ang2 = _proj_angle(kept[2]) % (2 * np.pi)

    ccw = ((ang0 < ang1 < ang2)
           or (ang1 < ang2 < ang0)
           or (ang2 < ang0 < ang1))

    return ccw


def _order_sp3d2_trans_2h(
    mol: Chem.Mol,
    center_idx: int,
    h1_idx: int,
    h2_idx: int,
    trans_pairs: list[tuple[int, int]]
) -> list[int] | None:
    """Order 2 trans H on an octahedral center via square chirality.

    The 4 cis substituents form a square. Analyzes its chirality:
    CW when looking from h1→h2 means h1 is at z⁺.
    Returns [z⁺_idx, z⁻_idx] or None if equivalent.
    """
    # Find the 4 cis substituents (all neighbors except h1 and h2)
    center_atom = mol.GetAtomWithIdx(center_idx)
    all_nbrs = {n.GetIdx() for n in center_atom.GetNeighbors()}
    h_set = {h1_idx, h2_idx}
    square_indices = list(all_nbrs - h_set)

    if len(square_indices) != 4:
        return None

    ccw = _analyze_square_chirality(
        mol, center_idx, h1_idx, h2_idx, square_indices
    )

    if ccw is None:
        return None
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
) -> list[int] | None:
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
        return None

    r1 = int(mol.GetAtomWithIdx(t1).GetPropsAsDict().get('_CIPRank', 0))
    r2 = int(mol.GetAtomWithIdx(t2).GetPropsAsDict().get('_CIPRank', 0))

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

    return None


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
    Geometric calls use explicit trans-pair z⁺ via _get_z_plus_ref when
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
            result = _order_sp3d2_trans_2h(
                mol, center_idx, h1_idx, h2_idx, trans_pairs
            )
        else:
            result = _order_sp3d2_cis_2h(
                mol, center_idx, h1_idx, h2_idx, trans_pairs
            )
        if result is not None:
            return result
        # 2H equivalent: sorted for determinism
        return sorted(h_indices)

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
                r = int(mol.GetAtomWithIdx(x).GetPropsAsDict().get(
                    '_CIPRank', 0))
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
                cis_result = _order_sp3d2_cis_2h(
                    mol, center_idx,
                    ranks[dup_rank][0], ranks[dup_rank][1], trans_pairs
                )
                if cis_result is not None:
                    result.extend(cis_result)
                else:
                    # 2H equivalent: sorted for determinism
                    result.extend(sorted(ranks[dup_rank]))
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

                cis_result = _order_sp3d2_cis_2h(
                    mol_tmp, center_idx, h2, h3, trans_pairs
                )
                if cis_result is not None:
                    result.extend(cis_result)
                else:
                    result.extend(sorted([h2, h3]))
        else:
            # mer: 1 H-H + 1 H-X
            if hx_pairs:
                result.append(hx_pairs[0][0])
            for h_a, h_b in hh_pairs:
                ordered = _order_sp3d2_trans_2h(
                    mol, center_idx, h_a, h_b, trans_pairs
                )
                if ordered is not None:
                    result.extend(ordered)
                else:
                    # 2H equivalent: sorted for determinism
                    result.extend(sorted([h_a, h_b]))

    # --- 4H ---
    elif n_h == 4:
        if len(hx_pairs) == 0:
            # non-H trans: 2 H-H pairs + 1 non-H trans pair
            # Use trans non-H pair for z⁺ direction
            ref = _get_z_plus_ref(mol, xx_pairs[0])
            result = _order_h_geometric(mol, center_idx, h_indices, ref_idx=ref)
        else:
            # non-H cis: 2 H-X + 1 H-H
            def _trans_rank(hx):
                return int(mol.GetAtomWithIdx(hx[1]).GetPropsAsDict().get(
                    '_CIPRank', 0))

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
                ordered = _order_sp3d2_trans_2h(
                    mol, center_idx, h_a, h_b, trans_pairs
                )
                if ordered is not None:
                    result.extend(ordered)
                else:
                    # 2H equivalent: sorted for determinism
                    result.extend(sorted([h_a, h_b]))

    # --- 5H ---
    elif n_h == 5:
        # 1 H-X + 2 H-H. H-X H first (unique trans partner).
        if hx_pairs:
            h_first, x = hx_pairs[0]
            result.append(h_first)
            eq_h = [h for h in h_indices if h != h_first]
            # Use trans H-X pair for z⁺ (x is at z⁺ end)
            result.extend(_order_h_geometric(mol, center_idx, eq_h, ref_idx=x))

    # --- 6H ---
    else:
        # 3 H-H pairs. Pick one trans pair as ax (min-idx).
        sorted_hh = sorted(hh_pairs, key=lambda p: min(p))
        ax_a, ax_b = sorted_hh[0]
        # ax 2H equivalent: min idx first
        result.extend([min(ax_a, ax_b), max(ax_a, ax_b)])
        eq_h = [h for h in h_indices if h not in result]
        # Use ax trans pair for z⁺ direction
        ref = _get_z_plus_ref(mol, (ax_a, ax_b))
        result.extend(_order_h_geometric(mol, center_idx, eq_h, ref_idx=ref))

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
