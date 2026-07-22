import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


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


def _signed_angle_between(
    axis: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    """Signed angle from v1 to v2 projected onto plane ⟂ axis.

    Uses right-hand rule around axis: positive = CCW, negative = CW.
    Returns angle in (-π, π].
    """
    v1_proj = v1 - np.dot(v1, axis) * axis
    v2_proj = v2 - np.dot(v2, axis) * axis
    cross = np.dot(axis, np.cross(v1_proj, v2_proj))
    dot = np.dot(v1_proj, v2_proj)
    return float(np.arctan2(cross, dot))


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


def _deuterate_atom(mol: Chem.Mol, atom_idx: int) -> Chem.Mol:
    """Return a copy of mol with atom_idx deuterated and stereochemistry reassigned."""
    mol_tmp = Chem.RWMol(Chem.Mol(mol))
    Chem.AssignAtomChiralTagsFromStructure(mol_tmp)
    mol_tmp.GetAtomWithIdx(atom_idx).SetIsotope(2)
    Chem.AssignStereochemistry(mol_tmp, cleanIt=True, force=True)
    return mol_tmp


def _get_z_plus_vec(
    mol: Chem.Mol,
    pair_indices: tuple[int, int]
) -> np.ndarray:
    """Determine the z⁺ direction (unit vector) of a trans/axial pair.

    z⁺ direction: z⁻ → z⁺ along the pair axis.  z⁺ end is determined by:
      - Both H    → smaller index (has no chemical meaning; deterministic)
      - H + X     → X (always higher CIP rank than H)
      - Both X    → CIP rank (higher → z⁺), then _CanonicalOrder (larger → z⁺)
    """
    a, b = pair_indices
    a_is_h = mol.GetAtomWithIdx(a).GetAtomicNum() == 1
    b_is_h = mol.GetAtomWithIdx(b).GetAtomicNum() == 1

    if a_is_h and b_is_h:
        z_plus = a if a < b else b
    elif a_is_h:
        z_plus = b  # X > H, X is z⁺
    elif b_is_h:
        z_plus = a  # X > H, X is z⁺
    else:
        # Both heavy atoms: use CIP rank + _CanonicalOrder
        r0 = _get_cip_rank(mol.GetAtomWithIdx(a))
        r1 = _get_cip_rank(mol.GetAtomWithIdx(b))
        if r0 != r1:
            z_plus = a if r0 > r1 else b
        else:
            pos0 = mol.GetAtomWithIdx(a).GetIntProp('_CanonicalOrder')
            pos1 = mol.GetAtomWithIdx(b).GetIntProp('_CanonicalOrder')
            z_plus = a if pos0 > pos1 else b

    z_minus = a if b == z_plus else b

    conf = mol.GetConformer()
    pos_plus = np.array(conf.GetAtomPosition(z_plus))
    pos_minus = np.array(conf.GetAtomPosition(z_minus))
    vec = pos_plus - pos_minus
    return vec / np.linalg.norm(vec)


def _compute_pair_angles(
    mol: Chem.Mol,
    center_idx: int
) -> list[tuple[float, int, int]]:
    """Return (angle_rad, idx_i, idx_j) for all neighbor pairs, sorted descending."""
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    center_atom = mol.GetAtomWithIdx(center_idx)
    all_nbrs = list(center_atom.GetNeighbors())

    vecs = {n.GetIdx(): np.array(conf.GetAtomPosition(n.GetIdx())) - center_pos
            for n in all_nbrs}
    nbr_indices = [n.GetIdx() for n in all_nbrs]

    pairs = []
    for i in range(len(nbr_indices)):
        for j in range(i + 1, len(nbr_indices)):
            vi = vecs[nbr_indices[i]]
            vj = vecs[nbr_indices[j]]
            cos_angle = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            angle = np.arccos(cos_angle)
            pairs.append((angle, nbr_indices[i], nbr_indices[j]))

    pairs.sort(key=lambda x: -x[0])
    return pairs


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


def _walk_cumulene_far_end(
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
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                other = bond.GetOtherAtomIdx(cursor_idx)
                if other != prev_idx:
                    next_atoms.append(other)
        if len(next_atoms) != 1:
            return None
        prev_idx = cursor_idx
        cursor_idx = next_atoms[0]
        cursor_atom = mol.GetAtomWithIdx(cursor_idx)

    return (cursor_idx, prev_idx, sp_count)


def _order_2h_cumulene(
    mol: Chem.Mol,
    center_idx: int,
    partner_idx: int,
    h_indices: list[int],
) -> list[int]:
    """Order 2 H on terminal =CH2 of cumulene (including allene).

    Odd sp count (axial chirality, e.g. propadiene):
      Projects H1, H2 and the far-end highest-CIP substituent onto a plane
      perpendicular to the C=C=C axis, then computes signed angles.
      CW arc → S_a (H2 = pro-R_a), CCW → R_a (H1 = pro-R_a).
      Returns [pro-R_a_idx, pro-S_a_idx].

    Even sp count (coplanar, e.g. butatriene):
      Uses the far-end highest-CIP substituent as dihedral reference.
      |dihedral| < 90° → pro-Z.
      Returns [pro-Z_idx, pro-E_idx].

    Returns sorted if H are equivalent.
    """
    h1_idx, h2_idx = h_indices
    far_info = _walk_cumulene_far_end(mol, center_idx, partner_idx)
    if far_info is None:
        return sorted(h_indices)
    far_idx, prev_idx, sp_count = far_info

    far_subs = [
        n for n in mol.GetAtomWithIdx(far_idx).GetNeighbors()
        if n.GetIdx() != prev_idx
    ]
    if not far_subs:
        return sorted(h_indices)

    ranks = [_get_cip_rank(n) for n in far_subs]
    if len(far_subs) > 1 and len(set(ranks)) == 1:
        return sorted(h_indices)  # Equivalent substituents → H are equivalent

    far_c = max(far_subs, key=lambda n: _get_cip_rank(n))

    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    partner_pos = np.array(conf.GetAtomPosition(partner_idx))
    h1_pos = np.array(conf.GetAtomPosition(h1_idx))
    far_c_pos = np.array(conf.GetAtomPosition(far_c.GetIdx()))

    if sp_count % 2 == 1:
        # Odd: axial chirality (planes perpendicular)
        axis = partner_pos - center_pos
        axis = axis / np.linalg.norm(axis)

        angle = _signed_angle_between(
            axis, h1_pos - center_pos, far_c_pos - center_pos
        )

        # CIP (IUPAC): view along axis near→far, trace
        #   near-big → near-small → far-big  (skip far-small)
        #   CW = R_a, CCW = S_a
        #
        # signed_angle(H1, far_c) is OPPOSITE to the three-point CIP path
        # because H1 and H2 are ~180° apart in the projection plane.
        if angle < 0:
            return [h2_idx, h1_idx]  # h2 is pro-R_a
        else:
            return [h1_idx, h2_idx]  # h1 is pro-R_a
    else:
        # Even: coplanar (pro-Z / pro-E via dihedral)
        dihedral = rdMolTransforms.GetDihedralDeg(
            conf, h1_idx, center_idx, partner_idx, far_c.GetIdx()
        )
        if abs(dihedral) < 90:
            return [h1_idx, h2_idx]  # h1 is pro-Z
        else:
            return [h2_idx, h1_idx]  # h2 is pro-Z


def _order_2h_sp2(
    mol: Chem.Mol,
    center_idx: int,
    partner_idx: int,
    h_indices: list[int],
) -> list[int]:
    """Order 2 H on sp2 =CH2 via CIP rank + dihedral angle.

    Uses the highest-CIP-ranked substituent on the partner atom as reference.
    |dihedral| < 90 deg means h1 is cis to ref (pro-Z).

    For cumulenes (partner is SP), delegates to axial chirality ordering.

    Returns [pro-Z_idx, pro-E_idx] or sorted if H are equivalent.
    """
    h1_idx, h2_idx = h_indices
    partner_atom = mol.GetAtomWithIdx(partner_idx)

    # Allene/cumulene: partner is sp → axial chirality ordering
    if partner_atom.GetHybridization() == Chem.HybridizationType.SP:
        return _order_2h_cumulene(mol, center_idx, partner_idx, h_indices)

    # Normal sp2: partner's directly-bonded substituents
    partner_subs = [
        n for n in partner_atom.GetNeighbors()
        if n.GetIdx() != center_idx
    ]
    if not partner_subs:
        return sorted(h_indices)

    ranks = [_get_cip_rank(n) for n in partner_subs]
    if len(partner_subs) > 1 and len(set(ranks)) == 1:
        return sorted(h_indices)  # All substituents equivalent -> H are truly equivalent

    ref_idx = max(partner_subs, key=lambda n: _get_cip_rank(n)).GetIdx()
    conf = mol.GetConformer()
    dihedral = rdMolTransforms.GetDihedralDeg(conf, h1_idx, center_idx, partner_idx, ref_idx)

    if abs(dihedral) < 90:
        return [h1_idx, h2_idx]  # h1 is pro-Z
    else:
        return [h2_idx, h1_idx]  # h2 is pro-Z


def _order_h_sp2(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H on an sp2 center: Z/E, cumulene axial chirality, or geometric.

    sp2 does not imply a double bond — carbocations, radicals, and
    heteroatoms (e.g. BH3) can be sp2 with only single bonds.
    """
    if len(h_indices) == 2:
        center_atom = mol.GetAtomWithIdx(center_idx)
        for bond in center_atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                partner_idx = bond.GetOtherAtomIdx(center_idx)
                return _order_2h_sp2(mol, center_idx, partner_idx, h_indices)
        return sorted(h_indices)
    elif len(h_indices) == 3:
        conf = mol.GetConformer()
        center_pos = np.array(conf.GetAtomPosition(center_idx))
        sorted_h = sorted(h_indices)
        v1 = np.array(conf.GetAtomPosition(sorted_h[0])) - center_pos
        v2 = np.array(conf.GetAtomPosition(sorted_h[1])) - center_pos
        z_axis = np.cross(v1, v2)
        z_axis = z_axis / np.linalg.norm(z_axis)
        return _order_h_geometric(mol, center_idx, h_indices, z_axis=z_axis)


def _order_2h_signed_volume(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
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
    h1_idx, h2_idx = h_indices
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
            return sorted(h_indices)  # H equivalent

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
        return sorted(h_indices)


def _order_2h_sp3(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
) -> list[int]:
    """
    Order 2 H on sp3 center via deuterium substitution + CIP assignment.
    Replaces h1 with D, then checks if center becomes R or S.
    Falls back to signed-volume method for non-carbon centers (P, S, etc.).

    Returns [pro-R_idx, pro-S_idx] or sorted if undetermined.
    """
    h1_idx, h2_idx = h_indices
    mol_tmp = _deuterate_atom(mol, h1_idx)
    cip = mol_tmp.GetAtomWithIdx(center_idx).GetPropsAsDict().get('_CIPCode')

    if cip == 'R':
        return [h1_idx, h2_idx]  # h1 is pro-R
    elif cip == 'S':
        return [h2_idx, h1_idx]  # h2 is pro-R

    # RDKit cannot assign CIP (non-carbon centers: P, S, As, etc.)
    # Fall back to manual signed-volume determination
    return _order_2h_signed_volume(mol, center_idx, h_indices)


def _order_h_sp3(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H on an sp3 center: CIP pro-R/S (deuterium -> signed volume)."""
    if len(h_indices) == 2:
        return _order_2h_sp3(mol, center_idx, h_indices)
    return _order_h_geometric(mol, center_idx, h_indices)


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
    center_atom = mol.GetAtomWithIdx(center_idx)
    all_nbrs = list(center_atom.GetNeighbors())
    nbr_indices = [n.GetIdx() for n in all_nbrs]

    if len(all_nbrs) != 5:
        return ([], nbr_indices)

    pairs = _compute_pair_angles(mol, center_idx)
    max_angle, i, j = pairs[0]

    if max_angle < np.radians(140.0):
        return ([], nbr_indices)

    axial_indices = [i, j]
    equatorial_indices = [idx for idx in nbr_indices if idx not in axial_indices]
    return (axial_indices, equatorial_indices)


def _order_sp3d_axial_2h(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    eq_indices: list[int],
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
    h1_idx, h2_idx = h_indices
    conf = mol.GetConformer()

    # Check equivalence: need 3 distinct CIP ranks among eq substituents
    eq_ranks = [_get_cip_rank(mol.GetAtomWithIdx(i)) for i in eq_indices]
    if len(set(eq_ranks)) < len(eq_indices):
        return sorted(h_indices)  # duplicate ranks → eq plane symmetric → H equivalent

    # Sort eq by CIP rank descending
    eq_sorted = sorted(eq_indices, key=lambda i: -_get_cip_rank(mol.GetAtomWithIdx(i)))

    # Axis direction: from h1 toward h2 (arbitrary initial choice)
    h1_pos = np.array(conf.GetAtomPosition(h1_idx))
    h2_pos = np.array(conf.GetAtomPosition(h2_idx))
    axis = h2_pos - h1_pos
    z_axis = axis / np.linalg.norm(axis)

    # a→b→c CCW when looking from h1 → h2 → h1 is at z⁺
    if _is_ccw(mol, center_idx, z_axis, tuple(eq_sorted)):
        return [h1_idx, h2_idx]
    else:
        return [h2_idx, h1_idx]


def _order_sp3d_equatorial_2h(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    axial_indices: list[int],
    eq_indices: list[int],
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
    h1_idx, h2_idx = h_indices
    # Check equivalence: need distinct CIP ranks among axial substituents
    ax_ranks = [_get_cip_rank(mol.GetAtomWithIdx(i)) for i in axial_indices]
    if len(set(ax_ranks)) < 2:
        return sorted(h_indices)  # equal ranks → no defined z⁺ → H equivalent

    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(center_idx))
    z_axis = _get_z_plus_vec(mol, tuple(axial_indices))

    ref_idx = [i for i in eq_indices if i not in (h1_idx, h2_idx)][0]

    ref_vec = np.array(conf.GetAtomPosition(ref_idx)) - center_pos
    h_pos_list = [(h, np.array(conf.GetAtomPosition(h))) for h in (h1_idx, h2_idx)]
    return _order_h_by_angle_projection(
        center_pos, z_axis, h_pos_list, x_direction=ref_vec)


def _order_h_sp3d(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H on an SP3D (trigonal bipyramidal) center.

    Dispatches by H count first (like SP3D2), so each branch knows the
    ax/eq H distribution and avoids _get_cip_rank on H-only groups.
    Axial H are ordered before equatorial H in the result.
    """
    axial_nbrs, eq_nbrs = _classify_sp3d_positions(mol, center_idx)
    if len(axial_nbrs) != 2:
        return _order_h_geometric(mol, center_idx, h_indices)

    n_h = len(h_indices)
    axial_h = [h for h in h_indices if h in axial_nbrs]
    eq_h = [h for h in h_indices if h in eq_nbrs]

    if n_h == 2:
        if len(axial_h) == 2 and len(eq_h) == 0:
            return _order_sp3d_axial_2h(mol, center_idx, axial_h, eq_nbrs)
        elif len(axial_h) == 0 and len(eq_h) == 2:
            return _order_sp3d_equatorial_2h(mol, center_idx, eq_h, axial_nbrs, eq_nbrs)
        elif len(axial_h) == 1 and len(eq_h) == 1:
            return sorted(axial_h) + sorted(eq_h)

    elif n_h == 3:
        if len(axial_h) == 2 and len(eq_h) == 1:
            return _order_sp3d_axial_2h(mol, center_idx, axial_h, eq_nbrs) + eq_h
        elif len(axial_h) == 1 and len(eq_h) == 2:
            return axial_h + _order_sp3d_equatorial_2h(mol, center_idx, eq_h, axial_nbrs, eq_nbrs)
        elif len(axial_h) == 0 and len(eq_h) == 3:
            z_axis = _get_z_plus_vec(mol, tuple(axial_nbrs))
            return _order_h_geometric(mol, center_idx, eq_h, z_axis=z_axis)

    elif n_h == 4:
        if len(axial_h) == 2 and len(eq_h) == 2:
            return sorted(axial_h) + sorted(eq_h)
        elif len(axial_h) == 1 and len(eq_h) == 3:
            z_axis = _get_z_plus_vec(mol, tuple(axial_nbrs))
            return axial_h + _order_h_geometric(mol, center_idx, eq_h, z_axis=z_axis)

    elif n_h == 5:
        return sorted(axial_h) + _order_h_geometric(
            mol, center_idx, eq_h,
            z_axis=_get_z_plus_vec(mol, tuple(axial_nbrs))
        )


def _find_sp3d2_trans_pairs(
    mol: Chem.Mol,
    center_idx: int
) -> list[tuple[int, int]]:
    """Find the three trans (180°) pairs in an octahedral center.

    Returns up to 3 (idx1, idx2) tuples, sorted by angle descending.
    Only pairs with angle > 150° are included.
    """
    pairs = _compute_pair_angles(mol, center_idx)

    used: set[int] = set()
    trans_pairs: list[tuple[int, int]] = []
    for angle, i, j in pairs:
        if angle > np.radians(150.0):
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
    square_indices: list[int],
    trans_of: dict[int, int],
) -> bool | None:
    """Analyze chirality of a 4-point square on the plane ⟂ the z axis.

    z_from → z_to defines the axis direction. The four atoms in
    square_indices lie approximately in the perpendicular plane.

    Steps:
      1. Reject if fewer than 3 distinct CIP ranks.
      2. Identify diagonal pairs (trans within the square).
         Reject if any diagonal has matching CIP ranks.
      3. Remove one atom (prefer duplicate rank, then lowest rank)
         to obtain 3 atoms with distinct CIP ranks.
      4. Triangle CW/CCW analysis on the 3 remaining atoms.

    Returns True if CCW when looking along z_from→z_to, False if CW,
    None if the square is symmetric (H equivalent).
    """
    conf = mol.GetConformer()

    # Collect CIP ranks
    sq_ranks = {idx: _get_cip_rank(mol.GetAtomWithIdx(idx)) for idx in square_indices}

    # --- Step 1: require at least 3 distinct CIP ranks ---
    rank_values = list(sq_ranks.values())
    if len(set(rank_values)) < 3:
        return None

    # --- Step 2: check diagonals (trans pairs within the square) ---
    sq_set = set(square_indices)
    diagonals = []
    for idx in square_indices:
        partner = trans_of[idx]
        if partner in sq_set and idx < partner:
            diagonals.append((idx, partner))

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

    # --- Step 4: triangle CW/CCW analysis ---
    # Axis direction: z_from → z_to
    h_from_pos = np.array(conf.GetAtomPosition(z_from))
    h_to_pos = np.array(conf.GetAtomPosition(z_to))
    z_axis = h_to_pos - h_from_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    return _is_ccw(mol, center_idx, z_axis, tuple(kept))


def _order_sp3d2_trans_2h(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    trans_of: dict[int, int],
) -> list[int]:
    """Order 2 trans H on an octahedral center via square chirality.

    The 4 cis substituents form a square. Analyzes its chirality:
    CCW when looking from h1→h2 means h1 is at z⁺.
    Returns [z⁺_idx, z⁻_idx] or sorted if equivalent.
    """
    h1_idx, h2_idx = h_indices
    center_atom = mol.GetAtomWithIdx(center_idx)
    all_nbrs = {n.GetIdx() for n in center_atom.GetNeighbors()}
    square_indices = list(all_nbrs - {h1_idx, h2_idx})

    ccw = _analyze_square_chirality(
        mol, center_idx, h1_idx, h2_idx, square_indices, trans_of
    )

    if ccw is None:
        return sorted(h_indices)
    elif ccw:
        return [h1_idx, h2_idx]  # CCW: h1 at z⁺
    else:
        return [h2_idx, h1_idx]  # CW: h2 at z⁺


def _order_sp3d2_cis_2h(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    trans_of: dict[int, int],
) -> list[int]:
    """Order 2 cis H on an octahedral center.

    Each H belongs to a different trans pair. Order by the CIP rank
    of their trans partners. If the trans partners have equal CIP rank,
    analyze each H's cis square chirality (viewed along T→H direction).
    """
    h1_idx, h2_idx = h_indices
    t1 = trans_of[h1_idx]
    t2 = trans_of[h2_idx]

    r1 = _get_cip_rank(mol.GetAtomWithIdx(t1))
    r2 = _get_cip_rank(mol.GetAtomWithIdx(t2))

    if r1 != r2:
        # Order by trans partner CIP rank (higher rank H first)
        if r1 > r2:
            return [h1_idx, h2_idx]
        else:
            return [h2_idx, h1_idx]
    else:
        # T_a = T_b: use cis-square chirality (viewed along T→H)
        center_atom = mol.GetAtomWithIdx(center_idx)
        all_nbrs = {n.GetIdx() for n in center_atom.GetNeighbors()}

        # H_a's cis square: all neighbors except H_a and T_a
        cis1 = list(all_nbrs - {h1_idx, t1})
        # H_b's cis square: all neighbors except H_b and T_b
        cis2 = list(all_nbrs - {h2_idx, t2})

        ccw1 = _analyze_square_chirality(mol, center_idx, t1, h1_idx, cis1, trans_of)
        ccw2 = _analyze_square_chirality(mol, center_idx, t2, h2_idx, cis2, trans_of)

        # The H whose cis square is CCW from T→H comes first
        if ccw1 is not None and ccw2 is not None:
            if ccw1 != ccw2:
                if ccw1:
                    return [h1_idx, h2_idx]
                else:
                    return [h2_idx, h1_idx]

        return sorted(h_indices)


def _order_sp3d2_3h(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    trans_of: dict[int, int],
    hh_pairs: list[tuple[int, int]],
    hx_pairs: list[tuple[int, int]],
) -> list[int]:
    """Order 3 H on an SP3D2 center (fac or mer)."""
    result: list[int] = []

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
            # AAB: unique H (singleton rank) + 2H (duplicate rank) → 2H cis
            unique_h = None
            dup_hs = None
            for hs in ranks.values():
                if len(hs) == 1:
                    unique_h = hs[0]
                else:
                    dup_hs = hs
            result.append(unique_h)
            result.extend(_order_sp3d2_cis_2h(
                mol, center_idx, dup_hs, trans_of
            ))
        elif len(ranks) == 1:
            # All equivalent → min-idx H first, deuterate, cis-2H
            h_sorted = sorted(h_indices)
            first_h = h_sorted[0]
            h2, h3 = h_sorted[1], h_sorted[2]
            result.append(first_h)
            mol_tmp = _deuterate_atom(mol, first_h)
            result.extend(_order_sp3d2_cis_2h(
                mol_tmp, center_idx, [h2, h3], trans_of
            ))

    elif len(hh_pairs) == 1 and len(hx_pairs) == 1:
        # mer: 1 H-H + 1 H-X. H-X H first, then H-H via trans_2h.
        result.append(hx_pairs[0][0])
        h_a, h_b = hh_pairs[0]
        result.extend(_order_sp3d2_trans_2h(
            mol, center_idx, [h_a, h_b], trans_of
        ))

    return result


def _order_sp3d2_4h(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    trans_of: dict[int, int],
    hh_pairs: list[tuple[int, int]],
    hx_pairs: list[tuple[int, int]],
    xx_pairs: list[tuple[int, int]],
) -> list[int]:
    """Order 4 H on an SP3D2 center (non-H trans or non-H cis)."""
    result: list[int] = []

    if len(hx_pairs) == 0 and len(hh_pairs) == 2:
        # non-H trans: 2 H-H pairs + 1 non-H trans pair.
        # Use trans non-H pair for z⁺ direction.
        z_axis = _get_z_plus_vec(mol, xx_pairs[0])
        result = _order_h_geometric(mol, center_idx, h_indices, z_axis=z_axis)

    elif len(hx_pairs) == 2 and len(hh_pairs) == 1:
        # non-H cis: 2 H-X + 1 H-H.
        # H-X group: order by trans partner CIP rank descending.
        ranks_hx = [_get_cip_rank(mol.GetAtomWithIdx(x)) for _, x in hx_pairs]
        if len(set(ranks_hx)) == 2:
            hx_pairs.sort(key=lambda p: _get_cip_rank(mol.GetAtomWithIdx(p[1])),
                          reverse=True)
            for h, _ in hx_pairs:
                result.append(h)
        else:
            # Equal rank → 2H equivalent: sorted
            result.extend(sorted(h for h, _ in hx_pairs))
        # H-H pair
        for h_a, h_b in hh_pairs:
            result.extend(_order_sp3d2_trans_2h(
                mol, center_idx, [h_a, h_b], trans_of
            ))

    return result


def _order_sp3d2_5h(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    hx_pairs: list[tuple[int, int]],
) -> list[int]:
    """Order 5 H on an SP3D2 center: 1 H-X + 2 H-H.

    H-X H (ax) first, remaining 4H in equatorial plane via CCW geometric.
    """
    h_first, x = hx_pairs[0]
    result = [h_first]
    eq_h = [h for h in h_indices if h != h_first]
    z_axis = _get_z_plus_vec(mol, (h_first, x))
    result.extend(_order_h_geometric(mol, center_idx, eq_h, z_axis=z_axis))
    return result


def _order_sp3d2_6h(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int],
    hh_pairs: list[tuple[int, int]],
) -> list[int]:
    """Order 6 H on an SP3D2 center: 3 H-H pairs.

    Pick a trans pair as ax (min-idx), ax pair first,
    remaining 4H via CCW geometric along ax direction.
    """
    sorted_hh = sorted(hh_pairs, key=lambda p: min(p))
    ax_a, ax_b = sorted_hh[0]
    # ax 2H equivalent: min idx first
    result = [min(ax_a, ax_b), max(ax_a, ax_b)]
    eq_h = [h for h in h_indices if h not in result]
    z_axis = _get_z_plus_vec(mol, (ax_a, ax_b))
    result.extend(_order_h_geometric(mol, center_idx, eq_h, z_axis=z_axis))
    return result


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

    Equivalent 2H are returned sorted (no geometric needed).
    Geometric calls use explicit trans-pair z⁺ via _get_z_plus_vec when
    CIP ranks are equal.
    """
    n_h = len(h_indices)

    trans_pairs = _find_sp3d2_trans_pairs(mol, center_idx)
    if len(trans_pairs) != 3:
        return _order_h_geometric(mol, center_idx, h_indices)

    trans_of = {}
    for a, b in trans_pairs:
        trans_of[a] = b
        trans_of[b] = a

    # Classify trans pairs by H content
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

    if n_h == 2:
        if len(hh_pairs) == 1 and len(xx_pairs) == 2:
            return _order_sp3d2_trans_2h(
                mol, center_idx, h_indices, trans_of
            )
        elif len(xx_pairs) == 1 and len(hx_pairs) == 2:
            return _order_sp3d2_cis_2h(
                mol, center_idx, h_indices, trans_of
            )
    elif n_h == 3:
        return _order_sp3d2_3h(
            mol, center_idx, h_indices, trans_of, hh_pairs, hx_pairs
        )
    elif n_h == 4:
        return _order_sp3d2_4h(
            mol, center_idx, h_indices, trans_of,
            hh_pairs, hx_pairs, xx_pairs
        )
    elif n_h == 5:
        return _order_sp3d2_5h(
            mol, center_idx, h_indices, hx_pairs
        )
    elif n_h == 6:
        return _order_sp3d2_6h(
            mol, center_idx, h_indices, hh_pairs
        )


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
