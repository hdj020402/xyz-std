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


def _order_h_geometric(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H atoms by geometric CCW angle projection.

    Picks a reference axis (strongest non-H neighbor, or min-index H if
    none), then sorts all H by CCW angle around that axis.

    Args:
        mol: RDKit Mol with explicit H and a 3D conformer
        center_idx: Index of the heavy atom center
        h_indices: Indices of H atoms attached to center

    Returns:
        Deterministically ordered list of H atom indices
    """
    if len(h_indices) <= 1:
        return list(h_indices)

    conf = mol.GetConformer()
    center_atom = mol.GetAtomWithIdx(center_idx)
    non_h_neighbors = [n for n in center_atom.GetNeighbors() if n.GetAtomicNum() != 1]
    center_pos = np.array(conf.GetAtomPosition(center_idx))

    if not non_h_neighbors:
        # No non-H reference (e.g., CH4, PH5): pick one H as the
        # reference axis, place it first, then order the rest by CCW
        # projection around center->ref_H.
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

    Separates axial from equatorial H. Axial H are ordered before
    equatorial H. Within each group, geometric CCW is used.
    """
    axial_nbrs, eq_nbrs = _classify_sp3d_positions(mol, center_idx)

    axial_h = [h for h in h_indices if h in axial_nbrs]
    eq_h = [h for h in h_indices if h in eq_nbrs]
    other_h = [h for h in h_indices if h not in axial_nbrs and h not in eq_nbrs]

    result = []
    if axial_h:
        result.extend(_order_h_geometric(mol, center_idx, axial_h))
    if eq_h:
        result.extend(_order_h_geometric(mol, center_idx, eq_h))
    if other_h:
        result.extend(_order_h_geometric(mol, center_idx, other_h))
    return result


def _order_h_sp3d2(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H on an SP3D2 (octahedral) center.

    All six positions are equivalent in a regular octahedron.
    Uses geometric CCW for deterministic ordering.
    """
    return _order_h_geometric(mol, center_idx, h_indices)


def _order_h_on_heavy_atom(
    mol: Chem.Mol,
    center_idx: int,
    h_indices: list[int]
) -> list[int]:
    """Order H atoms on a heavy atom using 3D-aware methods.

    Dispatches by RDKit hybridization:
      - SP2: 2H via Z/E or allene axial chirality
      - SP3: 2H via CIP pro-R/S (deuterium → signed volume fallback)
      - SP3D: axial/equatorial separation + geometric CCW
      - SP3D2: geometric CCW (all positions equivalent)
      - Other: geometric CCW fallback

    For >=3 H on any center type, falls through to geometric CCW.

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
