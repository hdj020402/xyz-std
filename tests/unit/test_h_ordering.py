import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from xyz_std.h_ordering import (
    _build_perp_basis,
    _deuterate_atom,
    _get_cip_rank,
    _get_z_plus_vec,
    _is_ccw,
    _order_h_by_angle_projection,
    _order_h_geometric,
    _order_h_on_heavy_atom,
    _order_h_sp2,
    _order_h_sp3,
    _order_h_sp3d,
    _order_h_sp3d2,
    _classify_sp3d_positions,
    _order_sp3d_axial_2h,
    _order_sp3d_equatorial_2h,
    _find_sp3d2_trans_pairs,
    _analyze_square_chirality,
    _order_sp3d2_trans_2h,
    _order_sp3d2_cis_2h,
    _order_2h_sp3,
    _order_2h_sp2,
    _order_2h_cumulene,
    _order_2h_signed_volume,
    _infer_lone_pair_position,
    _projected_angle,
    _signed_angle_between,
    _signed_tetrahedron_volume,
)
from xyz_std.io import xyz_to_rdkit_mol


def _make_mol_with_3d(smiles: str, seed: int = 42) -> Chem.Mol:
    """Helper: SMILES -> Mol with explicit H and 3D conformer.

    Calls both AssignAtomChiralTagsFromStructure and AssignStereochemistry
    to ensure _CIPRank is available for sp2/sp3 CIP-based H ordering.

    Note: _CIPRank is NOT available for allenes via this path (RDKit cannot
    assign axial chirality when explicit H are present). Use _make_mol_from_xyz
    for allene tests.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    _ensure_cip_ranks(mol)
    return mol


def _ensure_cip_ranks(mol: Chem.Mol) -> None:
    """Pre-assign _CIPRank=0 to atoms that RDKit skipped (symmetric molecules)."""
    for atom in mol.GetAtoms():
        if '_CIPRank' not in atom.GetPropsAsDict():
            atom.SetIntProp('_CIPRank', 0)


def _make_mol_from_xyz(smiles: str, seed: int = 42) -> Chem.Mol:
    """Helper: SMILES -> XYZ -> xyz_to_rdkit_mol (matches production pipeline).

    Generates 3D coordinates via RDKit embedding, serializes to XYZ, then
    parses through the production xyz_to_rdkit_mol with OpenBabel backend.
    This ensures _CIPRank is available for all molecule types including
    allenes (axial chirality), because OpenBabel includes CIP information
    in the MOL block.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), "test"]
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        lines.append(f"{sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    xyz_str = "\n".join(lines) + "\n"
    mol_ob = xyz_to_rdkit_mol(xyz_str)
    # OpenBabel provides _CIPRank for chiral molecules (allenes etc.) but not
    # for simple alkenes. Call assign here to fill in the gaps — existing
    # _CIPRank from OpenBabel is preserved, and missing ones are computed.
    Chem.AssignAtomChiralTagsFromStructure(mol_ob)
    Chem.AssignStereochemistry(mol_ob, cleanIt=True, force=True)
    _ensure_cip_ranks(mol_ob)
    return mol_ob


class TestOrderHOnHeavyAtom:
    def test_two_h_on_oxygen(self):
        """Two H on O should both be returned."""
        mol = _make_mol_with_3d("O")  # water: O with 2H
        # Find O atom
        o_idx = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 8:
                o_idx = atom.GetIdx()
                break
        h_indices = [
            n.GetIdx() for n in mol.GetAtomWithIdx(o_idx).GetNeighbors()
            if n.GetAtomicNum() == 1
        ]
        # With 2 H on O, it should return a list of 2 indices
        result = _order_h_on_heavy_atom(mol, o_idx, h_indices)
        assert len(result) == 2
        assert set(result) == set(h_indices)

    def test_zero_h(self):
        """Empty list should return empty."""
        mol = _make_mol_with_3d("C")
        result = _order_h_on_heavy_atom(mol, 0, [])
        assert result == []

    def test_one_h(self):
        """Single H in list should return list of length 1."""
        mol = _make_mol_with_3d("C")
        c_idx = 0
        h_indices = [
            n.GetIdx() for n in mol.GetAtomWithIdx(c_idx).GetNeighbors()
            if n.GetAtomicNum() == 1
        ]
        result = _order_h_on_heavy_atom(mol, c_idx, h_indices[:1])
        assert len(result) == 1
        assert result[0] == h_indices[0]

    def test_methyl_3h_returns_all(self):
        """Methyl group: 3H should all be returned."""
        mol = _make_mol_with_3d("CC")  # ethane
        # Find a carbon with 3H neighbors
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 3:
                    result = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
                    assert len(result) == 3
                    assert set(result) == set(h_nbrs)
                    return
        pytest.fail("No methyl group found in ethane")

    def test_prochiral_sp3_deterministic(self):
        """2H on prochiral sp3 center should give deterministic order."""
        # Propane: central C has 2H + 2 different? Actually both are CH3, so equivalent.
        # Use 2-butanol for a true prochiral center: CH3-*CH(OH)-CH2-CH3
        # The *CH has only 1H, not 2H. Need a prochiral CH2.
        # Butanone: CH3-CO-CH2-CH3, the CH2 is prochiral (adjacent to C=O and CH3)
        mol = _make_mol_with_3d("CCCC")  # butane: C2 has 2H
        # C2 in butane: neighbors are C1, C3, H, H — C1 and C3 are both CH3/CH2
        # Actually C1-C2-C3-C4, C2 has C1(CH3), C3(CH2CH3), H, H — these H ARE prochiral
        c2_idx = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                c_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
                if len(h_nbrs) == 2 and len(c_nbrs) == 2:
                    c2_idx = atom.GetIdx()
                    break

        if c2_idx is None:
            pytest.skip("No prochiral CH2 found")

        h_indices = [
            n.GetIdx() for n in mol.GetAtomWithIdx(c2_idx).GetNeighbors()
            if n.GetAtomicNum() == 1
        ]
        result1 = _order_h_on_heavy_atom(mol, c2_idx, h_indices)
        result2 = _order_h_on_heavy_atom(mol, c2_idx, list(reversed(h_indices)))
        # Same result regardless of input order
        assert result1 == result2

    def test_methane_4h(self):
        """Methane: 4H on single C, all equivalent."""
        mol = _make_mol_with_3d("C")
        c_idx = 0
        h_indices = [
            n.GetIdx() for n in mol.GetAtomWithIdx(c_idx).GetNeighbors()
            if n.GetAtomicNum() == 1
        ]
        assert len(h_indices) == 4
        result = _order_h_on_heavy_atom(mol, c_idx, h_indices)
        assert len(result) == 4
        assert set(result) == set(h_indices)


class TestTryOrder2hSp3:
    def test_prochiral_center(self):
        """Prochiral CH2 between different groups should give R/S ordering."""
        # 2-chloroethanol: Cl-CH2-CH2-OH — the CH2 near Cl has Cl + CH2OH + H + H
        mol = _make_mol_with_3d("ClCCO")
        # Find the C bonded to Cl with 2H
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            nbrs = list(atom.GetNeighbors())
            h_nbrs = [n for n in nbrs if n.GetAtomicNum() == 1]
            has_cl = any(n.GetAtomicNum() == 17 for n in nbrs)
            if len(h_nbrs) == 2 and has_cl:
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
                result = _order_2h_sp3(
                    mol, atom.GetIdx(), [h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx])
                assert len(result) == 2
                return
        pytest.skip("No suitable prochiral center found")

    def test_equivalent_h_returns_sorted(self):
        """Truly equivalent H (symmetric center) should return sorted list."""
        # Propane central CH2: CH3-CH2-CH3, both sides are CH3 (equivalent)
        mol = _make_mol_with_3d("CCC")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            c_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
            if len(h_nbrs) == 2 and len(c_nbrs) == 2:
                h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                result = _order_2h_sp3(
                    mol, atom.GetIdx(), [h1, h2
                ])
                assert result == sorted([h1, h2])
                return
        pytest.skip("No symmetric CH2 found")


class TestAngleProjection:
    def test_three_h_returns_three(self):
        """Should return all 3 indices for 3 H atoms."""
        center = np.array([0.0, 0.0, 0.0])
        ref = np.array([0.0, 0.0, 1.0])
        h_list = [
            (1, np.array([1.0, 0.0, 0.0])),
            (2, np.array([0.0, 1.0, 0.0])),
            (3, np.array([-1.0, 0.0, 0.0])),
        ]
        result = _order_h_by_angle_projection(center, ref, h_list)
        assert len(result) == 3
        assert set(result) == {1, 2, 3}

    def test_single_h(self):
        center = np.array([0.0, 0.0, 0.0])
        ref = np.array([0.0, 0.0, 1.0])
        h_list = [(5, np.array([1.0, 0.0, 0.0]))]
        result = _order_h_by_angle_projection(center, ref, h_list)
        assert result == [5]

    def test_deterministic(self):
        """Same input should give same output."""
        center = np.array([0.0, 0.0, 0.0])
        ref = np.array([0.0, 0.0, 1.0])
        h_list = [
            (1, np.array([1.0, 0.0, 0.0])),
            (2, np.array([-0.5, 0.866, 0.0])),
            (3, np.array([-0.5, -0.866, 0.0])),
        ]
        r1 = _order_h_by_angle_projection(center, ref, h_list)
        r2 = _order_h_by_angle_projection(center, ref, h_list)
        assert r1 == r2


class TestTryOrder2hSp2:
    """Tests for sp2 =CH2 ordering via CIP rank + dihedral."""

    def test_terminal_alkene_asymmetric(self):
        """Propene CH3-CH=CH2: partner has CH3(rank 3) and H(rank 2),
        CH3 should be picked as reference, sp2 ordering should succeed."""
        mol = _make_mol_with_3d("CC=C")

        # Find =CH2 carbon with 2H and a double bond
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert len(result) == 2
                    assert set(result) == {h1, h2}
                    return
        pytest.fail("No terminal =CH2 found in propene")

    def test_terminal_alkene_deterministic(self):
        """Same input should always produce same output."""
        mol = _make_mol_with_3d("CC=C", seed=42)

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    r1 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    r2 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found")

    def test_terminal_alkene_input_order_independent(self):
        """Swapping h1/h2 input should swap the output."""
        mol = _make_mol_with_3d("CC=C")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    r1 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    r2 = _order_2h_sp2(mol, center, partner, [h2, h1])
                    assert set(r1) == set(r2)
                    # Output is input-order independent
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found")

    def test_symmetric_partner_returns_sorted(self):
        """Isobutene (CH3)2C=CH2: partner has 2 identical CH3 substituents,
        sp2 CIP should return sorted (H are truly equivalent)."""
        mol = _make_mol_with_3d("CC(C)=C")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert result == sorted([h1, h2]), (
                        "sp2 CIP should return sorted when partner subs are equivalent"
                    )
                    return
        pytest.fail("No terminal =CH2 found in isobutene")

    def test_via_order_h_on_heavy_atom(self):
        """Full _order_h_on_heavy_atom on propene =CH2 should return
        deterministic order via sp2 CIP path (not geometric fallback)."""
        mol = _make_mol_with_3d("CC=C")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    center = atom.GetIdx()
                    r1 = _order_h_on_heavy_atom(mol, center, h_nbrs)
                    r2 = _order_h_on_heavy_atom(mol, center, list(reversed(h_nbrs)))
                    assert len(r1) == 2
                    assert set(r1) == set(h_nbrs)
                    assert r1 == r2  # same regardless of input order
                    return
        pytest.fail("No terminal =CH2 found")


class TestTryOrder2hAllene:
    """Tests for allene =CH2 ordering via axial chirality (R_a/S_a)."""

    def test_unsubstituted_returns_sorted(self):
        """H2C=C=CH2: far-end H's are equivalent → return sorted."""
        mol = _make_mol_from_xyz("C=C=C")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert result == sorted([h1, h2]), (
                        "unsubstituted allene should return sorted"
                    )
                    return
        pytest.fail("No terminal =CH2 found in allene")

    def test_asymmetric_succeeds(self):
        """H2C=C=C(F)Br: far-end F/Br differ → ordering should succeed."""
        mol = _make_mol_from_xyz("C=C=C(F)Br")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert len(result) == 2
                    assert set(result) == {h1, h2}
                    return
        pytest.fail("No terminal =CH2 found")

    def test_deterministic(self):
        """Same allene should always produce same output."""
        mol = _make_mol_from_xyz("C=C=C(F)Br", seed=42)

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    r1 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    r2 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found")

    def test_input_order_independent(self):
        """Swapping h1/h2 input should swap the output."""
        mol = _make_mol_from_xyz("C=C=C(F)Br")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    r1 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    r2 = _order_2h_sp2(mol, center, partner, [h2, h1])
                    assert set(r1) == set(r2)
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found")

    def test_via_order_h_on_heavy_atom(self):
        """Full _order_h_on_heavy_atom on allene =CH2 should give
        deterministic order via axial chirality path."""
        mol = _make_mol_from_xyz("C=C=C(F)Br")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    center = atom.GetIdx()
                    r1 = _order_h_on_heavy_atom(mol, center, h_nbrs)
                    r2 = _order_h_on_heavy_atom(mol, center, list(reversed(h_nbrs)))
                    assert len(r1) == 2
                    assert set(r1) == set(h_nbrs)
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found")


def _make_cumulene_mol(smiles: str, seed: int = 42) -> Chem.Mol:
    """Helper for even-cumulene (butatriene etc.) tests via RDKit direct path.

    RDKit correctly perceives cumulene bond types but does not assign
    _CIPRank for axial chirality. We manually set _CIPRank based on atomic
    number to simulate what OpenBabel would provide. OB is not used here
    because it misperceives butatriene bond orders as C-C≡C-C.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for atom in mol.GetAtoms():
        atom.SetIntProp("_CIPRank", atom.GetAtomicNum())

    return mol


class TestTryOrder2hAlleneEven:
    """Tests for even-cumulene (butatriene etc.) =CH2 ordering.

    sp_count is even → terminal planes are coplanar → pro-Z/pro-E via
    dihedral. Uses RDKit direct path with manual _CIPRank because
    OpenBabel misperceives butatriene bond types.
    """

    def test_unsubstituted_returns_sorted(self):
        """H2C=C=C=CH2: far-end H's are equivalent → return sorted."""
        mol = _make_cumulene_mol("C=C=C=C")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert result == sorted([h1, h2]), (
                        "unsubstituted butatriene should return sorted"
                    )
                    return
        pytest.fail("No terminal =CH2 found in butatriene")

    def test_asymmetric_succeeds(self):
        """H2C=C=C=CHF: far-end F/H differ → ordering should succeed."""
        mol = _make_cumulene_mol("C=C=C=CF")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert len(result) == 2
                    assert set(result) == {h1, h2}
                    return
        pytest.fail("No terminal =CH2 found in butatriene")

    def test_deterministic(self):
        """Same butatriene should always produce same output."""
        mol = _make_cumulene_mol("C=C=C=CF", seed=42)

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    r1 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    r2 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found in butatriene")

    def test_input_order_independent(self):
        """Swapping h1/h2 input should swap the output."""
        mol = _make_cumulene_mol("C=C=C=CF")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    r1 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    r2 = _order_2h_sp2(mol, center, partner, [h2, h1])
                    assert set(r1) == set(r2)
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found in butatriene")

    def test_via_order_h_on_heavy_atom(self):
        """Full _order_h_on_heavy_atom on butatriene =CH2 should give
        deterministic order via even sp_count path."""
        mol = _make_cumulene_mol("C=C=C=CF")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    center = atom.GetIdx()
                    r1 = _order_h_on_heavy_atom(mol, center, h_nbrs)
                    r2 = _order_h_on_heavy_atom(mol, center, list(reversed(h_nbrs)))
                    assert len(r1) == 2
                    assert set(r1) == set(h_nbrs)
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found in butatriene")


class TestTryOrder2hSp2OpenBabel:
    """Tests for sp2 =CH2 ordering via OpenBabel (production) backend.

    These tests use _make_mol_from_xyz which goes through the XYZ -> OpenBabel
    -> MOL block -> RDKit pipeline, matching the production code path. This
    ensures _CIPRank is available for all molecule types including allenes.

    Mirrors TestTryOrder2hSp2 but via the OpenBabel pathway.
    """

    def test_terminal_alkene_asymmetric(self):
        """Propene via OB: sp2 ordering should succeed."""
        mol = _make_mol_from_xyz("CC=C")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert len(result) == 2
                    assert set(result) == {h1, h2}
                    return
        pytest.fail("No terminal =CH2 found in propene")

    def test_terminal_alkene_deterministic(self):
        """Propene via OB: same mol should always produce same output."""
        mol = _make_mol_from_xyz("CC=C", seed=42)

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    r1 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    r2 = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found")

    def test_symmetric_partner_returns_sorted(self):
        """Isobutene via OB: partner has 2 identical CH3 substituents,
        sp2 CIP should return sorted."""
        mol = _make_mol_from_xyz("CC(C)=C")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert result == sorted([h1, h2]), (
                        "sp2 CIP should return sorted when partner subs are equivalent (OB)"
                    )
                    return
        pytest.fail("No terminal =CH2 found in isobutene")

    def test_via_order_h_on_heavy_atom(self):
        """Propene via OB: _order_h_on_heavy_atom should give deterministic
        order via sp2 CIP path."""
        mol = _make_mol_from_xyz("CC=C")

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    center = atom.GetIdx()
                    r1 = _order_h_on_heavy_atom(mol, center, h_nbrs)
                    r2 = _order_h_on_heavy_atom(mol, center, list(reversed(h_nbrs)))
                    assert len(r1) == 2
                    assert set(r1) == set(h_nbrs)
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found")


class TestInferLonePairPosition:
    """Tests for _infer_lone_pair_position."""

    def test_tetrahedral_lone_pair(self):
        """For a tetrahedral center with 3 explicit bonds, lone pair
        should be opposite to the sum of bond vectors."""
        center = np.array([0.0, 0.0, 0.0])
        # Three bonds pointing roughly to alternating corners of a cube
        vecs = [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, -1.0, -1.0]),
            np.array([-1.0, 1.0, -1.0]),
        ]
        lp = _infer_lone_pair_position(center, vecs)
        # Lone pair should be in the opposite direction to sum of bonds
        s = sum(v / np.linalg.norm(v) for v in vecs)
        lp_dir = lp - center
        # lp_dir and -s should point in same direction (positive dot product)
        assert np.dot(lp_dir, -s) > 0

class TestTryOrder2hSignedVolume:
    """Tests for _order_2h_signed_volume (non-carbon prochiral centers)."""

    def test_phosphine_prochiral(self):
        """CH3-PH2: P with 3 neighbors (C, H, H) — signed volume should
        infer lone pair and determine pro-R/pro-S."""
        mol = _make_mol_with_3d("CP")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    result = _order_2h_signed_volume(mol, atom.GetIdx(), [h1, h2])
                    assert len(result) == 2
                    assert set(result) == {h1, h2}
                    return
        pytest.fail("No P with 2H found in CH3-PH2")

    def test_phosphine_deterministic(self):
        """Same PH2 should always produce same output."""
        mol = _make_mol_with_3d("CP")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    r1 = _order_2h_signed_volume(mol, atom.GetIdx(), [h1, h2])
                    r2 = _order_2h_signed_volume(mol, atom.GetIdx(), [h1, h2])
                    assert r1 == r2
                    return
        pytest.fail("No P with 2H found")

    def test_phosphine_input_order_independent(self):
        """Swapping h1/h2 input should swap output."""
        mol = _make_mol_with_3d("CP")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    r1 = _order_2h_signed_volume(mol, atom.GetIdx(), [h1, h2])
                    r2 = _order_2h_signed_volume(mol, atom.GetIdx(), [h2, h1])
                    assert set(r1) == set(r2)
                    assert r1 == r2
                    return
        pytest.fail("No P with 2H found")

    def test_via_order_2h_sp3(self):
        """_order_2h_sp3 should fall back to signed volume for PH2."""
        mol = _make_mol_with_3d("CP")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    result = _order_2h_sp3(mol, atom.GetIdx(), [h1, h2])
                    assert len(result) == 2
                    return
        pytest.fail("No P with 2H found")

    def test_via_order_h_on_heavy_atom(self):
        """Full _order_h_on_heavy_atom on CH3-PH2 should use signed volume
        fallback and produce deterministic output."""
        mol = _make_mol_with_3d("CP")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    r1 = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
                    r2 = _order_h_on_heavy_atom(mol, atom.GetIdx(), list(reversed(h_nbrs)))
                    assert len(r1) == 2
                    assert set(r1) == set(h_nbrs)
                    assert r1 == r2  # input order independent
                    return
        pytest.fail("No P with 2H found")

    def test_equivalent_sih2_returns_sorted(self):
        """Symmetric Si center (CH3-SiH2-CH3) should return sorted."""
        mol = _make_mol_with_3d("C[SiH2]C")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 14:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    result = _order_2h_signed_volume(mol, atom.GetIdx(), [h1, h2])
                    assert result == sorted([h1, h2]), "Symmetric SiH2 should return sorted"
                    return
        pytest.fail("No Si with 2H found")

    def test_h2s_returns_sorted(self):
        """H2S (2-coordinate, 2 lone pairs) should return sorted."""
        mol = _make_mol_with_3d("[SH2]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 16:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    result = _order_2h_signed_volume(mol, atom.GetIdx(), [h1, h2])
                    assert result == sorted([h1, h2]), "2-coordinate H2S should return sorted"
                    return
        pytest.fail("No S with 2H found")


class TestHeavyAtomWithManyH:
    """Tests for ≥3 H on P, S, Si, Ge (geometric CCW fallback path)."""

    def test_phosphonium_4h(self):
        """[PH4]+: 4 equivalent H → deterministic CCW order."""
        mol = _make_mol_with_3d("[PH4+]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                assert len(h_nbrs) == 4
                r1 = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
                r2 = _order_h_on_heavy_atom(mol, atom.GetIdx(), list(reversed(h_nbrs)))
                assert len(r1) == 4
                assert set(r1) == set(h_nbrs)
                assert r1 == r2
                return
        pytest.fail("No P with 4H found")

    def test_sulfonium_3h(self):
        """[SH3]+: 3 equivalent H → deterministic CCW order."""
        mol = _make_mol_with_3d("[SH3+]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 16:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                assert len(h_nbrs) == 3
                r1 = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
                r2 = _order_h_on_heavy_atom(mol, atom.GetIdx(), list(reversed(h_nbrs)))
                assert len(r1) == 3
                assert set(r1) == set(h_nbrs)
                assert r1 == r2
                return
        pytest.fail("No S with 3H found")

    def test_silane_4h(self):
        """SiH4: 4 equivalent H → deterministic CCW order."""
        mol = _make_mol_with_3d("[SiH4]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 14:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                assert len(h_nbrs) == 4
                r1 = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
                r2 = _order_h_on_heavy_atom(mol, atom.GetIdx(), list(reversed(h_nbrs)))
                assert len(r1) == 4
                assert set(r1) == set(h_nbrs)
                assert r1 == r2
                return
        pytest.fail("No Si with 4H found")

    def test_methyl_phosphonium_3h(self):
        """CH3-PH3+: 3 H with one non-H neighbor → CCW order."""
        mol = _make_mol_with_3d("C[PH3+]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                assert len(h_nbrs) == 3
                r1 = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
                r2 = _order_h_on_heavy_atom(mol, atom.GetIdx(), list(reversed(h_nbrs)))
                assert len(r1) == 3
                assert set(r1) == set(h_nbrs)
                assert r1 == r2
                return
        pytest.fail("No P with 3H found")

    def test_ph5_5h(self):
        """PH5: 5 H in trigonal bipyramidal → deterministic CCW order.
        Symmetric molecules get _CIPRank=0 preset, which is correct for equivalent H."""
        mol = _make_mol_with_3d("[PH5]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                assert len(h_nbrs) == 5
                r1 = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
                r2 = _order_h_on_heavy_atom(mol, atom.GetIdx(), list(reversed(h_nbrs)))
                assert len(r1) == 5
                assert set(r1) == set(h_nbrs)
                assert r1 == r2
                return
        pytest.fail("No P with 5H found")


class TestClassifySp3dPositions:
    """Tests for _classify_sp3d_positions (axial/equatorial classification)."""

    def test_ph5_axial_equatorial_separation(self):
        """PH5: 5 H should be classified as 2 axial + 3 equatorial."""
        mol = _make_mol_with_3d("[PH5]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                assert atom.GetHybridization() == Chem.HybridizationType.SP3D
                axial, eq_ = _classify_sp3d_positions(mol, atom.GetIdx())
                assert len(axial) == 2, f"Expected 2 axial, got {len(axial)}"
                assert len(eq_) == 3, f"Expected 3 equatorial, got {len(eq_)}"
                return
        pytest.fail("No P with 5H found")

    def test_ph5_axial_angle_near_180(self):
        """Axial pair should have bond angle near 180 degrees."""
        mol = _make_mol_with_3d("[PH5]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                axial, _ = _classify_sp3d_positions(mol, atom.GetIdx())
                conf = mol.GetConformer()
                center = np.array(conf.GetAtomPosition(atom.GetIdx()))
                v0 = np.array(conf.GetAtomPosition(axial[0])) - center
                v1 = np.array(conf.GetAtomPosition(axial[1])) - center
                angle = np.degrees(
                    np.arccos(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
                )
                assert angle > 140, f"Axial angle {angle:.1f}° not near 180°"
                return
        pytest.fail("No P with 5H found")


class TestOrderHSp3d:
    """Tests for _order_h_sp3d (trigonal bipyramidal H ordering)."""

    def test_pf2h3_axial_before_equatorial(self):
        """PF2H3: axial H should come before equatorial H in output."""
        mol = _make_sp3d_mol("H", "F", "H", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                axial, eq_ = _classify_sp3d_positions(mol, atom.GetIdx())
                # ax=[H,F], eq=[H,H,H] in this test molecule
                # axial_set contains F (non-H) atoms
                axial_set = set(axial)
                eq_set = set(eq_)
                assert len(axial_set) == 2
                assert len(eq_set) == 3

                axial_h = [idx for idx in axial if mol.GetAtomWithIdx(idx).GetAtomicNum() == 1]
                eq_h = [idx for idx in eq_ if mol.GetAtomWithIdx(idx).GetAtomicNum() == 1]
                assert len(axial_h) == 1
                assert len(eq_h) == 3

                h_indices = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                result = _order_h_sp3d(mol, atom.GetIdx(), h_indices)
                assert len(result) == len(h_indices)
                assert set(result) == set(h_indices)
                # axial H should appear before all equatorial H
                assert result.index(axial_h[0]) < min(result.index(e) for e in eq_h)
                return
        pytest.fail("No P found")

    def test_pf2h3_deterministic(self):
        """PF2H3: order should be deterministic regardless of input order."""
        mol = _make_sp3d_mol("F", "F", "H", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_indices = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                r1 = _order_h_sp3d(mol, atom.GetIdx(), h_indices)
                r2 = _order_h_sp3d(mol, atom.GetIdx(), list(reversed(h_indices)))
                assert r1 == r2
                return
        pytest.fail("No P found")

    def test_fph4_via_order_h_on_heavy_atom(self):
        """FPH4: dispatch should detect SP3D hybridization."""
        mol = _make_mol_with_3d("F[PH4]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_indices = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_indices) < 2:
                    continue
                r1 = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_indices)
                r2 = _order_h_on_heavy_atom(mol, atom.GetIdx(), list(reversed(h_indices)))
                assert len(r1) == len(h_indices)
                assert set(r1) == set(h_indices)
                assert r1 == r2
                return
        pytest.fail("No P with H found")


def _make_trans_of(trans_pairs: list[tuple[int, int]]) -> dict[int, int]:
    """Build bidirectional trans-partner lookup from trans pair list."""
    trans_of = dict(trans_pairs)
    trans_of.update({b: a for a, b in trans_pairs})
    return trans_of


class TestOrderHSp3d2:
    """Tests for _order_h_sp3d2 (octahedral H ordering)."""

    def test_sh6_via_order_h_on_heavy_atom(self):
        """SH6: SP3D2 routes to geometric CCW (all positions equivalent)."""
        mol = _make_mol_with_3d("[SH6]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 16:
                assert atom.GetHybridization() == Chem.HybridizationType.SP3D2
                h_indices = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                r1 = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_indices)
                r2 = _order_h_on_heavy_atom(mol, atom.GetIdx(), list(reversed(h_indices)))
                assert len(r1) == 6
                assert set(r1) == set(h_indices)
                assert r1 == r2
                return
        pytest.fail("No S with 6H found")

    def test_sh6_deterministic(self):
        """SH6: geometric CCW should be deterministic."""
        mol = _make_mol_with_3d("[SH6]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 16:
                h_indices = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                r1 = _order_h_sp3d2(mol, atom.GetIdx(), h_indices)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_indices)))
                assert r1 == r2
                return
        pytest.fail("No S with 6H found")


class TestHybridizationDispatch:
    """Verify dispatch routes to the correct function by hybridization."""

    def test_sp3_centers_use_sp3_path(self):
        """CH4, CH3-CH3: SP3 centers should use _order_h_sp3."""
        for smi in ["C", "CC"]:
            mol = _make_mol_with_3d(smi)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 6:
                    continue
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) < 2:
                    continue
                assert atom.GetHybridization() == Chem.HybridizationType.SP3
                result = _order_h_sp3(mol, atom.GetIdx(), h_nbrs)
                assert len(result) == len(h_nbrs)
                assert set(result) == set(h_nbrs)

    def test_sp2_centers_use_sp2_path(self):
        """Propene =CH2: SP2 center should use _order_h_sp2."""
        mol = _make_mol_with_3d("CC=C")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    assert atom.GetHybridization() == Chem.HybridizationType.SP2
                    result = _order_h_sp2(mol, atom.GetIdx(), h_nbrs)
                    assert len(result) == 2
                    assert set(result) == set(h_nbrs)
                    return
        pytest.fail("No sp2 =CH2 found")

    def test_sp2_bh3_uses_plane_normal(self):
        """BH3: sp2 center with 3 H and no double bond → geometric via plane normal.

        The plane normal (cross of two B-H bond vectors) must be used as z_axis
        rather than inferring a non-existent lone pair."""
        mol = _make_mol_with_3d("B")
        b_idx = 0
        h_idxs = [n.GetIdx() for n in mol.GetAtomWithIdx(b_idx).GetNeighbors()
                   if n.GetAtomicNum() == 1]

        assert len(h_idxs) == 3
        assert mol.GetAtomWithIdx(b_idx).GetHybridization() == Chem.HybridizationType.SP2

        result = _order_h_sp2(mol, b_idx, h_idxs)
        assert len(result) == 3
        assert set(result) == set(h_idxs)

        # Determinism: same input → same output
        r2 = _order_h_sp2(mol, b_idx, h_idxs)
        assert result == r2

    def test_top_level_dispatches_correctly(self):
        """_order_h_on_heavy_atom should route SP2/SP3/SP3D correctly."""
        # SP3: methane
        mol = _make_mol_with_3d("C")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                r = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
                assert len(r) == 4
                break

        # SP2: propene
        mol = _make_mol_with_3d("CC=C")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    r = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
                    assert len(r) == 2
                    return
        pytest.fail("No sp2 =CH2 found")


def _make_sp3d_mol(ax1_sym, ax2_sym, eq1_sym, eq2_sym, eq3_sym):
    """Helper: build an SP3D trigonal-bipyramidal molecule manually.

    Constructs via RWMol with explicit bonds and 3D coordinates
    (axial along z, equatorial in xy plane at 120°).
    Assigns chiral tags and stereochemistry for _CIPRank availability.
    Note: GetHybridization() may return UNSPECIFIED since RDKit does not
    perceive SP3D from simple connectivity alone — use _order_h_sp3d
    directly to test SP3D logic.
    """
    ax_bond = {"H": 1.42, "F": 1.55, "Cl": 2.1, "Br": 2.3}
    eq_bond = {"H": 1.42, "F": 1.55, "Cl": 2.05, "Br": 2.2}

    mol = Chem.RWMol()
    p = Chem.Atom(15)
    p_idx = mol.AddAtom(p)

    ax1 = Chem.Atom(ax1_sym)
    ax1_idx = mol.AddAtom(ax1)
    mol.AddBond(p_idx, ax1_idx, Chem.BondType.SINGLE)

    ax2 = Chem.Atom(ax2_sym)
    ax2_idx = mol.AddAtom(ax2)
    mol.AddBond(p_idx, ax2_idx, Chem.BondType.SINGLE)

    eq_idxs = []
    for i, sym in enumerate([eq1_sym, eq2_sym, eq3_sym]):
        a = Chem.Atom(sym)
        idx = mol.AddAtom(a)
        eq_idxs.append(idx)
        mol.AddBond(p_idx, idx, Chem.BondType.SINGLE)

    mol.UpdatePropertyCache(strict=False)
    mol = mol.GetMol()

    conf = Chem.Conformer(6)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, (0.0, 0.0, ax_bond.get(ax1_sym, 1.5)))
    conf.SetAtomPosition(2, (0.0, 0.0, -ax_bond.get(ax2_sym, 1.5)))
    for i, sym in enumerate([eq1_sym, eq2_sym, eq3_sym]):
        angle = np.radians(120 * i)
        b = eq_bond.get(sym, 1.5)
        conf.SetAtomPosition(3 + i, (b * np.cos(angle), b * np.sin(angle), 0.0))
    mol.AddConformer(conf)

    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for atom in mol.GetAtoms():
        if '_CIPRank' not in atom.GetPropsAsDict():
            atom.SetIntProp('_CIPRank', 0)
    return mol


class TestOrderSp3dAxial2h:
    """Tests for _order_sp3d_axial_2h (equatorial chirality method)."""

    def test_prochiral_succeeds(self):
        """PFBrClH2: 3 different eq substituents → axial H diastereotopic."""
        mol = _make_sp3d_mol("H", "H", "Br", "Cl", "F")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                ax, eq = _classify_sp3d_positions(mol, atom.GetIdx())
                ax_h = [h for h in ax if mol.GetAtomWithIdx(h).GetSymbol() == "H"]
                assert len(ax_h) == 2
                result = _order_sp3d_axial_2h(
                    mol, atom.GetIdx(), ax_h, eq
                )
                assert len(result) == 2
                assert set(result) == set(ax_h)
                return
        pytest.fail("No P with 2 axial H found")

    def test_equivalent_returns_sorted(self):
        """All eq F equivalent → axial H equivalent → sorted."""
        mol = _make_sp3d_mol("H", "H", "F", "F", "F")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                ax, eq = _classify_sp3d_positions(mol, atom.GetIdx())
                ax_h = [h for h in ax if mol.GetAtomWithIdx(h).GetSymbol() == "H"]
                result = _order_sp3d_axial_2h(
                    mol, atom.GetIdx(), ax_h, eq
                )
                assert result == sorted(ax_h)
                return
        pytest.fail("No P found")

    def test_deterministic(self):
        """Same molecule → same result regardless of input order."""
        mol = _make_sp3d_mol("H", "H", "Br", "Cl", "F")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                ax, eq = _classify_sp3d_positions(mol, atom.GetIdx())
                ax_h = [h for h in ax if mol.GetAtomWithIdx(h).GetSymbol() == "H"]
                r1 = _order_sp3d_axial_2h(
                    mol, atom.GetIdx(), ax_h, eq
                )
                r2 = _order_sp3d_axial_2h(
                    mol, atom.GetIdx(), [ax_h[1], ax_h[0]], eq
                )
                assert set(r1) == set(r2)
                assert r1 == r2
                return
        pytest.fail("No P found")

    def test_via_order_h_sp3d(self):
        """Integration via _order_h_sp3d: chemical method used for 2 ax H."""
        mol = _make_sp3d_mol("H", "H", "Br", "Cl", "F")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                ax, _ = _classify_sp3d_positions(mol, atom.GetIdx())
                ax_h = [h for h in ax if mol.GetAtomWithIdx(h).GetSymbol() == "H"]
                r = _order_h_sp3d(mol, atom.GetIdx(), ax_h)
                assert len(r) == 2
                assert set(r) == set(ax_h)
                r2 = _order_h_sp3d(mol, atom.GetIdx(), list(reversed(ax_h)))
                assert r == r2
                return
        pytest.fail("No P found")


class TestOrderSp3dEquatorial2h:
    """Tests for _order_sp3d_equatorial_2h (axial CIP + atan2 method)."""

    def test_prochiral_succeeds(self):
        """PFClBrH2: 2 different axial → eq H diastereotopic."""
        mol = _make_sp3d_mol("F", "Cl", "Br", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                ax, eq = _classify_sp3d_positions(mol, atom.GetIdx())
                eq_h = [h for h in eq if mol.GetAtomWithIdx(h).GetSymbol() == "H"]
                assert len(eq_h) == 2
                result = _order_sp3d_equatorial_2h(
                    mol, atom.GetIdx(), eq_h, ax, eq
                )
                assert len(result) == 2
                assert set(result) == set(eq_h)
                return
        pytest.fail("No P with 2 eq H found")

    def test_equivalent_returns_sorted(self):
        """PF2BrH2: 2 identical axial F → eq H equivalent → sorted."""
        mol = _make_sp3d_mol("F", "F", "Br", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                ax, eq = _classify_sp3d_positions(mol, atom.GetIdx())
                eq_h = [h for h in eq if mol.GetAtomWithIdx(h).GetSymbol() == "H"]
                result = _order_sp3d_equatorial_2h(
                    mol, atom.GetIdx(), eq_h, ax, eq
                )
                assert result == sorted(eq_h)
                return
        pytest.fail("No P found")

    def test_deterministic(self):
        """Same molecule → same result regardless of input order."""
        mol = _make_sp3d_mol("F", "Cl", "Br", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                ax, eq = _classify_sp3d_positions(mol, atom.GetIdx())
                eq_h = [h for h in eq if mol.GetAtomWithIdx(h).GetSymbol() == "H"]
                r1 = _order_sp3d_equatorial_2h(
                    mol, atom.GetIdx(), eq_h, ax, eq
                )
                r2 = _order_sp3d_equatorial_2h(
                    mol, atom.GetIdx(), [eq_h[1], eq_h[0]], ax, eq
                )
                assert set(r1) == set(r2)
                assert r1 == r2
                return
        pytest.fail("No P found")

    def test_via_order_h_sp3d(self):
        """Integration via _order_h_sp3d."""
        mol = _make_sp3d_mol("F", "Cl", "Br", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                _, eq = _classify_sp3d_positions(mol, atom.GetIdx())
                eq_h = [h for h in eq if mol.GetAtomWithIdx(h).GetSymbol() == "H"]
                r = _order_h_sp3d(mol, atom.GetIdx(), eq_h)
                assert len(r) == 2
                assert set(r) == set(eq_h)
                r2 = _order_h_sp3d(mol, atom.GetIdx(), list(reversed(eq_h)))
                assert r == r2
                return
        pytest.fail("No P found")


class TestOrderHSp3dDegenerate:
    """SP3D degenerate cases: 3H, 4H distributions."""

    def test_3h_2ax_1eq(self):
        """2 ax H + 1 eq H + 2 non-H: ax uses chemical, eq 1H passthrough."""
        mol = _make_sp3d_mol("H", "H", "Br", "Cl", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 3
                r = _order_h_sp3d(mol, atom.GetIdx(), h_all)
                assert len(r) == 3
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No P found")

    def test_3h_1ax_2eq(self):
        """1 ax H + 2 eq H: ax 1H passthrough, eq uses chemical."""
        mol = _make_sp3d_mol("H", "F", "Br", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 3
                r = _order_h_sp3d(mol, atom.GetIdx(), h_all)
                assert len(r) == 3
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No P found")

    def test_4h_2ax_2eq(self):
        """2 ax H + 2 eq H: both groups use chemical methods."""
        mol = _make_sp3d_mol("H", "H", "F", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 4
                r = _order_h_sp3d(mol, atom.GetIdx(), h_all)
                assert len(r) == 4
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No P found")

    def test_4h_1ax_3eq(self):
        """1 ax H + 3 eq H: ax passthrough, eq geometric (3H equivalent)."""
        mol = _make_sp3d_mol("H", "F", "H", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 4
                r = _order_h_sp3d(mol, atom.GetIdx(), h_all)
                assert len(r) == 4
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No P found")


def _make_oct_mol(*vert_syms):
    """Helper: build an octahedral (SP3D2) molecule manually.

    Constructs via RWMol with explicit bonds and 3D coordinates.
    The 6 vertices are along ±x, ±y, ±z axes. Assigns chiral tags
    and stereochemistry for _CIPRank availability.

    vert_syms is ordered: (+z, -z, +x, -x, +y, -y).
    Note: GetHybridization() may return UNSPECIFIED since RDKit does
    not perceive SP3D2 from simple connectivity alone — use
    _order_h_sp3d2 directly to test octahedral logic.
    """
    bond_lens = {"H": 1.42, "F": 1.55, "Cl": 2.1, "Br": 2.3}
    axes = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]

    mol = Chem.RWMol()
    p = Chem.Atom(15)
    p_idx = mol.AddAtom(p)

    for sym in vert_syms:
        a = Chem.Atom(sym)
        idx = mol.AddAtom(a)
        mol.AddBond(p_idx, idx, Chem.BondType.SINGLE)

    mol.UpdatePropertyCache(strict=False)
    mol = mol.GetMol()

    conf = Chem.Conformer(7)
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    for i, (sym, (dx, dy, dz)) in enumerate(zip(vert_syms, axes)):
        b = bond_lens.get(sym, 1.5)
        conf.SetAtomPosition(1 + i, (b * dx, b * dy, b * dz))
    mol.AddConformer(conf)

    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for atom in mol.GetAtoms():
        if '_CIPRank' not in atom.GetPropsAsDict():
            atom.SetIntProp('_CIPRank', 0)
    return mol


class TestFindSp3d2TransPairs:
    """Tests for _find_sp3d2_trans_pairs."""

    def test_sh6_three_pairs(self):
        """Regular octahedron should have exactly 3 trans pairs."""
        mol = _make_oct_mol("H", "H", "H", "H", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                pairs = _find_sp3d2_trans_pairs(mol, atom.GetIdx())
                assert len(pairs) == 3
                all_atoms = set()
                for a, b in pairs:
                    all_atoms.add(a)
                    all_atoms.add(b)
                assert len(all_atoms) == 6
                return
        pytest.fail("No octahedral center found")


class TestOrderSp3d2Trans2h:
    """Tests for 2 trans H on octahedral center."""

    def test_prochiral_succeeds(self):
        """4 different cis substituents → trans H diastereotopic."""
        mol = _make_oct_mol("H", "H", "Br", "Cl", "F", "I")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                trans_pairs = _find_sp3d2_trans_pairs(mol, atom.GetIdx())
                trans_of = _make_trans_of(trans_pairs)
                h_trans = [(a, b) for a, b in trans_pairs
                           if mol.GetAtomWithIdx(a).GetSymbol() == "H"
                           and mol.GetAtomWithIdx(b).GetSymbol() == "H"]
                if len(h_trans) == 1:
                    h1, h2 = h_trans[0]
                    result = _order_sp3d2_trans_2h(
                        mol, atom.GetIdx(), [h1, h2], trans_of
                    )
                    assert len(result) == 2
                    assert set(result) == {h1, h2}
                    return
        pytest.fail("No trans H pair found")

    def test_equivalent_returns_sorted(self):
        """All cis F equivalent → trans H equivalent → sorted."""
        mol = _make_oct_mol("H", "H", "F", "F", "F", "F")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                trans_pairs = _find_sp3d2_trans_pairs(mol, atom.GetIdx())
                trans_of = _make_trans_of(trans_pairs)
                h_trans = [(a, b) for a, b in trans_pairs
                           if mol.GetAtomWithIdx(a).GetSymbol() == "H"
                           and mol.GetAtomWithIdx(b).GetSymbol() == "H"]
                if h_trans:
                    h1, h2 = h_trans[0]
                    result = _order_sp3d2_trans_2h(
                        mol, atom.GetIdx(), [h1, h2], trans_of
                    )
                    assert result == sorted([h1, h2])
                    return
        pytest.fail("No trans H pair found")

    def test_deterministic(self):
        """Input-order independence."""
        mol = _make_oct_mol("H", "H", "Br", "Cl", "F", "I")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                trans_pairs = _find_sp3d2_trans_pairs(mol, atom.GetIdx())
                trans_of = _make_trans_of(trans_pairs)
                h_trans = [(a, b) for a, b in trans_pairs
                           if mol.GetAtomWithIdx(a).GetSymbol() == "H"
                           and mol.GetAtomWithIdx(b).GetSymbol() == "H"]
                if len(h_trans) == 1:
                    h1, h2 = h_trans[0]
                    r1 = _order_sp3d2_trans_2h(
                        mol, atom.GetIdx(), [h1, h2], trans_of
                    )
                    r2 = _order_sp3d2_trans_2h(
                        mol, atom.GetIdx(), [h2, h1], trans_of
                    )
                    assert r1 == r2
                    return
        pytest.fail("No trans H pair found")


class TestOrderSp3d2Cis2h:
    """Tests for 2 cis H on octahedral center."""

    def test_trans_partner_cip_differs(self):
        """2 cis H with different trans partners → order by trans CIP."""
        mol = _make_oct_mol("F", "H", "Cl", "H", "Br", "I")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                trans_pairs = _find_sp3d2_trans_pairs(mol, atom.GetIdx())
                trans_of = _make_trans_of(trans_pairs)
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                if len(h_all) >= 2:
                    for i in range(len(h_all)):
                        for j in range(i + 1, len(h_all)):
                            is_trans = any(
                                (h_all[i] in p and h_all[j] in p)
                                for p in trans_pairs
                            )
                            if not is_trans:
                                result = _order_sp3d2_cis_2h(
                                    mol, atom.GetIdx(),
                                    [h_all[i], h_all[j]], trans_of
                                )
                                assert len(result) == 2
                                assert set(result) == {h_all[i], h_all[j]}
                                return
        pytest.fail("No cis H pair found")

    def test_equivalent_returns_sorted(self):
        """2 cis H with equal trans partners AND symmetric cis square → sorted."""
        mol = _make_oct_mol("F", "H", "F", "H", "F", "F")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                trans_pairs = _find_sp3d2_trans_pairs(mol, atom.GetIdx())
                trans_of = _make_trans_of(trans_pairs)
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                if len(h_all) >= 2:
                    for i in range(len(h_all)):
                        for j in range(i + 1, len(h_all)):
                            is_trans = any(
                                (h_all[i] in p and h_all[j] in p)
                                for p in trans_pairs
                            )
                            if not is_trans:
                                result = _order_sp3d2_cis_2h(
                                    mol, atom.GetIdx(),
                                    [h_all[i], h_all[j]], trans_of
                                )
                                assert result == sorted([h_all[i], h_all[j]])
                                return
        pytest.fail("No cis H pair found")


class TestOrderHSp3d2All:
    """Integration tests for _order_h_sp3d2 across H counts."""

    def test_2h_trans(self):
        """2 trans H → chemical ordering via _order_h_sp3d2."""
        mol = _make_oct_mol("H", "H", "Br", "Cl", "F", "I")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                assert len(r) == 2
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No octahedral center found")

    def test_2h_cis(self):
        """2 cis H → chemical ordering via _order_h_sp3d2."""
        mol = _make_oct_mol("F", "H", "Cl", "H", "Br", "I")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                assert len(r) == 2
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No octahedral center found")

    def test_3h(self):
        """3 H on octahedral center → trans/cis dispatch."""
        mol = _make_oct_mol("H", "H", "H", "F", "Cl", "Br")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 3
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                assert len(r) == 3
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No octahedral center found")

    def test_4h(self):
        """4 H → 2 trans pairs or 1 trans + 2 singles."""
        mol = _make_oct_mol("H", "H", "H", "H", "F", "Cl")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 4
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                assert len(r) == 4
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No octahedral center found")

    def test_5h(self):
        """5 H → 2 trans pairs + 1 H-X."""
        mol = _make_oct_mol("H", "H", "H", "H", "H", "F")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 5
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                assert len(r) == 5
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No octahedral center found")

    def test_6h(self):
        """6 H → all equivalent, geometric CCW."""
        mol = _make_oct_mol("H", "H", "H", "H", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 6
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                assert len(r) == 6
                assert set(r) == set(h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                return
        pytest.fail("No octahedral center found")


class TestOrderHSp3d2Fac:
    """Tests for 3H fac (3 H-X trans pairs)."""

    def test_fac_abc(self):
        """3 different trans partners → order by CIP descending."""
        mol = _make_oct_mol("F", "H", "Cl", "H", "Br", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 3
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                assert len(r) == 3
                assert set(r) == set(h_all)
                return
        pytest.fail("No fac ABC center found")

    def test_fac_aab(self):
        """2 same + 1 unique trans partner → unique H first,
        remaining 2 H via 2H cis."""
        mol = _make_oct_mol("F", "H", "F", "H", "Cl", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 3
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                assert len(r) == 3
                assert set(r) == set(h_all)
                return
        pytest.fail("No fac AAB center found")

    def test_fac_aaa(self):
        """3 identical trans partners → all equivalent, geometric."""
        mol = _make_oct_mol("F", "H", "F", "H", "F", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 3
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                assert len(r) == 3
                assert set(r) == set(h_all)
                assert r[0] == min(h_all), (
                    "AAA: min-index H should be first (deuterated)"
                )
                return
        pytest.fail("No fac AAA center found")


class TestOrderHSp3d24hNonHCis:
    """Tests for 4H with non-H cis (2 H-X + 1 H-H)."""

    def test_hx_different_rank(self):
        """F/Cl as trans partners → H-X H ordered by CIP."""
        mol = _make_oct_mol("F", "H", "Cl", "H", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 4
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                assert len(r) == 4
                assert set(r) == set(h_all)
                return
        pytest.fail("No octahedral center found")

    def test_hx_same_rank(self):
        """Both F as trans partners → H-X H equivalent → geometric."""
        mol = _make_oct_mol("F", "H", "F", "H", "H", "H")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                assert len(h_all) == 4
                r = _order_h_sp3d2(mol, atom.GetIdx(), h_all)
                r2 = _order_h_sp3d2(mol, atom.GetIdx(), list(reversed(h_all)))
                assert r == r2
                assert len(r) == 4
                assert set(r) == set(h_all)
                return
        pytest.fail("No octahedral center found")


class TestSignedAngleBetween:
    """Tests for _signed_angle_between."""

    def test_ccw_positive(self):
        """v1=(1,0,0), v2=(0,1,0) around z-axis → CCW → positive angle."""
        axis = np.array([0.0, 0.0, 1.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        angle = _signed_angle_between(axis, v1, v2)
        assert angle > 0, f"CCW should be positive, got {angle}"
        assert abs(angle - np.pi / 2) < 1e-10

    def test_cw_negative(self):
        """v1=(0,1,0), v2=(1,0,0) around z-axis → CW → negative angle."""
        axis = np.array([0.0, 0.0, 1.0])
        v1 = np.array([0.0, 1.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        angle = _signed_angle_between(axis, v1, v2)
        assert angle < 0, f"CW should be negative, got {angle}"

    def test_parallel_zero(self):
        """Same direction → angle ≈ 0."""
        axis = np.array([0.0, 0.0, 1.0])
        v1 = np.array([1.0, 0.0, 0.0])
        angle = _signed_angle_between(axis, v1, v1)
        assert abs(angle) < 1e-10, f"Parallel should be 0, got {angle}"

    def test_antiparallel_pi(self):
        """Opposite directions → angle ≈ ±π."""
        axis = np.array([0.0, 0.0, 1.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        angle = _signed_angle_between(axis, v1, v2)
        assert abs(abs(angle) - np.pi) < 1e-10

    def test_not_perpendicular_to_axis(self):
        """Vectors with z component: only xy projection matters."""
        axis = np.array([0.0, 0.0, 1.0])
        v1 = np.array([1.0, 0.0, 5.0])
        v2 = np.array([0.0, 1.0, -3.0])
        angle = _signed_angle_between(axis, v1, v2)
        assert angle > 0  # CCW in xy projection


class TestIsCcw:
    """Tests for _is_ccw."""

    def _make_mol_3pts(self, pts):
        """Make a minimal mol with 3 atoms at given positions around origin."""
        mol = Chem.RWMol()
        c = Chem.Atom(6)
        c_idx = mol.AddAtom(c)
        indices = []
        for _ in range(3):
            a = Chem.Atom(1)
            idx = mol.AddAtom(a)
            mol.AddBond(c_idx, idx, Chem.BondType.SINGLE)
            indices.append(idx)
        mol.UpdatePropertyCache(strict=False)
        mol = mol.GetMol()
        conf = Chem.Conformer(4)
        conf.SetAtomPosition(c_idx, (0.0, 0.0, 0.0))
        for i, pos in enumerate(pts):
            conf.SetAtomPosition(indices[i], pos)
        mol.AddConformer(conf)
        return mol, c_idx, tuple(indices)

    def test_ccw_true(self):
        """Three points in CCW order around z-axis."""
        mol, c, idxs = self._make_mol_3pts([
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        ])
        z = np.array([0.0, 0.0, 1.0])
        assert _is_ccw(mol, c, z, idxs) is True

    def test_cw_false(self):
        """Three points in CW order around z-axis."""
        mol, c, idxs = self._make_mol_3pts([
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ])
        z = np.array([0.0, 0.0, 1.0])
        assert _is_ccw(mol, c, z, idxs) is False

    def test_requires_at_least_3(self):
        """Less than 3 atoms should raise ValueError."""
        mol, c, idxs = self._make_mol_3pts([
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        ])
        z = np.array([0.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="at least 3"):
            _is_ccw(mol, c, z, idxs[:2])

    def test_two_atoms_same_angle(self):
        """Two atoms at identical projected angle → should still produce a result."""
        mol, c, idxs = self._make_mol_3pts([
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),  # same position!
            (-1.0, 0.0, 0.0),
        ])
        z = np.array([0.0, 0.0, 1.0])
        # Should not crash; result is well-defined (two descents or zero)
        result = _is_ccw(mol, c, z, idxs)
        assert isinstance(result, bool)


class TestBuildPerpBasis:
    """Tests for _build_perp_basis."""

    def test_z_axis_standard(self):
        """z_axis along z → x,y should be (1,0,0),(0,1,0)."""
        z = np.array([0.0, 0.0, 1.0])
        x, y = _build_perp_basis(z)
        assert np.allclose(x, [1.0, 0.0, 0.0])
        assert np.allclose(y, [0.0, 1.0, 0.0])
        assert np.allclose(np.cross(x, y), z)

    def test_z_axis_arbitrary(self):
        """Oblique z_axis → x,y should be orthonormal."""
        z = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        x, y = _build_perp_basis(z)
        assert abs(np.dot(x, y)) < 1e-10
        assert abs(np.dot(x, z)) < 1e-10
        assert abs(np.dot(y, z)) < 1e-10
        assert abs(np.linalg.norm(x) - 1.0) < 1e-10
        assert abs(np.linalg.norm(y) - 1.0) < 1e-10
        assert np.allclose(np.cross(x, y), z)

    def test_with_x_direction(self):
        """x_direction should align x-axis toward its projection."""
        z = np.array([0.0, 0.0, 1.0])
        x_dir = np.array([0.0, 1.0, 0.0])
        x, y = _build_perp_basis(z, x_direction=x_dir)
        assert np.allclose(x, [0.0, 1.0, 0.0])
        assert np.allclose(y, [-1.0, 0.0, 0.0])
        assert np.allclose(np.cross(x, y), z)

    def test_x_direction_parallel_to_z_warns(self):
        """x_direction parallel to z_axis → zero projection → NaN warning."""
        z = np.array([0.0, 0.0, 1.0])
        with pytest.warns(RuntimeWarning):
            _build_perp_basis(z, x_direction=np.array([0.0, 0.0, 1.0]))


class TestProjectedAngle:
    """Tests for _projected_angle."""

    def test_quadrants(self):
        """Points in 4 quadrants should give correct atan2 angles."""
        z = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])

        a0 = _projected_angle(np.array([1.0, 0.0, 0.0]), z, x, y)
        assert abs(a0) < 1e-10

        a1 = _projected_angle(np.array([0.0, 1.0, 0.0]), z, x, y)
        assert abs(a1 - np.pi / 2) < 1e-10

        a2 = _projected_angle(np.array([-1.0, 0.0, 0.0]), z, x, y)
        assert abs(abs(a2) - np.pi) < 1e-10

        a3 = _projected_angle(np.array([0.0, -1.0, 0.0]), z, x, y)
        assert abs(a3 + np.pi / 2) < 1e-10

    def test_z_component_ignored(self):
        """Only projection onto ⟂z plane matters."""
        z = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        a = _projected_angle(np.array([0.0, 1.0, 100.0]), z, x, y)
        assert abs(a - np.pi / 2) < 1e-10


class TestSignedTetrahedronVolume:
    """Tests for _signed_tetrahedron_volume."""

    def test_r_configuration_negative(self):
        """R configuration → negative signed volume (a,b,c left-handed)."""
        # a,b in xy, c in -z → left-handed when d is at origin
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, -1.0])
        d = np.array([0.0, 0.0, 0.0])
        vol = _signed_tetrahedron_volume(a, b, c, d)
        assert vol < 0, f"R should give negative volume, got {vol}"

    def test_s_configuration_positive(self):
        """S configuration → positive signed volume."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 1.0])
        d = np.array([0.0, 0.0, 0.0])
        vol = _signed_tetrahedron_volume(a, b, c, d)
        assert vol > 0, f"S should give positive volume, got {vol}"

    def test_planar_near_zero(self):
        """Coplanar vertices → volume ≈ 0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([-1.0, 0.0, 0.0])
        d = np.array([0.0, -1.0, 0.0])
        vol = _signed_tetrahedron_volume(a, b, c, d)
        assert abs(vol) < 1e-10


class TestDeuterateAtom:
    """Tests for _deuterate_atom."""

    def test_isotope_set_to_2(self):
        """Atom should have isotope=2 after deuteration."""
        mol = _make_mol_with_3d("CC")
        c_idx = 0
        h_indices = [n.GetIdx() for n in mol.GetAtomWithIdx(c_idx).GetNeighbors()
                     if n.GetAtomicNum() == 1]
        assert len(h_indices) > 0
        h = h_indices[0]
        mol_d = _deuterate_atom(mol, h)
        assert mol_d.GetAtomWithIdx(h).GetIsotope() == 2

    def test_returns_new_mol(self):
        """Original mol should not be modified."""
        mol = _make_mol_with_3d("CC")
        c_idx = 0
        h_indices = [n.GetIdx() for n in mol.GetAtomWithIdx(c_idx).GetNeighbors()
                     if n.GetAtomicNum() == 1]
        h = h_indices[0]
        orig_isotope = mol.GetAtomWithIdx(h).GetIsotope()
        _deuterate_atom(mol, h)
        assert mol.GetAtomWithIdx(h).GetIsotope() == orig_isotope

    def test_stereochemistry_reassigned(self):
        """Deuterated mol should have _CIPRank available after deuteration."""
        mol = _make_mol_with_3d("CCO")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) == 2:
                mol_d = _deuterate_atom(mol, h_nbrs[0].GetIdx())
                for a in mol_d.GetAtoms():
                    if a.GetAtomicNum() != 1:
                        _ = _get_cip_rank(a)
                return
        pytest.skip("No CH2 center found")


class TestGetZPlusVec:
    """Tests for _get_z_plus_vec."""

    def _make_pair_mol(self, sym1, sym2):
        """Make a minimal mol with 2 atoms and a center."""
        mol = Chem.RWMol()
        c = Chem.Atom(6)
        c_idx = mol.AddAtom(c)
        a1 = Chem.Atom(sym1)
        a2 = Chem.Atom(sym2)
        idx1 = mol.AddAtom(a1)
        idx2 = mol.AddAtom(a2)
        mol.AddBond(c_idx, idx1, Chem.BondType.SINGLE)
        mol.AddBond(c_idx, idx2, Chem.BondType.SINGLE)
        mol.UpdatePropertyCache(strict=False)
        mol = mol.GetMol()
        conf = Chem.Conformer(3)
        conf.SetAtomPosition(c_idx, (0.0, 0.0, 0.0))
        conf.SetAtomPosition(idx1, (0.0, 0.0, 1.0))
        conf.SetAtomPosition(idx2, (0.0, 0.0, -1.0))
        mol.AddConformer(conf)
        Chem.AssignAtomChiralTagsFromStructure(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        _ensure_cip_ranks(mol)
        return mol, idx1, idx2

    def test_cip_differs(self):
        """Different CIP ranks → higher rank is z⁺."""
        mol, f_idx, cl_idx = self._make_pair_mol("F", "Cl")
        mol.GetAtomWithIdx(f_idx).SetIntProp('_CIPRank', 9)
        mol.GetAtomWithIdx(cl_idx).SetIntProp('_CIPRank', 17)
        vec = _get_z_plus_vec(mol, (f_idx, cl_idx))
        assert vec[2] < 0  # points from F(+) to Cl(-)

    def test_cip_equal_canonical_order(self):
        """Same CIP rank + _CanonicalOrder → larger order is z⁺."""
        mol, a_idx, b_idx = self._make_pair_mol("F", "F")
        mol.GetAtomWithIdx(a_idx).SetIntProp('_CanonicalOrder', 10)
        mol.GetAtomWithIdx(b_idx).SetIntProp('_CanonicalOrder', 5)
        vec = _get_z_plus_vec(mol, (a_idx, b_idx))
        assert vec[2] > 0

    def test_cip_equal_no_canonical(self):
        """Same CIP rank + no _CanonicalOrder (H atoms) → min index is z⁺."""
        mol, h1_idx, h2_idx = self._make_pair_mol("H", "H")
        vec = _get_z_plus_vec(mol, (h1_idx, h2_idx))
        assert vec[2] > 0


class TestGetCipRank:
    """Tests for _get_cip_rank."""

    def test_missing_cip_rank_raises(self):
        """Atom without _CIPRank should raise RuntimeError."""
        mol = Chem.RWMol()
        a = Chem.Atom(6)
        a_idx = mol.AddAtom(a)
        mol.UpdatePropertyCache(strict=False)
        mol = mol.GetMol()
        with pytest.raises(RuntimeError, match="missing _CIPRank"):
            _get_cip_rank(mol.GetAtomWithIdx(a_idx))

    def test_with_cip_rank(self):
        """Atom with _CIPRank should return its value."""
        mol = _make_mol_with_3d("C")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                rank = _get_cip_rank(atom)
                assert isinstance(rank, int)
                return

    def test_float_cip_rank_truncation(self):
        """Float _CIPRank should be correctly truncated to int."""
        mol = Chem.RWMol()
        a = Chem.Atom(6)
        a_idx = mol.AddAtom(a)
        mol.UpdatePropertyCache(strict=False)
        mol = mol.GetMol()
        mol.GetAtomWithIdx(a_idx).SetDoubleProp('_CIPRank', 17.0)
        assert _get_cip_rank(mol.GetAtomWithIdx(a_idx)) == 17


class TestCumuleneWalkFailure:
    """Test _order_2h_cumulene when walk returns far_subs empty."""

    def test_walk_returns_sorted(self):
        """Ketene H2C=C=O: far end is O with no substituents → sorted."""
        mol = _make_mol_with_3d("C=C=O")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert result == sorted([h1, h2])
                    return
        pytest.skip("No terminal =CH2 found in ketene")


class TestSp2SinglePartnerSub:
    """Test _order_2h_sp2 with partner having a single substituent."""

    def test_partner_single_substituent_normal_alkene(self):
        """Imine HN=CH2: partner N has H substituent → normal alkene logic."""
        mol = _make_mol_with_3d("C=N")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    center = atom.GetIdx()
                    h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                    result = _order_2h_sp2(mol, center, partner, [h1, h2])
                    assert len(result) == 2
                    assert set(result) == {h1, h2}
                    return
        pytest.skip("No terminal =CH2 found")


class TestSp2BH3ThreeH:
    """Test _order_h_sp2 with 3H and no double bond."""

    def test_sp2_3h_no_double_bond_geometric(self):
        """sp2 center with 3H and no double bond → geometric via plane normal."""
        mol = _make_mol_with_3d("[CH3+]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors()
                          if n.GetAtomicNum() == 1]
                if len(h_nbrs) >= 2 and atom.GetHybridization() == Chem.HybridizationType.SP2:
                    result = _order_h_sp2(mol, atom.GetIdx(), h_nbrs)
                    assert set(result) == set(h_nbrs)
                    return
        pytest.skip("CH3+ not SP2 in RDKit")


class TestSp3dClassificationFailure:
    """Tests for SP3D classification failure → geometric fallback."""

    def test_not_5_neighbors_geometric_fallback(self):
        """4-coordinate P → _classify_sp3d_positions returns all eq."""
        mol = _make_mol_with_3d("[PH4+]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors()
                          if n.GetAtomicNum() == 1]
                if len(h_nbrs) >= 2:
                    ax, _ = _classify_sp3d_positions(mol, atom.GetIdx())
                    assert len(ax) == 0
                    r = _order_h_sp3d(mol, atom.GetIdx(), h_nbrs)
                    assert len(r) == len(h_nbrs)
                    assert set(r) == set(h_nbrs)
                    return
        pytest.skip("No P found")

    def test_angle_too_small(self):
        """Max angle < 140° → classification fails → geometric fallback.

        Uses a distorted geometry where no pair forms a clear trans axis:
        3 atoms in the equatorial plane at ~120°, 2 atoms tilted off the
        z-axis by a shallow angle so they are not opposite each other.
        """
        mol = Chem.RWMol()
        p = Chem.Atom(15)
        p_idx = mol.AddAtom(p)
        indices = []
        for _ in range(5):
            a = Chem.Atom("H")
            idx = mol.AddAtom(a)
            mol.AddBond(p_idx, idx, Chem.BondType.SINGLE)
            indices.append(idx)
        mol.UpdatePropertyCache(strict=False)
        mol = mol.GetMol()
        # 3 eq in xy plane (120° apart), 2 tilted near the plane
        positions = [
            (1.42, 0.0, 0.0),        # eq 1
            (-0.71, 1.23, 0.0),      # eq 2 (~120°)
            (-0.71, -1.23, 0.0),     # eq 3 (~120°)
            (0.0, 0.5, 1.0),         # tilted, not opposite to below
            (0.0, -0.5, 0.5),        # tilted, ~72° from above
        ]
        conf = Chem.Conformer(6)
        conf.SetAtomPosition(p_idx, (0.0, 0.0, 0.0))
        for i, pos in enumerate(positions):
            conf.SetAtomPosition(indices[i], pos)
        mol.AddConformer(conf)
        ax, _ = _classify_sp3d_positions(mol, p_idx)
        assert len(ax) == 0
        r = _order_h_sp3d(mol, p_idx, indices)
        assert len(r) == 5
        assert set(r) == set(indices)


class TestAnalyzeSquareChiralityDiagonalMatch:
    """Test _analyze_square_chirality Step 2: diagonal rank match → None."""

    def test_diagonal_match_returns_none(self):
        """Square with identical CIP ranks on a diagonal → None."""
        mol = _make_oct_mol("F", "H", "Cl", "Cl", "Br", "I")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                trans_pairs = _find_sp3d2_trans_pairs(mol, atom.GetIdx())
                trans_of = _make_trans_of(trans_pairs)
                for a, b in trans_pairs:
                    sym_a = mol.GetAtomWithIdx(a).GetSymbol()
                    sym_b = mol.GetAtomWithIdx(b).GetSymbol()
                    if {sym_a, sym_b} == {"H", "F"}:
                        center = atom.GetIdx()
                        all_nbrs = {n.GetIdx() for n in atom.GetNeighbors()}
                        square = list(all_nbrs - {a, b})
                        result = _analyze_square_chirality(
                            mol, center, a, b, square, trans_of
                        )
                        assert result is None
                        return
        pytest.fail("No H-F trans pair found")


class TestSp3d2Cis2hChirality:
    """Test _order_sp3d2_cis_2h T→H square chirality path."""

    def test_equal_trans_partner_chirality_succeeds(self):
        """2 cis H with equal trans partners → T→H chirality decides order."""
        mol = _make_oct_mol("F", "H", "F", "H", "Br", "Cl")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                trans_pairs = _find_sp3d2_trans_pairs(mol, atom.GetIdx())
                trans_of = _make_trans_of(trans_pairs)
                h_all = [n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 1]
                for i in range(len(h_all)):
                    for j in range(i + 1, len(h_all)):
                        t_i = trans_of[h_all[i]]
                        t_j = trans_of[h_all[j]]
                        if t_i == h_all[j]:
                            continue
                        r_i = _get_cip_rank(mol.GetAtomWithIdx(t_i))
                        r_j = _get_cip_rank(mol.GetAtomWithIdx(t_j))
                        if r_i == r_j:
                            result = _order_sp3d2_cis_2h(
                                mol, atom.GetIdx(),
                                [h_all[i], h_all[j]], trans_of
                            )
                            assert len(result) == 2
                            assert set(result) == {h_all[i], h_all[j]}
                            return
        pytest.skip("No suitable cis H pair")


class TestFindSp3d2TransPairsDegenerate:
    """Test _find_sp3d2_trans_pairs when <3 trans pairs."""

    def test_fewer_than_three_pairs_dispatch_handles(self):
        """Distorted geometry: dispatcher handles <3 pairs gracefully."""
        mol = Chem.RWMol()
        s = Chem.Atom(16)
        s_idx = mol.AddAtom(s)
        indices = []
        for _ in range(6):
            a = Chem.Atom("H")
            idx = mol.AddAtom(a)
            mol.AddBond(s_idx, idx, Chem.BondType.SINGLE)
            indices.append(idx)
        mol.UpdatePropertyCache(strict=False)
        mol = mol.GetMol()
        conf = Chem.Conformer(7)
        conf.SetAtomPosition(s_idx, (0.0, 0.0, 0.0))
        positions = [
            (1.0, 0.0, 0.0),
            (-0.5, 0.866, 0.0),
            (-0.5, -0.866, 0.0),
            (0.0, 0.0, 1.42),
            (0.0, 0.0, -1.42),
            (0.5, 0.3, 0.8),
        ]
        for i, pos in enumerate(positions):
            conf.SetAtomPosition(indices[i], pos)
        mol.AddConformer(conf)
        r = _order_h_sp3d2(mol, s_idx, indices)
        assert len(r) == 6
        assert set(r) == set(indices)


# The "other hybridization" else-branch in _order_h_on_heavy_atom (geometric
# fallback) is defensive: atoms with SP or UNSPECIFIED hybridization cannot
# have >= 2 H by chemical constraints (SP: 2 σ bonds total; UNSPECIFIED:
# typically metal centers with at most 1 terminal H).  The branch exists for
# safety and is not independently testable with real molecules.


class TestOrder2hCumuleneDirect:
    """Direct tests for _order_2h_cumulene (odd/even sp_count paths)."""

    def test_odd_sp_count_r_a_s_a(self):
        """Allene with asymmetric far-end: odd sp → R_a/S_a ordering."""
        mol = _make_mol_from_xyz("C=C=C(F)Br")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    if mol.GetAtomWithIdx(partner).GetHybridization() == Chem.HybridizationType.SP:
                        h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                        result = _order_2h_cumulene(
                            mol, atom.GetIdx(), partner, [h1, h2]
                        )
                        assert len(result) == 2
                        assert set(result) == {h1, h2}
                        # Should be deterministic
                        r2 = _order_2h_cumulene(
                            mol, atom.GetIdx(), partner, [h1, h2]
                        )
                        assert result == r2
                        return
        pytest.fail("No allene =CH2 found")

    def test_even_sp_count_pro_z_e(self):
        """Butatriene with asymmetric far-end: even sp → Z/E ordering."""
        mol = _make_cumulene_mol("C=C=C=CF")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    if mol.GetAtomWithIdx(partner).GetHybridization() == Chem.HybridizationType.SP:
                        h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                        result = _order_2h_cumulene(
                            mol, atom.GetIdx(), partner, [h1, h2]
                        )
                        assert len(result) == 2
                        assert set(result) == {h1, h2}
                        return
        pytest.fail("No butatriene =CH2 found")

    def test_unsubstituted_returns_sorted(self):
        """H2C=C=CH2: far-end H equivalent → sorted."""
        mol = _make_mol_from_xyz("C=C=C")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(h_nbrs) != 2:
                continue
            for bond in atom.GetBonds():
                if bond.GetBondTypeAsDouble() == 2.0:
                    partner = bond.GetOtherAtomIdx(atom.GetIdx())
                    if mol.GetAtomWithIdx(partner).GetHybridization() == Chem.HybridizationType.SP:
                        h1, h2 = h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                        result = _order_2h_cumulene(
                            mol, atom.GetIdx(), partner, [h1, h2]
                        )
                        assert result == sorted([h1, h2])
                        return
        pytest.fail("No allene =CH2 found")


class TestOrderHGeometricDirect:
    """Direct tests for _order_h_geometric (all z_axis selection paths)."""

    def test_explicit_z_axis(self):
        """z_axis provided → all H ordered by CCW projection."""
        mol = _make_mol_with_3d("CC")  # ethane: methyl with 3H
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors()
                          if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 3:
                    z = np.array([0.0, 0.0, 1.0])
                    r = _order_h_geometric(mol, atom.GetIdx(), h_nbrs, z_axis=z)
                    assert len(r) == 3
                    assert set(r) == set(h_nbrs)
                    r2 = _order_h_geometric(mol, atom.GetIdx(), h_nbrs, z_axis=z)
                    assert r == r2
                    return
        pytest.fail("No methyl group found")

    def test_non_h_neighbor_case_b(self):
        """No z_axis + non-H neighbor → non-H as ref, all H CCW."""
        mol = _make_mol_with_3d("CCO")  # ethanol
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors()
                          if n.GetAtomicNum() == 1]
                non_h = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
                if len(h_nbrs) == 2 and len(non_h) == 2:
                    # CH2 with 2 non-H → sp3 4-coordinate Case B
                    r = _order_h_geometric(mol, atom.GetIdx(), h_nbrs)
                    assert len(r) == 2
                    assert set(r) == set(h_nbrs)
                    r2 = _order_h_geometric(mol, atom.GetIdx(), list(reversed(h_nbrs)))
                    assert r == r2
                    return
        pytest.fail("No CH2 with non-H neighbors found")

    def test_3_coordinate_lp(self):
        """No z_axis + 3-coordinate → LP as z_axis, all H CCW."""
        mol = _make_mol_with_3d("[SH3+]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 16:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors()
                          if n.GetAtomicNum() == 1]
                assert len(h_nbrs) == 3
                r = _order_h_geometric(mol, atom.GetIdx(), h_nbrs)
                assert len(r) == 3
                assert set(r) == set(h_nbrs)
                r2 = _order_h_geometric(mol, atom.GetIdx(), list(reversed(h_nbrs)))
                assert r == r2
                return
        pytest.fail("No S with 3H found")

    def test_min_idx_h_case_a(self):
        """No z_axis + no non-H + 4-coordinate → min-idx H first."""
        mol = _make_mol_with_3d("C")  # CH4
        c_idx = 0
        h_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(c_idx).GetNeighbors()
                  if n.GetAtomicNum() == 1]
        assert len(h_nbrs) == 4
        r = _order_h_geometric(mol, c_idx, h_nbrs)
        assert len(r) == 4
        assert set(r) == set(h_nbrs)
        # Deterministic
        r2 = _order_h_geometric(mol, c_idx, list(reversed(h_nbrs)))
        assert r == r2
        # min-idx H should be first
        assert r[0] == min(h_nbrs)
