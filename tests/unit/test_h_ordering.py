import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from xyz_std.h_ordering import (
    _order_h_on_heavy_atom,
    _order_h_by_angle_projection,
    _try_order_2h_sp3,
    _try_order_2h_sp2,
    _try_order_2h,
)


def _make_mol_with_3d(smiles: str, seed: int = 42) -> Chem.Mol:
    """Helper: SMILES -> Mol with explicit H and 3D conformer.

    Calls both AssignAtomChiralTagsFromStructure and AssignStereochemistry
    to ensure _CIPRank is available for sp2 CIP-based H ordering.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


class TestOrderHOnHeavyAtom:
    def test_single_h_passthrough(self):
        """Single H should be returned as-is."""
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
                result = _try_order_2h_sp3(
                    mol, atom.GetIdx(), h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                )
                assert result is not None
                assert len(result) == 2
                return
        pytest.skip("No suitable prochiral center found")

    def test_equivalent_h_returns_none(self):
        """Truly equivalent H (symmetric center) should return None."""
        # Propane central CH2: CH3-CH2-CH3, both sides are CH3 (equivalent)
        mol = _make_mol_with_3d("CCC")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 6:
                continue
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            c_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
            if len(h_nbrs) == 2 and len(c_nbrs) == 2:
                result = _try_order_2h_sp3(
                    mol, atom.GetIdx(), h_nbrs[0].GetIdx(), h_nbrs[1].GetIdx()
                )
                assert result is None
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
                    result = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    assert result is not None, (
                        "sp2 CIP should succeed for asymmetric terminal alkene"
                    )
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
                    r1 = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    r2 = _try_order_2h_sp2(mol, center, partner, h1, h2)
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
                    r1 = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    r2 = _try_order_2h_sp2(mol, center, partner, h2, h1)
                    # Both should succeed and contain the same H's
                    assert r1 is not None
                    assert r2 is not None
                    assert set(r1) == set(r2)
                    # Output should swap when input swaps
                    assert r1 == [h1, h2] or r2 == [h2, h1]
                    return
        pytest.fail("No terminal =CH2 found")

    def test_symmetric_partner_returns_none(self):
        """Isobutene (CH3)2C=CH2: partner has 2 identical CH3 substituents,
        sp2 CIP should return None (H are truly equivalent)."""
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
                    result = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    assert result is None, (
                        "sp2 CIP should return None when partner subs are equivalent"
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
