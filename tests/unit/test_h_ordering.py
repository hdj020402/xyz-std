import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from xyz_std.h_ordering import (
    order_h_on_heavy_atom,
    _order_h_by_angle_projection,
    _try_order_2h_sp3,
)


def _make_mol_with_3d(smiles: str, seed: int = 42) -> Chem.Mol:
    """Helper: SMILES -> Mol with explicit H and 3D conformer."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
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
        result = order_h_on_heavy_atom(mol, o_idx, h_indices)
        assert len(result) == 2
        assert set(result) == set(h_indices)

    def test_zero_h(self):
        """Empty list should return empty."""
        mol = _make_mol_with_3d("C")
        result = order_h_on_heavy_atom(mol, 0, [])
        assert result == []

    def test_one_h(self):
        """Single H in list should return list of length 1."""
        mol = _make_mol_with_3d("C")
        c_idx = 0
        h_indices = [
            n.GetIdx() for n in mol.GetAtomWithIdx(c_idx).GetNeighbors()
            if n.GetAtomicNum() == 1
        ]
        result = order_h_on_heavy_atom(mol, c_idx, h_indices[:1])
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
                    result = order_h_on_heavy_atom(mol, atom.GetIdx(), h_nbrs)
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
        result1 = order_h_on_heavy_atom(mol, c2_idx, h_indices)
        result2 = order_h_on_heavy_atom(mol, c2_idx, list(reversed(h_indices)))
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
        result = order_h_on_heavy_atom(mol, c_idx, h_indices)
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
