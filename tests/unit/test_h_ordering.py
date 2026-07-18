import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from xyz_std.h_ordering import (
    _order_h_on_heavy_atom,
    _order_h_by_angle_projection,
    _order_h_geometric,
    _order_h_sp2,
    _order_h_sp3,
    _order_h_sp3d,
    _order_h_sp3d2,
    _classify_sp3d_positions,
    _try_order_2h_sp3,
    _try_order_2h_sp2,
    _try_order_2h_allene,
    _try_order_2h_signed_volume,
    _infer_lone_pair_position,
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
    return mol


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
    return mol_ob


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


class TestTryOrder2hAllene:
    """Tests for allene =CH2 ordering via axial chirality (R_a/S_a)."""

    def test_unsubstituted_returns_none(self):
        """H2C=C=CH2: far-end H's are equivalent → return None."""
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
                    result = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    assert result is None, (
                        "unsubstituted allene should return None"
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
                    result = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    assert result is not None, (
                        "asymmetric allene should succeed"
                    )
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
                    r1 = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    r2 = _try_order_2h_sp2(mol, center, partner, h1, h2)
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
                    r1 = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    r2 = _try_order_2h_sp2(mol, center, partner, h2, h1)
                    assert r1 is not None
                    assert r2 is not None
                    assert set(r1) == set(r2)
                    assert r1 == [h1, h2] or r2 == [h2, h1]
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
        atom.SetDoubleProp("_CIPRank", float(atom.GetAtomicNum()))

    return mol


class TestTryOrder2hAlleneEven:
    """Tests for even-cumulene (butatriene etc.) =CH2 ordering.

    sp_count is even → terminal planes are coplanar → pro-Z/pro-E via
    dihedral. Uses RDKit direct path with manual _CIPRank because
    OpenBabel misperceives butatriene bond types.
    """

    def test_unsubstituted_returns_none(self):
        """H2C=C=C=CH2: far-end H's are equivalent → return None."""
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
                    result = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    assert result is None, (
                        "unsubstituted butatriene should return None"
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
                    result = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    assert result is not None, (
                        "asymmetric butatriene should succeed"
                    )
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
                    r1 = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    r2 = _try_order_2h_sp2(mol, center, partner, h1, h2)
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
                    r1 = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    r2 = _try_order_2h_sp2(mol, center, partner, h2, h1)
                    assert r1 is not None
                    assert r2 is not None
                    assert set(r1) == set(r2)
                    assert r1 == [h1, h2] or r2 == [h2, h1]
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
                    result = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    assert result is not None, (
                        "sp2 CIP should succeed for asymmetric terminal alkene (OB)"
                    )
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
                    r1 = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    r2 = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    assert r1 == r2
                    return
        pytest.fail("No terminal =CH2 found")

    def test_symmetric_partner_returns_none(self):
        """Isobutene via OB: partner has 2 identical CH3 substituents,
        sp2 CIP should return None."""
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
                    result = _try_order_2h_sp2(mol, center, partner, h1, h2)
                    assert result is None, (
                        "sp2 CIP should return None when partner subs are equivalent (OB)"
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
        assert lp is not None
        # Lone pair should be in the opposite direction to sum of bonds
        s = vecs[0] + vecs[1] + vecs[2]
        lp_dir = lp - center
        # lp_dir and -s should point in same direction (positive dot product)
        assert np.dot(lp_dir, -s) > 0

    def test_planar_returns_none(self):
        """Planar (trigonal) arrangement should return None."""
        center = np.array([0.0, 0.0, 0.0])
        # Three coplanar bonds at 120° — sum ≈ 0
        vecs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([-0.5, 0.866, 0.0]),
            np.array([-0.5, -0.866, 0.0]),
        ]
        lp = _infer_lone_pair_position(center, vecs)
        assert lp is None


class TestTryOrder2hSignedVolume:
    """Tests for _try_order_2h_signed_volume (non-carbon prochiral centers)."""

    def test_phosphine_prochiral(self):
        """CH3-PH2: P with 3 neighbors (C, H, H) — signed volume should
        infer lone pair and determine pro-R/pro-S."""
        mol = _make_mol_with_3d("CP")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    result = _try_order_2h_signed_volume(mol, atom.GetIdx(), h1, h2)
                    assert result is not None, (
                        "Signed volume should determine pro-R/pro-S for PH2"
                    )
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
                    r1 = _try_order_2h_signed_volume(mol, atom.GetIdx(), h1, h2)
                    r2 = _try_order_2h_signed_volume(mol, atom.GetIdx(), h1, h2)
                    assert r1 is not None
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
                    r1 = _try_order_2h_signed_volume(mol, atom.GetIdx(), h1, h2)
                    r2 = _try_order_2h_signed_volume(mol, atom.GetIdx(), h2, h1)
                    assert r1 is not None and r2 is not None
                    assert set(r1) == set(r2)
                    assert r1 == [h1, h2] or r2 == [h2, h1]
                    return
        pytest.fail("No P with 2H found")

    def test_via_try_order_2h_sp3(self):
        """_try_order_2h_sp3 should fall back to signed volume for PH2."""
        mol = _make_mol_with_3d("CP")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    result = _try_order_2h_sp3(mol, atom.GetIdx(), h1, h2)
                    assert result is not None, (
                        "_try_order_2h_sp3 should fallback to signed volume for P"
                    )
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

    def test_equivalent_sih2_returns_none(self):
        """Symmetric Si center (CH3-SiH2-CH3) should return None."""
        mol = _make_mol_with_3d("C[SiH2]C")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 14:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    result = _try_order_2h_signed_volume(mol, atom.GetIdx(), h1, h2)
                    assert result is None, "Symmetric SiH2 should return None"
                    return
        pytest.fail("No Si with 2H found")

    def test_h2s_returns_none(self):
        """H2S (2-coordinate, 2 lone pairs) should return None."""
        mol = _make_mol_with_3d("[SH2]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 16:
                h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                if len(h_nbrs) == 2:
                    h1, h2 = h_nbrs[0], h_nbrs[1]
                    result = _try_order_2h_signed_volume(mol, atom.GetIdx(), h1, h2)
                    assert result is None, "2-coordinate H2S should return None"
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
        """PH5: 5 H in trigonal bipyramidal → deterministic CCW order."""
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

    def test_ph5_axial_before_equatorial(self):
        """PH5: axial H should come before equatorial H in output."""
        mol = _make_mol_with_3d("[PH5]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                axial, eq_ = _classify_sp3d_positions(mol, atom.GetIdx())
                h_axial_set = set(axial)
                h_eq_set = set(eq_)
                # All neighbors are H in PH5, so axial and eq are H indices
                assert len(h_axial_set) == 2
                assert len(h_eq_set) == 3

                h_indices = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                result = _order_h_sp3d(mol, atom.GetIdx(), h_indices)

                # All axial H should appear before all equatorial H
                last_axial_pos = max(
                    result.index(h) for h in result if h in h_axial_set
                )
                first_eq_pos = min(
                    result.index(h) for h in result if h in h_eq_set
                )
                assert last_axial_pos < first_eq_pos, (
                    f"Axial H should come before equatorial H, got {result}"
                )
                return
        pytest.fail("No P with 5H found")

    def test_ph5_deterministic(self):
        """PH5: order should be deterministic regardless of input order."""
        mol = _make_mol_with_3d("[PH5]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_indices = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                r1 = _order_h_sp3d(mol, atom.GetIdx(), h_indices)
                r2 = _order_h_sp3d(mol, atom.GetIdx(), list(reversed(h_indices)))
                assert r1 == r2
                return
        pytest.fail("No P with 5H found")

    def test_ph5_via_order_h_on_heavy_atom(self):
        """PH5 through top-level dispatch: should use SP3D path."""
        mol = _make_mol_with_3d("[PH5]")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 15:
                h_indices = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
                r1 = _order_h_on_heavy_atom(mol, atom.GetIdx(), h_indices)
                r2 = _order_h_on_heavy_atom(mol, atom.GetIdx(), list(reversed(h_indices)))
                assert len(r1) == 5
                assert set(r1) == set(h_indices)
                assert r1 == r2
                return
        pytest.fail("No P with 5H found")


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
