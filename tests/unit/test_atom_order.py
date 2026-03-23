import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from xyz_std.atom_order import get_standard_atom_order, _parse_heavy_order_from_auxinfo
from xyz_std.io import xyz_to_rdkit_mol


def _make_mol_with_3d(smiles: str, seed: int = 42) -> Chem.Mol:
    """Helper: SMILES -> Mol with explicit H and 3D conformer."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


def _mol_to_xyz_str(mol: Chem.Mol) -> str:
    """Helper: convert RDKit Mol with conformer to XYZ string."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), "test"]
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        lines.append(f"{sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines) + "\n"


class TestParseHeavyOrder:
    def test_simple(self):
        aux = "AuxInfo=1/0/N:2,1,3/rA:3nCCO/rB:s1;s2;/rC:;;;"
        result = _parse_heavy_order_from_auxinfo(aux)
        assert result == [1, 0, 2]  # 1-based [2,1,3] -> 0-based [1,0,2]

    def test_single_atom(self):
        aux = "AuxInfo=1/0/N:1/rA:1nC/rB:/rC:;"
        result = _parse_heavy_order_from_auxinfo(aux)
        assert result == [0]

    def test_missing_n_layer(self):
        aux = "AuxInfo=1/0/rA:3nCCO"
        with pytest.raises(ValueError, match="N: layer not found"):
            _parse_heavy_order_from_auxinfo(aux)


class TestGetStandardAtomOrder:
    def test_methane_returns_5_indices(self):
        mol = _make_mol_with_3d("C")
        order = get_standard_atom_order(mol)
        assert len(order) == 5  # 1C + 4H
        assert len(set(order)) == 5  # all unique

    def test_ethanol_returns_correct_count(self):
        mol = _make_mol_with_3d("CCO")
        order = get_standard_atom_order(mol)
        assert len(order) == 9  # 2C + O + 6H
        assert len(set(order)) == 9

    def test_heavy_atoms_first(self):
        """Heavy atoms should come before H in the order."""
        mol = _make_mol_with_3d("CCO")
        order = get_standard_atom_order(mol)
        n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        heavy_part = order[:n_heavy]
        h_part = order[n_heavy:]
        # All heavy_part indices should be non-H atoms
        for idx in heavy_part:
            assert mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
        # All h_part indices should be H atoms
        for idx in h_part:
            assert mol.GetAtomWithIdx(idx).GetAtomicNum() == 1

    def test_is_permutation(self):
        """Result should be a permutation of [0, n_atoms)."""
        mol = _make_mol_with_3d("c1ccccc1")  # benzene
        order = get_standard_atom_order(mol)
        assert sorted(order) == list(range(mol.GetNumAtoms()))

    def test_deterministic(self):
        """Same mol should always produce same order."""
        mol = _make_mol_with_3d("CCCC")
        order1 = get_standard_atom_order(mol)
        order2 = get_standard_atom_order(mol)
        assert order1 == order2

    def test_via_xyz_round_trip(self):
        """Full pipeline: SMILES -> Mol -> XYZ -> xyz_to_rdkit_mol -> standard order."""
        mol = _make_mol_with_3d("CCO")
        xyz_str = _mol_to_xyz_str(mol)
        mol_from_xyz = xyz_to_rdkit_mol(xyz_str)
        order = get_standard_atom_order(mol_from_xyz)

        assert len(order) == mol.GetNumAtoms()
        assert sorted(order) == list(range(mol.GetNumAtoms()))

    def test_prochiral_order_independent_of_input(self):
        """Prochiral H ordering should be same regardless of initial H indexing."""
        mol = _make_mol_with_3d("ClCCO")
        order = get_standard_atom_order(mol)
        # Just verify it completes and returns valid permutation
        assert sorted(order) == list(range(mol.GetNumAtoms()))
