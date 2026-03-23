import os
import tempfile
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from xyz_std.io import xyz_to_rdkit_mol, xyz_to_symbols_coords, write_multi_xyz


def _mol_to_xyz_str(mol: Chem.Mol) -> str:
    """Helper: convert RDKit Mol with conformer to XYZ string."""
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), "test molecule"]
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        lines.append(f"{sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines) + "\n"


class TestXyzToRdkitMol:
    def test_methane(self):
        mol = Chem.MolFromSmiles("C")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        xyz_str = _mol_to_xyz_str(mol)

        result = xyz_to_rdkit_mol(xyz_str)
        assert result is not None
        assert result.GetNumAtoms() == 5  # C + 4H
        assert result.GetNumConformers() == 1

    def test_ethanol(self):
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        xyz_str = _mol_to_xyz_str(mol)

        result = xyz_to_rdkit_mol(xyz_str)
        assert result is not None
        assert result.GetNumAtoms() == 9  # 2C + O + 6H

    def test_preserves_atom_count(self):
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        xyz_str = _mol_to_xyz_str(mol)

        result = xyz_to_rdkit_mol(xyz_str)
        assert result.GetNumAtoms() == mol.GetNumAtoms()

    def test_has_3d_conformer(self):
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        xyz_str = _mol_to_xyz_str(mol)

        result = xyz_to_rdkit_mol(xyz_str)
        conf = result.GetConformer()
        pos = conf.GetAtomPosition(0)
        # Should have non-zero coordinates
        assert not (pos.x == 0.0 and pos.y == 0.0 and pos.z == 0.0)


class TestXyzToSymbolsCoords:
    def test_basic(self, tmp_path):
        xyz_content = "3\ntest\nC  0.0 0.0 0.0\nH  1.0 0.0 0.0\nH  0.0 1.0 0.0\n"
        xyz_file = tmp_path / "test.xyz"
        xyz_file.write_text(xyz_content)

        symbols, coords = xyz_to_symbols_coords(str(xyz_file))
        assert symbols == ["C", "H", "H"]
        assert coords.shape == (3, 3)
        np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(coords[1], [1.0, 0.0, 0.0])


class TestWriteMultiXyz:
    def test_two_frames(self, tmp_path):
        symbols = ["C", "H"]
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        energies = [-100.0, -99.5]
        output = tmp_path / "multi.xyz"

        write_multi_xyz(symbols, [coords1, coords2], energies, str(output))

        text = output.read_text()
        lines = text.strip().split("\n")
        # Frame 1: count + energy + 2 atoms = 4 lines
        # Frame 2: same
        assert len(lines) == 8
        assert lines[0].strip() == "2"
        assert "Energy:" in lines[1]
        assert lines[4].strip() == "2"

    def test_round_trip(self, tmp_path):
        symbols = ["O", "H", "H"]
        coords = [np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])]
        energies = [-76.123456]
        output = tmp_path / "water.xyz"

        write_multi_xyz(symbols, coords, energies, str(output))

        read_syms, read_coords = xyz_to_symbols_coords(str(output))
        assert read_syms == symbols
        np.testing.assert_allclose(read_coords, coords[0], atol=1e-5)
