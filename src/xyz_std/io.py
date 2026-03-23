import numpy as np
from openbabel import pybel
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def xyz_to_rdkit_mol(xyz_str: str) -> Chem.Mol:
    """
    Convert XYZ string to RDKit Mol with 3D coordinates and bond connectivity.

    Uses OpenBabel for bond perception from 3D geometry, then converts to RDKit Mol.
    Tries full sanitization first; falls back to skip-valence sanitization for
    molecules with unusual valence (e.g., hypervalent atoms).

    Args:
        xyz_str: XYZ format string (atom count, comment line, then coordinates)

    Returns:
        RDKit Mol with explicit H atoms and a 3D conformer
    """
    ob_mol = pybel.readstring("xyz", xyz_str)
    molblock = ob_mol.write("mol")

    mol = Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=True)
    if mol is not None:
        return mol

    mol = Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=False)
    if mol is None:
        raise ValueError("RDKit failed to parse the MOL block from OpenBabel.")

    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
    )
    return mol


def xyz_to_symbols_coords(xyz_path: str) -> tuple[list[str], np.ndarray]:
    """
    Parse a single-frame XYZ file into atom symbols and coordinates.

    Args:
        xyz_path: Path to the XYZ file

    Returns:
        Tuple of (atom_symbols, coordinates) where coordinates is shape (n_atoms, 3)
    """
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    symbols, coords = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return symbols, np.array(coords)


def write_multi_xyz(
    atom_symbols: list[str],
    coord_list: list[np.ndarray],
    energy_list: list[float],
    output_path: str
) -> None:
    """
    Write a multi-frame XYZ file with energy annotations.

    Each frame has:
      - Line 1: atom count
      - Line 2: "Energy: {value}"
      - Lines 3+: atom symbol and x, y, z coordinates

    Args:
        atom_symbols: Atom symbols (same for all frames)
        coord_list: List of coordinate arrays, each shape (n_atoms, 3)
        energy_list: Energy value for each frame
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        for coords, energy in zip(coord_list, energy_list):
            f.write(f"{len(atom_symbols)}\n")
            f.write(f"Energy: {energy:.8f}\n")
            for sym, (x, y, z) in zip(atom_symbols, coords):
                f.write(f"{sym:>2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")
