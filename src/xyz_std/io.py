import numpy as np
from openbabel import pybel
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdDetermineBonds

RDLogger.DisableLog('rdApp.*')


def _read_xyz_content(xyz: str) -> str:
    """Normalize xyz input: if it's a file path, read it; if it's content, return as-is."""
    if '\n' in xyz:
        return xyz
    with open(xyz, 'r') as f:
        return f.read()


def xyz_to_rdkit_mol(xyz_str: str, backend: str = "openbabel") -> Chem.Mol:
    """
    Convert XYZ string to RDKit Mol with 3D coordinates and bond connectivity.

    Args:
        xyz_str: XYZ format string (atom count, comment line, then coordinates)
        backend: Bond perception method. "openbabel" uses OpenBabel with relaxed
            sanitization; "rdkit" uses RDKit's rdDetermineBonds from 3D coordinates.

    Returns:
        RDKit Mol with explicit H atoms and a 3D conformer
    """
    if backend == "openbabel":
        ob_mol = pybel.readstring("xyz", xyz_str)
        molblock = ob_mol.write("mol")

        mol = Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError("RDKit failed to parse the MOL block from OpenBabel.")

        Chem.SanitizeMol(
            mol,
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
        return mol

    if backend == "rdkit":
        mol = Chem.MolFromXYZBlock(xyz_str)
        if mol is None:
            raise ValueError("RDKit failed to parse the XYZ block.")
        rdDetermineBonds.DetermineBonds(mol)
        Chem.SanitizeMol(mol)
        return mol

    raise ValueError(f"Unknown backend: {backend!r}, expected 'openbabel' or 'rdkit'")


def xyz_to_symbols_coords(xyz: str) -> tuple[list[str], np.ndarray]:
    """
    Parse a single-frame XYZ into atom symbols and coordinates.

    Args:
        xyz: XYZ content string (contains newlines) or file path

    Returns:
        Tuple of (atom_symbols, coordinates) where coordinates is shape (n_atoms, 3)
    """
    text = _read_xyz_content(xyz)
    lines = text.strip().split('\n')
    n_atoms = int(lines[0].strip())
    symbols, coords = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return symbols, np.array(coords)


def format_xyz(
    symbols: list[str],
    coords: np.ndarray,
    comment: str = ""
) -> str:
    """
    Format atom symbols and coordinates into an XYZ string.

    Args:
        symbols: Atom symbols list
        coords: Coordinates array, shape (n_atoms, 3)
        comment: Comment line (second line of XYZ format)

    Returns:
        XYZ format string
    """
    lines = [str(len(symbols)), comment]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym:>2s} {x:12.6f} {y:12.6f} {z:12.6f}")
    return '\n'.join(lines) + '\n'


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


def standardize_xyz(
    xyz: str,
    output_path: str | None = None,
    comment: str | None = None,
) -> str:
    """
    Standardize atom ordering in a single-frame XYZ.

    Full pipeline: parse XYZ -> OpenBabel bond perception -> InChI canonical
    heavy atom order -> 3D-aware H ordering -> reorder atoms -> format XYZ.

    Accepts either an XYZ content string or a file path as input.
    Always returns the standardized XYZ string. If output_path is given,
    also writes the result to that file.

    Args:
        xyz: XYZ content string (contains newlines) or file path
        output_path: If provided, write standardized XYZ to this file
        comment: Custom comment line for the output XYZ. If None, preserve
            the original comment line from the input.

    Returns:
        Standardized XYZ string
    """
    from xyz_std.atom_order import get_standard_atom_order

    xyz_str = _read_xyz_content(xyz)

    # Use custom comment or preserve the original
    if comment is None:
        lines = xyz_str.strip().split('\n')
        comment = lines[1] if len(lines) > 1 else ""

    # Parse symbols/coords from the original text
    symbols, coords = xyz_to_symbols_coords(xyz_str)

    # Create mol (with bonds and 3D) and compute standard order
    mol = xyz_to_rdkit_mol(xyz_str)
    try:
        order = get_standard_atom_order(mol)
    except Chem.rdchem.AtomValenceException:
        mol = xyz_to_rdkit_mol(xyz_str, backend="rdkit")
        order = get_standard_atom_order(mol)

    # Reorder
    symbols_std = [symbols[i] for i in order]
    coords_std = coords[order]

    # Format output
    result = format_xyz(symbols_std, coords_std, comment)

    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(result)

    return result
