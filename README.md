# xyz-std

Standardize atom ordering in XYZ molecular coordinate files.

Given a molecule (as XYZ string, file path, or RDKit Mol), produces a canonical atom order where:
- **Heavy atoms** follow InChI standard ordering (via AuxInfo `/N:` section)
- **Hydrogen atoms** are ordered by 3D-aware methods:
  - 2H prochiral sp3: deuterium substitution + CIP (pro-R before pro-S)
  - 2H prochiral sp2: CIP rank + dihedral angle (pro-Z before pro-E)
  - 3+ H (e.g., methyl): geometric CCW angle projection
  - Truly equivalent 2H: geometric fallback

## Installation

```bash
cd xyz-std
pip install -e .            # core
pip install -e ".[dev]"     # + pytest, black, flake8
```

## Quick Start

```python
from xyz_std import standardize_xyz

# From file path -> standardized XYZ string
result = standardize_xyz("molecule.xyz")

# From XYZ string -> standardized XYZ string
result = standardize_xyz(xyz_str)

# From file path -> write to file (also returns string)
result = standardize_xyz("input.xyz", output_path="output.xyz")
```

All input parameters that accept XYZ data support both **file paths** and **XYZ content strings** (auto-detected by newline presence).

## Usage

### One-step standardization

```python
from xyz_std import standardize_xyz

# Input: file path or XYZ string
# Output: standardized XYZ string (+ optional file write)
std_xyz = standardize_xyz("molecule.xyz")
std_xyz = standardize_xyz(xyz_string)
std_xyz = standardize_xyz("molecule.xyz", output_path="standardized.xyz")
```

### Step-by-step control

```python
from xyz_std import xyz_to_rdkit_mol, get_standard_atom_order, xyz_to_symbols_coords, format_xyz

# 1. Parse XYZ (accepts file path or string)
symbols, coords = xyz_to_symbols_coords("molecule.xyz")

# 2. Create RDKit Mol with bonds and 3D coordinates
with open("molecule.xyz") as f:
    mol = xyz_to_rdkit_mol(f.read())

# 3. Get canonical atom order
order = get_standard_atom_order(mol)
# [3, 0, 1, 7, 8, 5, ...]  (heavy atoms first, then H)

# 4. Reorder and format
symbols_std = [symbols[i] for i in order]
coords_std = coords[order]
xyz_out = format_xyz(symbols_std, coords_std, comment="standardized")
```

### From existing RDKit Mol

If you already have an RDKit Mol with explicit H and a 3D conformer:

```python
from xyz_std import get_standard_atom_order

order = get_standard_atom_order(mol)
```

### Write multi-frame XYZ

```python
from xyz_std import write_multi_xyz

write_multi_xyz(
    atom_symbols=symbols_std,
    coord_list=[coords_frame1, coords_frame2, ...],
    energy_list=[-100.5, -100.3, ...],
    output_path="output.xyz"
)
```

### Log Level Control

Control verbosity of RDKit and OpenBabel log output:

```python
from xyz_std import set_log_level, get_log_level

# Available levels: 'none', 'error', 'warning', 'info', 'debug'
set_log_level('none')     # Silence all logs
set_log_level('error')    # Show only errors (default)
set_log_level('warning')  # Show warnings and errors
set_log_level('info')     # Show info, warnings, and errors
set_log_level('debug')    # Show all log messages

# Get current level
level = get_log_level()  # Returns current level string
```

## Testing

```bash
pytest
```

## API Reference

| Function | Description |
|----------|-------------|
| `standardize_xyz(xyz, output_path=None)` | One-step: XYZ string/path -> standardized XYZ string (+ optional file) |
| `get_standard_atom_order(mol)` | Core: RDKit Mol -> canonical atom order `[heavy..., H...]` |
| `xyz_to_rdkit_mol(xyz_str)` | XYZ string -> RDKit Mol (via OpenBabel) |
| `xyz_to_symbols_coords(xyz)` | XYZ string/path -> (symbols, coordinates) |
| `format_xyz(symbols, coords, comment)` | Symbols + coords -> XYZ format string |
| `write_multi_xyz(symbols, coords, energies, path)` | Write multi-frame XYZ file |
| `set_log_level(level)` | Set log level for RDKit and OpenBabel (`'none'`, `'error'`, `'warning'`, `'info'`, `'debug'`) |
| `get_log_level()` | Get current log level |
