# xyz-std

Standardize atom ordering in XYZ molecular coordinate files.

Given a molecule (as XYZ string or RDKit Mol), produces a canonical atom order where:
- **Heavy atoms** follow InChI standard ordering (via AuxInfo `/N:` section)
- **Hydrogen atoms** are ordered by 3D-aware methods:
  - 2H prochiral sp3: deuterium substitution + CIP (pro-R before pro-S)
  - 2H prochiral sp2: CIP rank + dihedral angle (pro-Z before pro-E)
  - 3+ H (e.g., methyl): geometric CCW angle projection
  - Truly equivalent 2H: geometric fallback

## Installation

**Prerequisites**: Open Babel must be installed via conda:

```bash
conda install openbabel -c conda-forge
```

Then install this package:

```bash
cd xyz-std
pip install -e .            # core
pip install -e ".[dev]"     # + pytest, black, flake8
```

## Usage

### From XYZ string

```python
from xyz_std import xyz_to_rdkit_mol, get_standard_atom_order

# Read XYZ and create RDKit Mol (with bonds inferred by OpenBabel)
with open("molecule.xyz") as f:
    xyz_str = f.read()
mol = xyz_to_rdkit_mol(xyz_str)

# Get canonical atom order
order = get_standard_atom_order(mol)
# order = [3, 0, 1, 7, 8, 5, ...]  (heavy atoms first, then H)
```

### From existing RDKit Mol

If you already have an RDKit Mol with explicit H and a 3D conformer:

```python
from xyz_std import get_standard_atom_order

order = get_standard_atom_order(mol)
```

### Reorder coordinates

```python
import numpy as np
from xyz_std import xyz_to_symbols_coords

symbols, coords = xyz_to_symbols_coords("molecule.xyz")
# Apply the standard order
symbols_std = [symbols[i] for i in order]
coords_std = coords[order]
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

### H ordering on a specific atom

```python
from xyz_std import order_h_on_heavy_atom

# Order H atoms on a specific heavy atom center
h_order = order_h_on_heavy_atom(mol, center_idx=2, h_indices=[5, 6])
# Returns [pro-R, pro-S] for prochiral, or geometric order for equivalent H
```

## Testing

```bash
pytest
```

## API Reference

| Function | Description |
|----------|-------------|
| `get_standard_atom_order(mol)` | Main entry: returns canonical atom order `[heavy..., H...]` |
| `order_h_on_heavy_atom(mol, center_idx, h_indices)` | 3D-aware H ordering on a single heavy atom |
| `xyz_to_rdkit_mol(xyz_str)` | XYZ string -> RDKit Mol (via OpenBabel) |
| `xyz_to_symbols_coords(xyz_path)` | Parse XYZ file -> (symbols, coordinates) |
| `write_multi_xyz(symbols, coords, energies, path)` | Write multi-frame XYZ file |
