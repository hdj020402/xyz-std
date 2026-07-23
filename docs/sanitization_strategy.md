# Sanitization strategy for `xyz_to_rdkit_mol`

This document explains why `xyz_to_rdkit_mol` with `backend="openbabel"` uses a
**two-step sanitization strategy** rather than a single `sanitize=True` call.
Read this before modifying the sanitization logic — reverting to strict-only
sanitization will break charged species (NH4+, NO3-, SO4(2-), etc.).

## Background: why sanitization is hard for XYZ-derived molecules

XYZ files carry only atom symbols and 3D coordinates. They contain no bond
orders, no formal charges, and no aromaticity information. The pipeline must
infer all of this:

1. **Bond perception** — deduce connectivity from interatomic distances (done by
   OpenBabel or RDKit's `rdDetermineBonds`)
2. **Sanitization** — assign bond orders, valences, hybridization, etc. (done by
   RDKit's `Chem.SanitizeMol`)

The problem is step 2: bond perception from 3D coordinates always produces
single bonds, but RDKit's sanitization checks valences against a table of
allowed values. For neutral nitrogen, the maximum allowed valence is 3. If
OpenBabel correctly connects N to 4 H atoms (ammonium), RDKit's sanitization
rejects the molecule because neutral N with 4 bonds violates the valence rules.

## Two-step strategy

```text
Step 1: sanitize=True
  → Works for all neutral organic molecules
  → Preserves _CIPRank for allenes and other chiral molecules
  → If successful, return immediately

Step 2: sanitize=False + SANITIZE_ALL ^ SANITIZE_PROPERTIES
  → Skip valence/charge checks only
  → Still run kekulization, hybridization, chirality cleanup, etc.
  → Handles charged species with unusual valence
```

### Why not always use relaxed sanitization?

`MolFromMolBlock(sanitize=True)` sets `_CIPRank` on all atoms **during the MOL
block parsing step** (this is internal to `MolFromMolBlock`, not part of any
`SanitizeFlags`). No combination of `sanitize=False` + separate `SanitizeMol`
calls can reproduce this for allene axial chirality — allene H-ordering via
`_order_2h_cumulene` requires `_CIPRank` on the far-end substituents.

### Why not always use strict sanitization?

Strict sanitization (`sanitize=True`) calls `UpdatePropertyCache(strict=True)`,
which rejects atoms with valence outside the allowed range. Common molecules
that fail:

| Molecule | Formula | Valence issue |
|----------|---------|---------------|
| Ammonium | NH4+ | N has 4 bonds, no formal charge → valence 4 > max 3 |
| Sulfate | SO4(2-) | (passes — S valence 6 is allowed) |
| Nitrate | NO3- | (passes — N valence 3 with +1 charge is OK in OB's bond order) |

Note: SO4(2-) and NO3- pass Step 1 only because OpenBabel assigns appropriate
bond orders (S=O double bonds) that keep valence within limits. NH4+ fails
because all 4 N–H bonds are single bonds.

## RDKit `SanitizeMol` steps (v2025.09.3)

`SANITIZE_ALL` is the bitwise OR of 12 flags:

| Flag | Hex | Role | Needed for downstream? |
|------|-----|------|----------------------|
| `SANITIZE_CLEANUP` | 0x0001 | Normalize charges, radicals | No effect on XYZ mols |
| `SANITIZE_PROPERTIES` | 0x0002 | **Valence/charge validation** ← skipped in Step 2 | Only for validation |
| `SANITIZE_SYMMRINGS` | 0x0004 | Ring symmetry perception | No effect |
| `SANITIZE_KEKULIZE` | 0x0008 | Aromatic → Kekulé bond alternation | **Required** for InChI |
| `SANITIZE_FINDRADICALS` | 0x0010 | Radical detection | No effect |
| `SANITIZE_SETAROMATICITY` | 0x0020 | Aromaticity perception | No effect on XYZ mols |
| `SANITIZE_SETCONJUGATION` | 0x0040 | Conjugation perception | No effect |
| `SANITIZE_SETHYBRIDIZATION` | 0x0080 | Assign SP/SP2/SP3 | **Required** — h_ordering dispatches by `GetHybridization()` |
| `SANITIZE_CLEANUPCHIRALITY` | 0x0100 | Chirality info cleanup | No effect |
| `SANITIZE_ADJUSTHS` | 0x0200 | Adjust implicit/explicit H count | No effect (all H explicit) |
| `SANITIZE_CLEANUP_ORGANOMETALLICS` | 0x0400 | Organometallic cleanup | No effect |
| `SANITIZE_CLEANUPATROPISOMERS` | 0x0800 | Atropisomer cleanup | No effect |

**Only 2 steps are functionally required for the downstream pipeline**:
`SANITIZE_KEKULIZE` and `SANITIZE_SETHYBRIDIZATION`. `SANITIZE_PROPERTIES` is
needed for validation but can use `UpdatePropertyCache(strict=False)` to accept
unusual valences.

`SANITIZE_PROPERTIES` is the **only** flag excluded from Step 2. All other 11
steps run identically to full sanitization.

## Why the RDKit-backend fallback is kept in `standardize_xyz`

`standardize_xyz` retains a last-resort fallback to `backend="rdkit"` when
`get_standard_atom_order` raises `RuntimeError` (e.g. missing `_CIPRank`).
This covers rare cases where OpenBabel produces a structurally incorrect mol:

- **Butatriene (C=C=C=C)**: OpenBabel misidentifies the bonding as C-C≡C-C
  (single-triple-single) instead of consecutive double bonds, causing
  `_CIPRank` to be missing. RDKit's `DetermineBonds` handles this correctly.
- **Highly hypervalent molecules** (SH₆ etc.): OB may generate too few bonds,
  leaving atoms disconnected.

Note: the `AtomValenceException` catch was removed — the two-step sanitization
strategy in `xyz_to_rdkit_mol` handles valence issues internally, so this
exception no longer propagates to `standardize_xyz`.

### RDKit `DetermineBonds` with charge parameter

When the total charge is known, `rdDetermineBonds.DetermineBonds(mol, charge=N)`
performs much better — it can assign correct bond orders and formal charges for
NH4+, SO4(2-), and NO3-. However, XYZ files from QM calculations rarely carry
charge information, so the OB two-step strategy (which does not need it) is the
better default. The RDKit backend is useful as a fallback when charge is
available or OB produces incorrect connectivity.

## Test cases: how to verify the sanitization strategy

### Quick verification (run locally, `root` ML environment)

Generate test XYZ strings and verify each backend handles them correctly:

```python
from xyz_std.io import xyz_to_rdkit_mol
from rdkit import Chem

# --- NH4+ (charged — must use Step 2 relaxed fallback) ---
xyz_nh4 = """5
ammonium
N  0.0  0.0  0.0
H  0.6  0.6  0.6
H -0.6 -0.6  0.6
H  0.6 -0.6 -0.6
H -0.6  0.6 -0.6
"""
mol = xyz_to_rdkit_mol(xyz_nh4)
assert mol is not None
assert mol.GetNumAtoms() == 5
# Verify hybridization was assigned (Step 2 runs SETHYBRIDIZATION)
assert mol.GetAtomWithIdx(0).GetHybridization() == Chem.HybridizationType.SP3

# --- Allene (requires Step 1 strict for _CIPRank) ---
from rdkit.Chem import AllChem
mol_ref = Chem.MolFromSmiles("C=C=C")
mol_ref = Chem.AddHs(mol_ref)
AllChem.EmbedMolecule(mol_ref, randomSeed=42)
conf = mol_ref.GetConformer()
n = mol_ref.GetNumAtoms()
lines = [str(n), "allene"]
for i in range(n):
    pos = conf.GetAtomPosition(i)
    lines.append(f"{mol_ref.GetAtomWithIdx(i).GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
xyz_allene = "\n".join(lines) + "\n"
mol_allene = xyz_to_rdkit_mol(xyz_allene)
# _CIPRank must be present for allene axial chirality H-ordering
for atom in mol_allene.GetAtoms():
    assert "_CIPRank" in atom.GetPropsAsDict(), \
        f"Atom {atom.GetIdx()} missing _CIPRank — Step 1 not used!"
```

### How to verify Step 1 vs Step 2 is being used

Check the RDKit warning output. Step 1 produces no warnings for normal
molecules. Step 2 triggers:

```text
[WARNING] Explicit valence for atom # 0 N, 4, is greater than permitted
[WARNING] Accepted unusual valence(s): N(4)
```

Note: the first warning comes from `MolFromMolBlock(sanitize=True)` failing
in Step 1 (expected — this is what triggers the fallback). The second comes
from `SanitizeMol(..., SANITIZE_ALL ^ SANITIZE_PROPERTIES)` in Step 2
accepting the unusual valence.

### Regression checklist

When modifying `xyz_to_rdkit_mol`, verify all of these molecules produce valid
RDKit Mols with correct hybridization:

| Molecule | Type | Step used | Key check |
|----------|------|-----------|-----------|
| Ethanol (CCO) | Neutral organic | 1 | trivial |
| Allene (C=C=C) | Axial chiral | 1 | `_CIPRank` present on all atoms |
| Benzene | Aromatic | 1 | Kekulize works |
| NH4+ | Charged | 2 | N hybridization = SP3 |
| SO4(2-) | High valence | 1 or 2 | All O connected to S |
| NO3- | Charged | 1 or 2 | N hybridization = SP2 |
| PF5 | Hypervalent | 1 | 5 F connected to P |

Run the full test suite after any changes:

```bash
pytest tests/ -x -q
```

## Commit history: how we got here

The sanitization logic changed three times:

| Commit | Date | Strategy | Issue |
|--------|------|----------|-------|
| `ed027d6` (init) | — | Two-step: strict → relaxed fallback | Correct |
| `b9609f0` | 2026-03-25 | Always relaxed (`SANITIZE_ALL ^ PROPERTIES`) | Lost `_CIPRank` for allenes |
| `41fea90` | 2026-07-?? | Strict only (`sanitize=True`) | Broke NH4+, charged species |
| (current) | 2026-07-24 | **Restored two-step** (V0 strategy) | Correct |

The `41fea90` commit ("feat: add allene axial chirality H-ordering support")
reverted to strict-only sanitization because allene `_CIPRank` was missing with
the always-relaxed approach. The correct fix was to restore the two-step
strategy where strict sanitization is tried first and relaxed is the fallback,
not to abandon relaxed sanitization entirely.
