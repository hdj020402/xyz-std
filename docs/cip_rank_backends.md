# CIP Rank (`_CIPRank`) 可用性：OpenBabel vs RDKit

本文档记录 `_CIPRank` 属性在不同后端和 RDKit 版本中的行为差异，以及本项目的应对策略。

## 背景

`_CIPRank` 是 RDKit 原子上的一个 int 属性，由 `AssignStereochemistry()` 写入。它是 CIP（Cahn-Ingold-Prelog）优先级规则在原子级别的编码——数值越高，优先级越高。本项目的 H 排序逻辑大量依赖 `_CIPRank` 来判断取代基的化学优先级。

## 生产管线

`standardize_xyz` 的生产路径为：

```text
SMILES / XYZ 输入
  → xyz_to_rdkit_mol(xyz_str, backend="openbabel")   # 首选 OpenBabel
  → get_standard_atom_order(mol)
      → AssignAtomChiralTagsFromStructure(mol)
      → AssignStereochemistry(mol, cleanIt=True, force=True)
      → MolToInchiAndAuxInfo → InChI /N: → heavy_order
      → _order_h_on_heavy_atom → _get_cip_rank(...)
```

若 OpenBabel 路径抛出 `AtomValenceException` 或 `RuntimeError`（含缺失 `_CIPRank`），则回退到 RDKit 直接路径：

```text
  → xyz_to_rdkit_mol(xyz_str, backend="rdkit")        # 回退 RDKit
  → get_standard_atom_order(mol)                       # 重新走相同流程
```

## 各后端行为对比

### OpenBabel（`backend="openbabel"`）

| 特性 | 行为 |
| :--- | :--- |
| Bond perception | OpenBabel 从 XYZ 坐标推断键连 → 写 MOL block → RDKit 解析 |
| `_CIPRank` 覆盖 | **所有原子**，包括对称分子中的等价原子 |
| 已知问题 | 高配位分子（SH₆、PH₅ 等）键连推断错误，无法正确往返 |

OpenBabel 为**所有**原子（包括 H）分配 `_CIPRank`。对称分子（如 CH₄、PH₅）中的等价原子也会得到相同的 rank 值，不存在缺失 `_CIPRank` 的情况。

OpenBabel 对高配位（hypervalent）分子的键连推断不可靠：

- SH₆：OpenBabel 只生成 2 个 S-H 键（其余 H 被视为游离原子）
- 这类分子无法通过 XYZ → OpenBabel → RDKit roundtrip，需要在测试中用 `_make_oct_mol` 直接构建

### RDKit 直接（`backend="rdkit"`）

| RDKit 版本 | `_CIPRank` 行为 |
| :--- | :--- |
| **2025.03.5** | 所有原子均获得 `_CIPRank` |
| **2025.09.3** | 对称分子中部分原子可能缺失 `_CIPRank` |

RDKit 2025.09.3 对高度对称的分子（如 CH₄、PH₅、C₂H₆ 等）调用 `AssignStereochemistry` 后，某些原子的 `_CIPRank` 属性不会被写入——RDKit 判定这些分子不具有立体化学，因此跳过了 CIP rank 的计算。

RDKit 2025.03.5 的行为不同：即使分子对称，也会为所有原子分配 `_CIPRank`（等价原子获得相同 rank 值）。

## `_get_cip_rank` 的设计

```python
def _get_cip_rank(atom: Chem.Atom) -> int:
    props = atom.GetPropsAsDict()
    if '_CIPRank' not in props:
        raise RuntimeError(...)
    return int(props['_CIPRank'])
```

此函数**不尝试处理缺失 `_CIPRank` 的情况**——它直接抛出 `RuntimeError`。这是有意为之：

1. **生产环境中**：OpenBabel 路径保证所有原子都有 `_CIPRank`。若 OpenBabel 失败，`standardize_xyz` 会捕获 `RuntimeError` 并回退到 RDKit 直接路径。
2. **调用方责任**：所有 `_get_cip_rank` 的调用方都有责任确保只在确实有 `_CIPRank` 的原子（由 OpenBabel 提供）上调用。

### `_get_z_plus_vec` 的 H 预检

`_get_z_plus_vec` 在调用 `_get_cip_rank` 之前先检查原子是否为 H：

```python
a_is_h = mol.GetAtomWithIdx(a).GetAtomicNum() == 1
b_is_h = mol.GetAtomWithIdx(b).GetAtomicNum() == 1

if a_is_h and b_is_h:
    z_plus = a if a < b else b       # H/H: 按 index 选
elif a_is_h:
    z_plus = b                        # H/X: X 是 z⁺
elif b_is_h:
    z_plus = a
else:
    # 两个都是重原子 → 可以安全调用 _get_cip_rank
    r0 = _get_cip_rank(mol.GetAtomWithIdx(a))
    r1 = _get_cip_rank(mol.GetAtomWithIdx(b))
    ...
```

这保证了 `_get_cip_rank` 永远不会在 H 原子上被调用——H 原子的 `_CIPRank` 在某些 RDKit 版本中可能不可靠（即使 OpenBabel 分配了），但实际调用路径已经通过 H/H 和 H/X 臂避免了触及 H 上的 `_CIPRank`。

## 各调用方安全性审查

| 函数 | 调用 `_get_cip_rank` 的对象 | 安全性 |
| :--- | :--- | :--- |
| `_get_z_plus_vec` | 两个都是重原子（X/X） | ✅ H 已被预检排除 |
| `_order_2h_sp2` | partner 的取代基 | ✅ 取代基为 C/N/O/F/Cl/Br 等重原子 |
| `_order_2h_cumulene` | 远端取代基 | ✅ 同上 |
| `_order_2h_signed_volume` | 中心原子的非 H 邻居 | ✅ 显式过滤 H |
| `_order_sp3d_axial_2h` | 赤道取代基 | ✅ 赤道非 H 取代基 |
| `_order_sp3d_equatorial_2h` | 轴向取代基 | ✅ 轴向非 H 取代基 |
| `_analyze_square_chirality` | 正方形上的 cis 取代基 | ✅ 正方形在排除 H 后形成 |
| `_order_sp3d2_cis_2h` | trans partner（T_a、T_b） | ✅ trans partner 为 X（非 H） |
| `_order_sp3d2_3h` | trans partner（H-X 中的 X） | ✅ X 为 H 的 trans partner |
| `_order_sp3d2_4h` | H-X 中的 trans partner X | ✅ 同上 |

**结论**：所有 `_get_cip_rank` 调用都发生在重原子（非 H）上，且这些原子在生产管线中由 OpenBabel 保证 `_CIPRank` 可用。

## 测试中的处理

单元测试使用 `_make_mol_from_xyz`（OpenBabel 路径），与生产管线一致：

```python
def _make_mol_from_xyz(smiles, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    # ... 生成 XYZ 字符串 ...
    try:
        mol_ob = xyz_to_rdkit_mol(xyz_str)                     # OpenBabel
    except Exception:
        mol_ob = xyz_to_rdkit_mol(xyz_str, backend="rdkit")    # RDKit fallback
    Chem.AssignAtomChiralTagsFromStructure(mol_ob)
    Chem.AssignStereochemistry(mol_ob, cleanIt=True, force=True)
    return mol_ob
```

**例外**：

- 偶数累积烯烃（butatriene）测试使用 `_make_cumulene_mol`（RDKit 直接 + 手动 `_CIPRank`），因为 OpenBabel 会误判丁三烯的键型（C-C≡C-C 而非 C=C=C=C）。
- 高配位分子（SH₆ 等）测试使用 `_make_oct_mol`（RWMol 直接构建），因为它们无法通过 XYZ roundtrip。
- SP3D 分子测试使用 `_make_sp3d_mol`（RWMol + 手动坐标 + 手动 `_CIPRank`），因为 RDKit 不会自动为简单连接分配 SP3D 杂化。

## 环境总结

| 环境 | RDKit 版本 | OpenBabel 可用 | `_CIPRank` 行为 |
| :--- | :--- | :--- | :--- |
| **CPU 服务器**（newconsole） | 2025.03.5 | ✅ | 所有原子均有 |
| **WSL**（本地） | 2025.09.3 | ✅（通过 sshfs 访问远程文件） | RDKit 直连路径缺失对称分子 |
| **GPU 服务器**（gpu0001/2） | 取决于安装 | 取决于安装 | 未测试 |

由于生产管线优先走 OpenBabel 路径，`_CIPRank` 在生产中始终可用。RDKit 版本差异仅在 OpenBabel 路径失败、回退到 RDKit 直接路径时才可能暴露——此时 `RuntimeError` 被 `standardize_xyz` 捕获并触发 RDKit 回退。
