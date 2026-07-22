# SP2 平面型 H 排序逻辑

## 通用原则

以下原则来自 `h_geometric_ordering.md` 的系统分析，适用于所有杂化类型（包括 SP2）：

1. **等价 2H 不需要 geometric**：化学方法返回 `sorted(h_indices)`（等效于 flat ordered）。取代后只剩 1H，不存在 prochiral pair。

2. **`len(h_indices) <= 1` 在调用方处理**：单 H 不应进入 geometric 排序函数。`_order_h_on_heavy_atom` 入口直接返回，`_order_h_geometric` 假设输入 ≥ 2。

3. **SP2 的 2H 区分通过二面角**：与 SP3 的 CIP 氘代法不同，SP2 使用 partner 原子的最高 CIP rank 取代基作为二面角参考，判定 pro-Z / pro-E。

---

## 基础概念

SP2 碳为平面三角形（trigonal planar），3 个 σ 键方向共面呈 120°。

**SP2 ≠ 有双键**。双键（C=C、C=O 等）中的原子通常为 sp2，但 sp2 也可以只含单键：
碳正离子（R₃C⁺）、自由基、共轭碳负离子，以及 B（BH₃）、N、O 等杂原子。RDKit 基于几何/电子环境分配杂化类型，与键型不完全对应。

因此 `_order_h_sp2` 需要处理三种场景：

| 场景 | 示例 | H 数 | 双键 | 处理方式 |
| :--- | :--- | :--- | :--- | :--- |
| 烯烃 =CH₂ | H₂C=CH–R | 2 | 有 | `_order_2h_sp2`（Z/E 或累积烯） |
| sp2 无双键 | H₂B–R、H₂C⁺–R | 2 | 无 | `sorted`（无 prochiral 参考方向） |
| sp2 多 H | BH₃ | 3 | 无 | `sorted`（3H 在平面正三角形上全部等价） |

---

## 1H

单 H 直接返回 `list(h_indices)`，在 `_order_h_on_heavy_atom` 入口处处理，不进入 SP2 子逻辑。

---

## 2H

### 有双键：烯烃 (`_order_2h_sp2`)

partner 为 sp² 时的标准路径：

```text
原始分子:  R₁R₂C=CH₂
partner:   =CR₁R₂（与 center 双键相连）

partner 取代基（排除 center）= R₁, R₂
参考 = max(R₁, R₂) by CIP rank

二面角: H₁–C–C–Ref
  |dihedral| < 90° → H₁ = pro-Z（H₁ cis 于参考基团）
  |dihedral| ≥ 90° → H₂ = pro-Z（H₂ cis 于参考基团）

返回: [pro-Z_idx, pro-E_idx]
```

**等价判定**：partner 的所有取代基 CIP rank 全相同（`len(set(ranks)) == 1`）→ 2H 等价 → `sorted(h_indices)`。

例如：`H₂C=CH₂`（乙烯）partner 只有 H 作为取代基 → 等价；`R–CH=CH₂` partner 有 R 和 H → 可区分。

### 有双键：累积烯 (`_order_2h_cumulene`)

partner 为 sp 时触发。沿累积双键 walk 到远端终点（`_walk_cumulene_far_end`），根据路径中 sp 碳数量奇偶选择方法：

```text
center=C=...=C=far_end
        └── sp_count 个 sp 碳
```

#### 奇数 sp（轴向手性，Rₐ/Sₐ）

以丙二烯（allene，`sp_count = 1`）为代表。两端 =CH₂ 平面互相垂直，存在轴手性。

**CIP 判定规则**（IUPAC）：沿轴方向从近端看向远端，路径为**近端大基团 → 近端小基团 → 远端大基团**（不看远端小基团）。若为顺时针 → Rₐ，逆时针 → Sₐ。

```text
        H₁ (近端大基团, pro-Rₐ 候选)
       ╱
  C=C=C=C(far)
       ╲
        H₂ (近端小基团, pro-Sₐ 候选)

沿 C=C=C 轴方向观察 ⟂轴 平面投影，
CIP 路径 H₁ → H₂ → far_c（远端最高 CIP 取代基）:
    CW  → Rₐ → H₁ = pro-Rₐ → [H₁, H₂]
    CCW → Sₐ → H₂ = pro-Rₐ → [H₂, H₁]
```

**代码实现**：用 `_signed_angle_between(axis, H1, far_c)` 计算 H1 到 far_c 的有向角。由于 H1 和 H2 在投影面互为对跖点（约 180°），H1→far_c 两点的 CW/CCW 方向与 CIP 三点路径 H1→H2→far_c 的 CW/CCW 方向**恰好相反**：

```text
  signed_angle(H1, far_c) < 0 (CW)
    → CIP 路径 H1→H2→far_c 为 CCW → Sₐ → H₂ = pro-Rₐ

  signed_angle(H1, far_c) > 0 (CCW)
    → CIP 路径 H1→H2→far_c 为 CW → Rₐ → H₁ = pro-Rₐ
```

#### 偶数 sp（共面，Z/E）

以丁三烯（butatriene，`sp_count = 2`）为代表。两端 =CH₂ 平面共面，退化回二面角判定：

```text
二面角: H₁–C–C=C=C–far_c
  |dihedral| < 90° → H₁ = pro-Z → [H₁, H₂]
  |dihedral| ≥ 90° → H₂ = pro-Z → [H₂, H₁]
```

**等价判定**（两种 sp 奇偶共用）：

- 远端无取代基（`far_subs` 为空）→ sorted
- 远端所有取代基 CIP rank 全相同 → sorted
- `_walk_cumulene_far_end` 失败（walk 中途异常）→ sorted

### 无双键：`sorted(h_indices)`

sp2 无双键时（如碳正离子 R₂C⁺–CH₃），没有 partner 端提供二面角参考方向，2H 无法通过化学方法区分 → `sorted(h_indices)`。

### `_walk_cumulene_far_end` — 累积烯 Walk

```text
从 partner_idx 出发，沿双键 chain 前进：
  while cursor 是 sp:
    沿双键找 next（排除 prev 方向）
    若 next 不是唯一 → 失败返回 None
    prev = cursor, cursor = next, sp_count += 1

返回 (far_atom_idx, prev_idx, sp_count)
```

sp_count 即为累积双键中 sp 碳的数量：

- `sp_count = 0`：邻位直接是非 sp 原子（罕见退化情况）
- `sp_count = 1`：allene（丙二烯）→ 奇数 → 轴向手性 Rₐ/Sₐ
- `sp_count = 2`：butatriene（丁三烯）→ 偶数 → 二面角 Z/E
- 以此类推

---

## ≥ 3H

sp2 为平面正三角形（trigonal planar），3 个 H 化学等价。无需 geometric CCW——直接 `sorted(h_indices)` 即可。

---

## 异常处理

### `_walk_cumulene_far_end` 失败

在 walk 过程中遇到以下情况返回 None：

- sp 碳沿双键的下一原子不唯一（分支）
- 累积链中断

返回 None 时 `_order_2h_cumulene` 直接 `sorted(h_indices)`。

### 远端无取代基或全等价

- 远端无取代基（`far_subs` 为空）→ sorted
- 远端取代基 CIP rank 全相同 → sorted

### partner 无取代基（普通烯烃）

`partner_subs` 为空（partner 只连着 center，极其罕见）→ sorted。

---

## 流程总结

```text
_order_h_sp2(mol, center_idx, h_indices):
  n_H == 2:
    找 center 的双键 partner:
      有 → _order_2h_sp2:
              partner 是 sp?
                ├─ 是 → _order_2h_cumulene:
                │         _walk_cumulene_far_end:
                │           失败 → sorted
                │           远端无取代基 / 全等价 → sorted
                │           sp_count 奇数（轴向手性）:
                │             signed_angle(H1, far_c) < 0 (CW)
                │               → CIP 三点路径 CCW → Sₐ → [h2, h1] (h2=pro-Rₐ)
                │             signed_angle(H1, far_c) > 0 (CCW)
                │               → CIP 三点路径 CW → Rₐ → [h1, h2] (h1=pro-Rₐ)
                │           sp_count 偶数（共面）:
                │             dihedral |···| < 90° → [h1, h2]  (h1 = pro-Z)
                │             |dihedral| ≥ 90° → [h2, h1]  (h2 = pro-Z)
                │
                └─ 否（普通烯烃）:
                      partner 取代基 CIP rank 全相同? → sorted
                      取最高 CIP rank 取代基为 ref
                      dihedral |···| < 90° → [h1, h2]  (h1 = pro-Z)
                               |···| ≥ 90° → [h2, h1]  (h2 = pro-Z)
      无 → sorted  (sp2 无双键：碳正离子、自由基等)

  n_H ≥ 3:
    sorted(h_indices)  (如 BH₃ 有 3H，平面正三角形全部等价)

  说明：
  - SP2 ≠ 有双键：sp2 描述轨道几何，双键描述成键类型
  - 无双键的 sp2（碳正离子等）2H 无法用 Z/E 区分，直接 sorted
  - sp2 是平面正三角形，3H 全部化学等价，直接 sorted 即可
  - 累积烯轴向手性（奇数 sp）是 SP2 特有的场景
  - 等价 2H 不需要 geometric，直接 sorted 即可
```
