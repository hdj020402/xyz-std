# SP3D 三角双锥 H 排序逻辑

## 通用原则

以下原则来自 `h_geometric_ordering.md` 的系统分析，适用于所有杂化类型（包括 SP3D）：

1. **等价 2H 不需要 geometric**：化学方法返回 `None` 意味着 2H 等价。取代后只剩 1H，不存在 prochiral pair，返回 `sorted(h_indices)` 即可。

2. **`len(h_indices) <= 1` 在调用方处理**：单 H 不应进入 geometric 排序函数。各分发函数入口检查，`_order_h_geometric` 假设输入 ≥ 2。

3. **geometric 的 z 轴显式指定**：SP3D 的 geometric 场景均以轴向 pair 确定 z 轴方向，不使用 `max(non_h_neighbors, key=...)` 自动选择。

4. **等价候选用 `_CanonicalOrder` 定方向**：当轴向 pair 两端 CIP rank 相同时，z⁺ 由 InChI 规范序号确定（规范序号小的 → z⁺）。`_CanonicalOrder` 在 `get_standard_atom_order` 中通过 `SetIntProp` 写入，RWMol copy 与 `AssignStereochemistry` 均不丢失。受影响的场景：ax 等价 + eq 3H。

---

## 基础概念

三角双锥（trigonal bipyramidal）有 5 个顶点，分为两组：

- **2 个轴向（axial）**：互为 trans（180°），垂直于赤道平面
- **3 个赤道（equatorial）**：在赤道平面内，互相间隔约 120°

`_classify_sp3d_positions` 通过几何角度检测来区分 axial 与 equatorial：5 个邻居中夹角最大（>140°）的一对为 axial，其余 3 个为 equatorial。

### 化学等价判定

- **ax 2H**：等价条件是赤道 3 个取代基的 CIP rank 有重复（`len(set(eq_ranks)) < 3`）。赤道平面对称时，两个轴向 H 无法区分。
- **eq 2H**：等价条件是轴向 2 个取代基的 CIP rank 相同（`len(set(ax_ranks)) < 2`）。轴向两端不可区分时，赤道 H 无法区分。
- **eq 3H**：等价条件是轴向 2 个取代基的 CIP rank 相同。此时 z⁺ 方向由 `_CanonicalOrder` 确定，geometric CCW 仍可给出确定性排序。

---

## 1H

单 H 直接返回 `list(h_indices)`，在 `_order_h_sp3d` 入口处处理，不进入任何子逻辑。

---

## 2H

### 2H 均为 axial（ax-2H）

两个 H 在轴向两端互为 trans。赤道 3 个取代基在赤道平面上。

`_order_sp3d_axial_2h`：

1. 检查赤道 3 个取代基的 CIP rank 是否全不同 → 否则等价
2. 赤道取代基按 CIP rank 降序排列：a > b > c
3. 沿 h1→h2 方向观察，a→b→c 是否呈 CCW → 是则 h1 为 z⁺ 端
4. 返回 `[z⁺_idx, z⁻_idx]`；等价时返回 `None` → `sorted(h_indices)`

```text
       h1 (ax, z⁺)
       │
   a───●───b   ← 赤道平面（3 取代基 a>b>c CIP）
      ╱ ╲
     c   │
         │
       h2 (ax, z⁻)
```

### 2H 均为 equatorial（eq-2H）

两个 H 在赤道平面上。轴向 2 个取代基定义 z 轴。

`_order_sp3d_equatorial_2h`：

1. 检查轴向 2 个取代基的 CIP rank 是否不同 → 否则等价
2. z⁺ 方向 = ax_low → ax_high（z⁺ 端为高 CIP rank 轴向）
3. 赤道非 H 取代基作为角度参考（0°）
4. 两个 H 按 atan2 CCW 角度排序
5. 等价时返回 `None` → `sorted(h_indices)`

```text
       ax_high (z⁺)
       │
   H_a───●───H_b   ← 赤道平面
      ╱  ╲
    NH ╱    ╲
       │
       ax_low (z⁻)

   赤道非 H（NH）为 0° 参考，H 按 CCW 排序
```

### 1 ax + 1 eq

ax H 直接排第一，eq H 直接排第二。无需化学或几何排序（各自仅 1 个，无歧义）。

---

## 3H

### 2 ax + 1 eq

- ax 2H → `_order_sp3d_axial_2h`（等价时 `sorted`）
- eq 1H → 直接追加

输出：ax 组 + eq 组

### 1 ax + 2 eq

- ax 1H → 直接排第一
- eq 2H → `_order_sp3d_equatorial_2h`（等价时 `sorted`）

输出：ax H + eq 组

### 0 ax + 3 eq（ax 均为非 H）

3 个赤道 H，2 个轴向非 H。

- ax pair 的 z⁺ 方向由 `_get_z_plus_vec` 确定（CIP rank → `_CanonicalOrder`）
- 3 个 eq H 沿此方向做 geometric CCW 投影排序

```text
       NH_high (z⁺)
       │
   H_a───●───H_b   ← 赤道平面（3H）
      ╱  ╲
    H_c╱    ╲
       │
       NH_low (z⁻)

   3H CCW geometric 排序
```

z⁺ 决定规则：

- **NH_high CIP ≠ NH_low CIP**：高 CIP rank → z⁺
- **NH_high CIP = NH_low CIP**：`_CanonicalOrder` 小 → z⁺

---

## 4H

### 2 ax + 2 eq

- ax 2H → `_order_sp3d_axial_2h`（等价时 `sorted`）
- eq 2H → `_order_sp3d_equatorial_2h`（等价时 `sorted`）。第 3 个赤道位置为非 H，作为角度参考

输出：ax 组 + eq 组

### 1 ax + 3 eq

- ax 1H → 直接排第一
- eq 3H → axial pair 中一个为 H、一个为非 H。`_get_z_plus_vec` 比较 CIP rank：非 H（高 rank）→ z⁺。3 个 eq H 沿此方向 CCW geometric 排序

```text
       NH (z⁺, CIP rank 高于 H)
       │
   H_a───●───H_b   ← 赤道平面（3H）
      ╱  ╲
    H_c╱    ╲
       │
       H_ax (z⁻)

   3H CCW geometric 排序
```

输出：ax H + eq 3H

---

## 5H

5 个邻居全是 H。2 ax + 3 eq。

- ax 2H：`_order_sp3d_axial_2h` 检查赤道取代基 CIP rank。赤道均为 H → rank 相同 → `len(set) = 1 < 3` → 等价 → `sorted(axial_h)`
- eq 3H：`_get_z_plus_vec` 在轴向 pair（均为 H）上，直接按较小 index → z⁺，不触及 `_get_cip_rank`。geometric CCW 排序

```text
       H_ax1 (z⁺)
       │
   H_a───●───H_b   ← 赤道平面（3H）
      ╱  ╲
    H_c╱    ╲
       │
       H_ax2 (z⁻)

   ax 2H 等价 → sorted
   eq 3H → geometric CCW（z⁺ = min(ax_indices)）
```

---

## 异常处理

### 位置分类失败

`_classify_sp3d_positions` 要求：

- 恰好 5 个邻居
- 存在夹角 >140° 的 axial pair

若不满足（邻居数 ≠ 5 或最大夹角 < 140°），全部邻居视为 equatorial，fallback 到 `_order_h_geometric(mol, center_idx, h_indices)` 做纯几何排序。

---

## 流程总结

```text
_order_h_sp3d:
  _classify_sp3d_positions → axial_nbrs, eq_nbrs
  分类失败 (≠2 axial) → fallback _order_h_geometric

  axial_h = H ∩ axial_nbrs
  eq_h    = H ∩ eq_nbrs

  n_H == 2:
    ax-2H    → _order_sp3d_axial_2h（赤道手性 / 等价 sorted）
    eq-2H    → _order_sp3d_equatorial_2h（轴向 CIP + atan2 / 等价 sorted）
    1ax+1eq  → 各直接追加

  n_H == 3:
    2ax+1eq  → ax-2H 子问题 + eq 直接
    1ax+2eq  → ax 直接 + eq-2H 子问题
    0ax+3eq  → _get_z_plus_vec(ax pair) + geometric CCW

  n_H == 4:
    2ax+2eq  → ax-2H 子问题 + eq-2H 子问题
    1ax+3eq  → ax 直接 + _get_z_plus_vec + geometric CCW

  n_H == 5:
    2ax+3eq  → ax-2H（等价 sorted）+ _get_z_plus_vec + geometric CCW

  说明：
  - geometric 的 z 轴均显式由 axial pair 确定，不使用 max-Z 自动选择
  - 等价 2H 在任何路径中都不需要 geometric，直接返回 sorted 即可
  - axial H 始终排在 equatorial H 之前
```
