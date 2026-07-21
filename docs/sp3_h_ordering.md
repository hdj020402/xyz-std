# SP3 四面体 H 排序逻辑

## 通用原则

以下原则来自 `h_geometric_ordering.md` 的系统分析，适用于所有杂化类型（包括 SP3）：

1. **等价 2H 不需要 geometric**：化学方法返回 `sorted(h_indices)`（等效于 flat ordered）。取代后只剩 1H，不存在 prochiral pair。

2. **`len(h_indices) <= 1` 在调用方处理**：单 H 不应进入 geometric 排序函数。`_order_h_sp3` 入口直接返回，`_order_h_geometric` 假设输入 ≥ 2。

3. **SP3 的 geometric 无需显式 z 轴**：与 SP3D/SP3D2 不同，SP3 最多 1 个非 H 邻居（Case B）或 0 个非 H 邻居（Case A），不存在多个候选需要 tiebreak 的场景，无需 `_get_z_plus_vec` 或 `_CanonicalOrder` 参与。

---

## 基础概念

四面体（tetrahedral）有 4 个顶点，sp3 杂化。H 的化学环境取决于其余 3 个取代基的 CIP rank。

### 化学区分方式

2H 的区分通过两种互补方法：

1. **CIP 氘代法**（`_order_2h_sp3`）：将 h1 替换为 D（`SetIsotope(2)`），RDKit `AssignStereochemistry` 重新分配中心手性。若得到 R → h1 是 pro-R；S → h2 是 pro-R。

2. **有向体积法**（`_order_2h_signed_volume`）：RDKit 无法为某些中心分配 `_CIPCode` 时（P、S、As 等非碳手性中心或 3 配位锥形中心），直接用几何有向体积（四面体四个顶点按 CIP 优先序排列后计算 signed volume）判定 R/S。

### 几何排列

3H 以上的等价 H（及 1H 时不进入 geometric），沿唯一非 H 方向（Case B）或 min-idx H 方向（Case A）做 CCW angle 投影排序。算法细节见 [`geometric_algorithm.md`](geometric_algorithm.md)。

---

## 1H

单 H 直接返回 `list(h_indices)`，在 `_order_h_sp3` 入口处处理，不进入任何子逻辑。

---

## 2H

### CIP 氘代法 (`_order_2h_sp3`)

```text
原始分子:  C(H_a)(H_b)(R₁)(R₂)
氘代 H_a:  C(D)(H_b)(R₁)(R₂)  →  RDKit 分配 _CIPCode
          R → H_a = pro-R → [H_a, H_b]
          S → H_b = pro-R → [H_b, H_a]
```

RDKit `AssignStereochemistry` 能为大多数碳中心正确分配 R/S，但对 P、S、As 等非碳中心可能失败（`_CIPCode` 缺失），此时进入有向体积 fallback。

### 有向体积 fallback (`_order_2h_signed_volume`)

RDKit 无法分配的 `_CIPCode` 时，手动构建四面体四顶点按 CIP 优先序（CIP desc）排列，计算有向体积：

$$
V = (\vec{a} - \vec{d}) \cdot [(\vec{b} - \vec{d}) \times (\vec{c} - \vec{d})]
$$

其中 a > b > c > d 按 CIP rank 降序。V < 0 → R（h1 = pro-R）；V > 0 → S（h2 = pro-R）。|V| < 1e-10 → planar（无法判定）→ sorted。

#### 4 配位中心

4 个显式邻居：2 非 H + 2H。优先序：非 H（CIP desc）> D(h1) > H(h2)。

**等价判定**：若非 H 取代基 CIP rank 全相同（`len(set(non_h_ranks)) == 1`），2H 等价 → 返回 sorted。

#### 3 配位锥形中心

3 个显式邻居：1 非 H + 2H + 1 个孤对电子（phantom）。孤对电子位置由键向量和取反推断（`_infer_lone_pair_position`）。

$$
\vec{\text{lp}} \approx -\sum_i \vec{v}_i
$$

优先序：非 H > D(h1) > H(h2) > 孤对电子（最低优先）。

**等价判定**：仅 1 个非 H，无等价检查（2H 在此构型下通常是 diastereotopic）。若孤对电子位置无法推断（键向量和 ≈ 0，即平面构型）→ 返回 sorted。

---

## 3H

### Case B：有非 H 邻居（如 R–CH₃）

```text
        R (ref)
        │
    H_a─●─H_c    ← H 在 ⟂(center→R) 平面内投影，CCW atan2 排序
       ╱ ╲
     H_b
```

唯一非 H 邻居为参考方向（z 轴）。3 个 H 沿此方向投影到 ⟂z 平面，CCW atan2 排序。

### Case A：无非 H 邻居（如 NH₃、PH₃）

3 个 H 全等价且无重原子参考。min-idx H 排第一，该 H 作为参考方向（z 轴），剩余 2H 沿此方向 CCW atan2 排序。

```text
      H_min_idx (ref, 排第一)
        │
    H_b─●─H_c    ← 剩余 2H CCW 排序
```

---

## 4H（CH₄、SiH₄）

4 个 H 全等价。min-idx H 排第一，该 H 作为参考方向（z 轴），剩余 3H 沿此方向 CCW atan2 排序。

```text
      H_min_idx (ref, 排第一)
        │
    H_b─●─H_d    ← 剩余 3H CCW 排序
       ╱ ╲
     H_c
```

---

## 异常处理

### 3 配位孤对电子推断失败

`_infer_lone_pair_position` 在键向量和 ≈ 0（平面构型）时返回 None → `_order_2h_signed_volume` 返回 sorted。

### 有向体积 ≈ 0

四面体顶点共面（planar）时 |V| < 1e-10，无法判定手性 → 返回 sorted。

### 非 3/4 配位

`_order_2h_signed_volume` 要求 `n_explicit` 为 3 或 4，否则直接返回 sorted。

---

## 流程总结

```text
_order_h_sp3:
  n_H ≤ 1 → 直接返回

  n_H == 2:
    _order_2h_sp3:
      └─ 氘代 h1 → RDKit AssignStereochemistry
         ├─ _CIPCode = R → [h1, h2]  (h1 = pro-R)
         ├─ _CIPCode = S → [h2, h1]  (h2 = pro-R)
         └─ 无 _CIPCode → _order_2h_signed_volume:
              ├─ 4 配位: 非 H（CIP desc）> D(h1) > H(h2)
              ├─ 3 配位: 非 H > D(h1) > H(h2) > 孤对电子
              ├─ V < 0 → [h1, h2]  (R)
              ├─ V > 0 → [h2, h1]  (S)
              └─ 等价/退化 → sorted(h_indices)

  n_H ≥ 3:
    _order_h_geometric:
      ├─ 有非 H 邻居（Case B）→ 非 H 为 ref，所有 H CCW 排序
      └─ 无非 H 邻居（Case A）→ min-idx H 排第一 + 剩余 H CCW 排序

  说明：
  - SP3 的 geometric 场景中 ref 选择无歧义（≤1 个非 H 候选），
    无需 tiebreak 或 _CanonicalOrder
  - 2H 等价等价于 sorted，无需 geometric
  - 有向体积 fallback 涵盖 RDKit 不支持的非碳手性中心
```
