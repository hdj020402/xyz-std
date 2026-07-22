# SP3D2 八面体 H 排序逻辑

## 通用原则

以下原则来自 `h_geometric_ordering.md` 的系统分析，适用于所有杂化类型（包括 SP3D2）：

1. **等价 2H 不需要 geometric**：化学方法返回 `None` 意味着 2H 等价。取代后只剩 1H，不存在 prochiral pair，返回 `sorted(h_indices)` 即可。

2. **`len(h_indices) <= 1` 在调用方处理**：单 H 不应进入 geometric 排序函数。各分发函数入口检查，`_order_h_geometric` 假设输入 ≥ 2。

3. **geometric 的 z 轴显式指定**：SP3D2 的 geometric 场景均以 **trans pair** 确定 z 轴方向，不使用 `max(non_h_neighbors, key=lambda n: (n.GetAtomicNum(), -n.GetIdx()))` 自动选择。

4. **等价候选用 `_CanonicalOrder` 定方向**：当 trans pair 两端 CIP rank 相同时，z⁺ 由 InChI 规范序号确定（规范序号小的 → z⁺）。`_CanonicalOrder` 在 `get_standard_atom_order` 中通过 `SetIntProp` 写入，RWMol copy 与 `AssignStereochemistry` 均不丢失。受影响的场景：SP3D2 4H non-H trans（ax 等价）、SP3D eq 3H（ax 等价）。

---

## 基础概念

八面体 6 个顶点，组成 3 对 trans（对位，180°）。每个顶点有 1 个 trans + 4 个 cis 邻居。H 的化学环境由其 trans 伙伴是谁 + 4 个 cis 邻居是谁共同决定。

### 正方形手性分析

2 个 trans 原子的 cis 平面（正方形，4 点）的手性判定：

- **Step 1**：4 个 cis 原子的 CIP rank 是否有 2 组（或以上）重复 → 是则等价，返回 None
- **Step 2**：正方形对角线（即 4 点中互为 trans 的对）两端是否 CIP rank 相同 → 是则等价，返回 None
- **Step 3**：去掉一个原子（优先去重复 rank，其次去最低 rank）使剩下 3 个 rank 全不同。若无法达到 → 等价，返回 None
- **Step 4**：3 点三角形 CW/CCW 分析：沿 trans 轴看，CIP 降序排列呈 CCW → z⁺ 端 H 优先

---

## 2H

> **原则**：等价 2H（化学方法返回 None）不需要 geometric。取代后只剩 1H，不存在 prochiral pair，返回 `sorted(h_indices)` 即可。详见 `h_geometric_ordering.md`。

### 2H trans

两个 H 互为 trans。4 个 cis 取代基在正方形上。

调用正方形手性分析。CCW → [h1, h2]；CW → [h2, h1]；None → 2H 等价，直接返回。

### 2H cis

两个 H 互不 trans。各自有 trans 伙伴 T_a、T_b。

- **T_a CIP ≠ T_b CIP**：trans 伙伴 rank 高的 H 在前
- **T_a CIP = T_b CIP**：分别分析每个 H 的 cis 正方形（视线方向 T→H，即从 trans 伙伴看向 H）。两个正方形互为镜像，结果必然一个 CCW 一个 CW。CCW 的 H 在前。两个 square 都返回 None → 2H 等价，直接返回

---

## 3H

### fac（3 个 H 互为 cis）

3 个 trans 对均为 H-X。三个 H 按 trans 伙伴的 CIP rank 处理：

- **ABC**：trans 伙伴 rank 全不同 → 按 rank 降序排列
- **AAB**：2 个 H 的 trans 伙伴 rank 相同（A），1 个不同（B）→ 独特 H（trans = B）排第一；剩下 2 个 H 退化为 2H cis（T_a = T_b = A）继续处理
- **AAA**：3 个 H 的 trans 伙伴 rank 全相同 → 三个 H 化学等价。任意选一个 H（min idx）排第一，氘代该 H 后剩余 2H cis 走 cis-2H 逻辑（对角线天然异 rank，氘代破缺对称后可通过 cis-square chirality 确定顺序）

### mer（含 1 对 H-H trans + 1 个 H-X）

H-X 的 H 环境独特（trans 非 H），排第一。H-H 对走 2H trans 逻辑。

---

## 4H

### 非 H trans（2 个非 H 互为 trans）

4 个 H 全在赤道正方形上。以 trans 非 H 对为 z 轴，CCW 排序 eq 4H。

z⁺ 方向：若非 H 两端 CIP rank 不同 → 高 CIP rank → z⁺；若相同 → `_CanonicalOrder` 小 → z⁺。

### 非 H cis（2 个非 H 互为 cis）

trans 对分布：1 对 H-H + 2 对 H-X。

- **2 个 H-X 的 H**：按 trans 伙伴 CIP rank 降序排列。若 rank 相同 → 2H 等价，直接返回
- **1 对 H-H**：走 2H trans 逻辑

输出：H-X 组 + H-H 对

---

## 5H

1 个非 H + 5 个 H。trans 对分布：2 对 H-H + 1 对 H-X。

H-X 的 H 环境独特（trans 非 H），作为 ax 排第一。其余 4 个 H（2 对 H-H）在赤道正方形上，环境等价 → 以 trans H-X 对为 z 轴，CCW 排序 eq 4H。

---

## 6H

3 对 H-H 全等价。ax 对 2H 等价，无需 geometric。选一对 trans 作为 ax（按 trans pair 的 min idx 确定），ax 对排第一；其余 4 个 eq H 沿 ax 方向做 CCW geometric 排序。

---

## 流程总结

```text
_order_h_sp3d2:
  _find_sp3d2_trans_pairs → 3 对 trans（<3 → RuntimeError）→ trans_of 双向映射
  分类 trans pairs → hh_pairs / hx_pairs / xx_pairs

  n_H == 2:
    len(hh_pairs)==1 → _order_sp3d2_trans_2h（正方形手性 / sorted）
    len(hx_pairs)==2 → _order_sp3d2_cis_2h（trans 伙伴 CIP / T→H 正方形手性 / sorted）

  n_H == 3 → _order_sp3d2_3h:
    fac (3×H-X) → ABC（CIP 降序）/ AAB（unique H + 2H cis）
                    / AAA（min-idx H + _deuterate_atom + cis-2H）
    mer (1×H-H + 1×H-X) → H-X 先 + _order_sp3d2_trans_2h

  n_H == 4 → _order_sp3d2_4h:
    非 H trans → _get_z_plus_vec z⁺ + _order_h_geometric CCW
    非 H cis  → H-X 组（CIP 降序 / rank 同则 sorted）+ _order_sp3d2_trans_2h

  n_H == 5 → _order_sp3d2_5h:
    H-X 先（ax）→ _get_z_plus_vec(H, X) z⁺ + _order_h_geometric CCW eq 4H

  n_H == 6 → _order_sp3d2_6h:
    选 ax trans 对（min-idx）先 → _get_z_plus_vec z⁺ + 其余 4H geometric

  说明：
  - _analyze_square_chirality 接收 trans_of 避免重复计算对角线
  - Step 1 用 len(set(rank_values)) < 3 取代 Counter
  - geometric 的 z 轴均显式由 trans pair 确定，不使用 max-Z 自动选择
  - 等价 2H 在任何路径中都不需要 geometric，直接 sorted 返回
```
