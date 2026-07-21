# `_order_h_geometric` 函数分析与改进方案

## 原始设计目的

`_order_h_geometric` 最初为 sp3 中心上 ≥3 个等价 H 设计，目标是保证**帧间一致性**。以 R-CH₃ 为例：

```text
帧 1: H_a, H_b, H_c (CCW)     帧 2: H_a, H_c, H_b (CW 翻转)
```

若无 geometric ordering，不同帧中"第一个 H"对应的空间方向可能不同：

1. 帧 1 用 D 取代"第一个 H" → R-CDH₂，剩余 2H 中"第二个 H"在 pro-R 位置
2. 帧 2 用 D 取代"第一个 H" → R-CDH₂，剩余 2H 中"第二个 H"在 pro-S 位置

两次产物虽然都是 R-CDH₂（化学上相同），但后续用另一个取代基 R' 取代"第二个 H"时：

- 帧 1 得到 (R)-R-CDHR'
- 帧 2 得到 (S)-R-CDHR'

**帧间手性跳变**。geometric CCW 排序通过在每一帧中建立一致的参考系，保证"第一个 H"始终对应同一空间方向，消除了这个问题。

核心机制：选一个参考方向（z 轴），将所有 H 投影到 ⟂z 平面，按 atan2 逆时针角度排序。算法细节见 [`geometric_algorithm.md`](geometric_algorithm.md)。

---

## 何时需要 geometric ordering

关键判据：**取代后剩余是否 ≥2 个 H 形成 prochiral pair**。

| 初始 H 数 | 等价 H 数 | 第一次取代后剩余 H | 能形成 prochiral? | 需要 geometric? |
| :--: | :--: | :--: | :--: | :--: |
| 2 | 2（等价） | 1 | 否 | **否** |
| 3 | 3（等价） | 2 | 是 | **是** |
| 4 | 4（等价） | 3 | 是（进一步取代后剩 2） | **是** |
| 2 | 2（不等价） | N/A | — | 已由化学方法处理 |

### 2H 等价不需要 geometric 的严格论证

```text
2H 等价 → try_order_2h 返回 None
        → R₁ 取代任意 H：产物唯一（剩 1H）
        → R₂ 取代剩余 H：无选择余地，唯一产物
        → 不存在"帧间手性不一致"问题
```

等价 2H 的 geometric CCW 排序不产生化学上的区分，纯属多余计算。当前 `_order_h_sp3` / `_order_h_sp2` 在 `try_order` 返回 `None` 后 fallback 到 `_order_h_geometric` 是不必要的。等价 2H 返回 `sorted(h_indices)` 即可。

### 3+H 等价必须用 geometric

≥3 个等价 H → 取代一个后剩余 ≥2 个 H → 可能形成 prochiral pair → geometric CCW 排序的帧间一致性是正确性的保证。

---

## ref neighbor 选择的问题

### 当前逻辑

```python
ref_neighbor = max(non_h_neighbors, key=lambda n: (n.GetAtomicNum(), -n.GetIdx()))
```

两个问题：

1. **用原子序数而非 CIP rank**：CIP rank 才是化学上正确的取代基优先级标准。原子序数是粗粒度的代理（如 O=8 < F=9，但两者 CIP rank 关系取决于更复杂的上下文）。

2. **用 `n.GetIdx()` 做 tiebreak**：`GetIdx()` 是输入文件中的原始编号，依赖输入顺序，不是真正的规范化编号。虽然对等价 H 场景化学上不影响，但在要求严格规范性的场景下构成了不确定性来源。

### 按杂化类型分析实际需求

| 杂化 | 需要选 ref neighbor 的场景 | 非 H 邻居数 | 是否容易选 |
| :--: | :-- | :--: | :--: |
| sp2 | 无。2H 由 Z/E 或 allene 处理，≥3H 几乎不存在于 sp2 | — | N/A |
| sp3 | 3H 等价（R-CH₃）或 4H 等价（CH₄） | 1 或 0 | 最多 1 个，直接选 |
| sp3d | eq 3H（ax 取代基不同） | 2（都是 ax） | **应显式用 ax 对定 z⁺** |
| sp3d2 | 6H eq 4H | 0 | **应显式用 trans pair 定 z⁺**（3H fac AAA / 5H 不需 geometric） |

结论：**只有 sp3 需要选 ref neighbor 且通常 ≤1 个候选**。sp3d/sp3d2 应当显式用轴向/trans pair 确定 z 轴方向，而非依赖 max-Z 自动选择。

由此可以彻底消除 `max(non_h_neighbors, key=lambda n: (n.GetAtomicNum(), -n.GetIdx()))` 这个逻辑。sp3 的 0 个非 H 邻居走 Case A（等价 H 中按编号选参考），1 个非 H 邻居直接取那一个。不存在多个非 H 邻居需要选、需要 tiebreak 的场景。

---

## `len(h_indices) == 1` 不应在 geometric 内部处理

当前代码：

```python
def _order_h_geometric(mol, center_idx, h_indices):
    if len(h_indices) <= 1:
        return list(h_indices)
    ...
```

将单 H 的处理放在 geometric 内部**违背了职责分离原则**。`_order_h_geometric` 的语义是"用几何投影排序多个 H"，对 1 个 H 做"排序"本身没有意义——调用方就不应该为单 H 调用此函数。

正确的做法是：**所有调用方在调用前检查 `len(h_indices) <= 1`**，单 H 直接返回，只有 ≥2 H 时才进入 geometric。这属于调用方的逻辑（"我有几个 H 需要排序？"），而非 geometric 的内部逻辑。

---

## SP3D 场景细化

### eq 2H

已由 `_order_sp3d_equatorial_2h` 化学处理。等价时无需 geometric。

### eq 3H（当前缺失）

```text
      NH1(ax)  CIP rank 大
       │
  H ───●─── H(eq)    3 个 eq H 全是 H
      ╱  ╲
  H ╱      NH2(ax)  CIP rank 小
```

当 NH1 ≠ NH2（CIP rank 不同）时，3 个 eq H 不等价。应当：

1. 由轴向取代基 CIP rank 确定 z⁺ 方向（低→高，与 eq 2H 一致）
2. 3 个 eq H 投影到 ⟂z 平面，atan2 CCW 排序

当 NH1 与 NH2 CIP rank **相同**时，z⁺ 无法由化学确定。此时轴向两端为等价重原子，需用 `_CanonicalOrder`（InChI /N: 中的规范序号）定方向：规范序号小的 → 大的为 z⁺。这与 Case A 中用 `min(h_indices)` 选参考 H 同理——等价中任选，仅需确定性。

当前代码直接 fallback 到 generic geometric（max-Z 选 ref），没有利用轴向的方向性信息。

### ax 等价 2H

无需 geometric（与其他等价 2H 一致）。

---

## SP3D2 场景细化

### 3H fac AAA

三个 H 各 trans 到一个 CIP rank 相同的 X。所有 H 等价。需要 geometric?

```text
取代 H_a → 剩 2H cis → 但 2H 看起来等价？
关键：H_a 氘代后，分子对称性被打破，剩余 2H 不再等价。
```

验证：在正八面体 fac 构型（e.g. H 在 +x/+y/+z, X 在 -x/-y/-z）中，H_b 的 cis square 为 {D, X_a, X_c, H_c}。四个原子在 ⟂(H_b-X_b) 平面中排列：

```text
        H_c (+z)
         │
X_a (-x)─●─ D (+x)
         │
        X_c (-z)
```

对角线为 (D, X_a) 和 (H_c, X_c)——而非 (D, H_c) 和 (X_a, X_c)。**对角线两端 rank 天然不同**（H/D vs X），Step 2 必然通过。Step 3、4 可正常完成。

处理方式（与 mer 分支平行）：

```text
fac AAA:  min-idx H 排第一（等价中任选）
          + 氘代该 H
          + 剩余 2H cis → _order_sp3d2_cis_2h
          (cis-2H 失败时 → 剩余 2H 任意序)
```

**不需要 geometric。**

### 4H non-H trans（ax 等价）

两个非 H 互为 trans，4 个 H 在赤道正方形上。以 trans 非 H 对为 z 轴做 CCW 排序。

当两个非 H 的 CIP rank **相同**时，z⁺ 无法由化学确定——需用 `_CanonicalOrder` 定方向（规范序号小的 → 大的为 z⁺）。受影响的场景：SP3D ax 等价 + eq 3H、SP3D2 4H non-H trans ax 等价。

注意这与此前要删除的 `max(non_h_neighbors, key=lambda n: -n.GetIdx())` 有本质区别：旧逻辑是在**化学不等价**候选之间用 idx 强行选；新逻辑是在**化学等价**候选之间用规范序号保确定性——与 Case A 的 `min(h_indices)` 同类。

### 5H（1 非 H + 5 H）

trans 对：1×H-X + 2×H-H。H-X 的 H 唯一（trans 非 H）。

非 H 恰好在 trans 位置，垂直于赤道平面。应当显式以该 trans 对定义 z⁺，对 eq 4H 做 CCW 排序。当前代码 max-Z 选 ref → 恰好选到该非 H → 巧合正确，但代码意图不明确。

### 6H（全 H）

3 个 H-H trans 对。所有 H 等价。ax trans pair 的 2H 等价 → 无需 geometric。eq 4H 需要 geometric：

1. 选一个 trans pair（确定性选择）作为 ax
2. ax 方向 → ⟂z 平面 → eq 4H CCW 排序

当前代码对 ax 2H 也调了 geometric（多余），eq 4H 走 Case A（手动选 min-idx H 做 ref），结果对但逻辑绕。

---

## max-Z 选 ref 逻辑可以整体删除

由上文分析可知，`max(non_h_neighbors, key=lambda n: (n.GetAtomicNum(), -n.GetIdx()))` 在所有实际场景中都没有真正的选择需求：

- **sp3**：geometric 被需要时最多 1 个非 H 邻居 → 直接取即可
- **sp3d/sp3d2**：应显式用轴向/trans pair 定 z⁺ → 不该走这条路径
- **无重原子邻居**：走 Case A（等价 H 中按编号选参考）→ 不需要选

因此不仅是用 `_CIPRank` 替换 `GetAtomicNum()` 的问题——**整个 `max` 选择逻辑都可以删除**，包括其中的 idx tiebreak。sp3 的 Case B 改为直接取唯一的非 H 邻居（`non_h_neighbors[0]`）即可。

---

## 改进要点总结

1. **等价 2H 跳过 geometric**：`_order_h_sp3` / `_order_h_sp2` 中 `try_order` 返回 `None` 时返回 `sorted(h_indices)`，无需 fallback

2. **单 H 在调用方处理**：所有 `len(h_indices) <= 1` 的判断上移到各分发函数入口，`_order_h_geometric` 假设输入 `len(h_indices) >= 2`

3. **移除 ref neighbor 的 idx tiebreak 逻辑**：`max(non_h_neighbors, key=lambda n: (n.GetAtomicNum(), -n.GetIdx()))` 没有实际使用场景——sp3 最多 1 个非 H 邻居无需选，sp3d/sp3d2 应显式用轴向/trans pair 定 z⁺。该逻辑可直接删除

4. **SP3D eq 3H**：新增显式处理，以轴向 CIP rank 定 z⁺，CCW 排序 eq H

5. **SP3D2 3H fac AAA**：改为 min-idx H 排第一 + 氘代 + cis-2H（与 mer 分支结构平行），不需要 geometric

6. **SP3D2 5H / 6H**：显式以 trans pair 定 z⁺，而非依赖 max-Z 自动选择

7. **SP3D / SP3D2 等价 2H**：跳过 geometric，与此规则统一
