# 轴手性 H 排序覆盖分析

本文档梳理所有 IUPAC 轴手性类型，分析其是否影响 H 原子排序，以及现有代码的覆盖情况。

## 背景

`_order_h_on_heavy_atom` 对含有 2 个 H 的重原子使用 CIP 方法排序。当重原子是 `=CH2` 且双键 partner 是 sp 碳时，进入联烯/累积烯烃专用路径 `_order_2h_allene`。

核心判断标准：**轴是否穿过这个 2H 中心本身**。

## 轴手性类型全覆盖

### 1. 联烯 / 累积烯烃（C=C=C...C=C）

| 子类型 | sp 碳数 | 末端平面关系 | 判断方法 | 代码路径 |
|--------|---------|-------------|---------|---------|
| 联烯（如 H2C=C=CHF） | 1（奇） | 垂直 | 投影有符号角度 R_a/S_a | `_order_2h_allene` odd |
| 丁三烯（如 H2C=C=C=CHF） | 2（偶） | 共面 | 二面角 pro-Z/pro-E | `_order_2h_allene` even |
| 更长奇数累积烯 | 奇数 | 垂直 | 同联烯 | 同上 |
| 更长偶数累积烯 | 偶数 | 共面 | 同丁三烯 | 同上 |

状态：✅ 已覆盖

### 2. 杂累积烯烃（C=C=O, C=C=NR, C=C=S 等）

中间 sp 碳只走一步即遇到非 SP 杂原子（O/N/S 为 SP2），sp_count=1（奇数），走轴手性投影路径。

- `H2C=C=NH`：远端 N-H 打破对称，排序成功 ✅
- `H2C=C=O`：远端 O 无取代基，far_subs 为空，返回 None，H 等价 ✅

状态：✅ 自动覆盖（无需额外处理）

### 3. 阻转异构体（Atropisomers，如联苯类 BINAP）

轴是芳环间的 C(sp2)—C(sp2) 单键，所有含 H 中心均在轴的取代基上，不在轴上。

- 轴心原子（芳环 sp2 碳）：通常无 H
- 取代基上的 H 排序按中心杂化类型分派：
  - sp2 `=CH2`：partner 是芳环 sp2 碳（非 SP），走 `_order_2h_sp2`（CIP + 二面角）
  - sp3 `-CH2-`：无双键，走 `_order_2h_sp3`（氘代 CIP）
  - sp3 `-CH3` / 3H+：走几何 CCW 投影
- CIP 排名沿分子图遍历，轴手性信息被自然编码在取代基不等价性中

状态：✅ 由现有 sp2/sp3 路径覆盖

### 4. 螺环化合物（Spiranes）

轴心是 sp3 季碳（4 个键全连非 H 原子），**零个 H**。其余含 H 中心在环上，按各自杂化走对应路径：

- sp3 `-CH2-`（环上亚甲基）→ `_order_2h_sp3`
- sp3 `-CH3` / 3H+ → 几何 CCW
- sp2 `=CH2`（环上若有双键）→ `_order_2h_sp2`

状态：✅ 无需处理（轴上无 H）

### 5. 螺旋手性（Helicenes）

手性来自芳环的螺旋排列，所有含 H 中心在芳环或取代基上：

- sp2 `=CH-`（芳环 CH，1H）→ 直通
- sp2 `=CH2`（取代基）→ `_order_2h_sp2`
- sp3 `-CH2-`（取代基）→ `_order_2h_sp3`
- sp3 `-CH3` / 3H+ → 几何 CCW

状态：✅ 由现有路径覆盖

### 6. 平面手性（Planar chirality，如对环芳烷）

手性来自环平面的不对称取代或桥连，H 中心均在环或桥链上：

- sp2 / sp3 各类 H 中心 → 按杂化走 `_order_2h_sp2` / `_order_2h_sp3` / 几何 CCW

状态：✅ 由现有路径覆盖

### 7. 酰胺/亚胺 C—N 轴手性

轴是 C(sp2)—N(sp2)，N 上通常 0-1 个 H，不触发 2H 排序。若 N 上有取代基含 H 中心，按该中心的杂化走对应路径。

状态：✅ 无需额外处理

## 结论

所有轴手性类型中，**只有联烯/累积烯烃（含杂原子类似物）的轴穿过 2H 中心**，需要专用处理，已通过 `_order_2h_allene` 覆盖。其余类型的 2H 中心均在轴的取代基上，由现有的 sp2/sp3 CIP 路径自动处理。

## 相关文件

- `src/xyz_std/h_ordering.py`：`_order_2h_allene`（联烯/累积烯）、`_order_2h_sp2`（sp2 =CH2）、`_order_2h_sp3`（sp3 -CH2-）
- `tests/unit/test_h_ordering.py`：`TestTryOrder2hAllene`（奇数 sp_count）、`TestTryOrder2hAlleneEven`（偶数 sp_count）
