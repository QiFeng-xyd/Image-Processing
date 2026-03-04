# Seam Carving 图像内容感知缩放

> 一种基于动态规划的智能图像缩放算法，能够在调整图像尺寸时自动保留视觉重要内容。

---

## 项目简介

**Seam Carving**（接缝雕刻）是由 Shai Avidan 和 Ariel Shamir 于 2007 年提出的内容感知图像缩放算法。与传统裁剪或拉伸不同，该算法通过识别并移除图像中"能量最低"的像素路径（Seam），在缩减尺寸的同时最大程度保留图像中的重要内容（人物、边缘、高对比度区域等）。

本项目使用 Python 实现了完整的 Seam Carving 流程，支持同时对图像宽度和高度进行内容感知缩放。

---

## 算法流程

```
原始图像
   │
   ▼
① 计算梯度能量图（Sobel 算子，三通道）
   │
   ▼
② 动态规划计算累积最小能量矩阵 M
   │
   ▼
③ 回溯找出最低能量 Seam（垂直路径）
   │
   ▼
④ 从图像中删除该 Seam（宽度 -1）
   │
   └─── 重复 n_cols 次 ──► 输出宽度缩减后的图像
```

对于**高度缩减**，将图像转置后执行同样流程，再转置回来。

---

## 核心模块说明

### 1. 能量计算 `compute_energy`

对 R、G、B 三个通道分别应用 Sobel 算子，计算水平和垂直方向的梯度，累加绝对值得到每个像素的能量值：

$$e(x, y) = \sum_{c \in \{R,G,B\}} \left( |dx_c| + |dy_c| \right)$$

高能量区域通常对应图像中的边缘和细节，低能量区域为平坦背景。

### 2. 累积能量矩阵 `compute_cumulative_energy`

使用动态规划，从第一行向下逐行累积：

$$M(i, j) = e(i, j) + \min\bigl(M(i-1, j-1),\ M(i-1, j),\ M(i-1, j+1)\bigr)$$

边界处理：越界位置设为 `inf`，防止路径"绕出"图像。

### 3. Seam 回溯 `find_seam`

从最后一行能量最小处出发，逐行向上追踪，每行选择相邻三格中累积能量最小的列，形成一条连通的垂直路径（宽度为 1 像素）。

### 4. Seam 删除 `remove_seam_fast`

利用布尔掩码向量化操作，将 Seam 所在列标记为 `False`，一次性过滤并重塑数组，相比逐行循环速度更快。

### 5. 完整流程 `seam_carving`

| 参数 | 说明 |
|------|------|
| `img` | 输入图像，`np.ndarray`，shape `(H, W, 3)`，float64 |
| `n_cols` | 需要减少的**宽度**像素数 |
| `n_rows` | 需要减少的**高度**像素数 |
| `verbose` | 是否打印进度（预留参数） |

---

## 环境依赖

```bash
pip install numpy opencv-python pillow matplotlib
```

| 库 | 用途 |
|----|------|
| `numpy` | 数组运算、掩码操作 |
| `opencv-python` | 图像读写、Sobel 梯度计算 |
| `Pillow` | 图像辅助处理 |
| `matplotlib` | 结果可视化对比 |

---

## 使用方法

### 基本用法

将目标图像放在脚本同目录下，修改主程序中的参数后运行：

```bash
python seam_carving.py
```

### 参数调整

在 `if __name__ == "__main__":` 部分修改以下内容：

```python
# 加载你的图像
img = load_image("./your_image.jpg")

# 设置缩减量（单位：像素）
result = seam_carving(img, n_cols=233, n_rows=108)
# n_cols：减少宽度 233px
# n_rows：减少高度 108px
```

### 在其他脚本中调用

```python
from seam_carving import load_image, seam_carving
import cv2

img = load_image("input.jpg")
result = seam_carving(img, n_cols=100)  # 仅缩减宽度

out = cv2.cvtColor(result.astype('uint8'), cv2.COLOR_RGB2BGR)
cv2.imwrite("output.jpg", out)
```

### 可视化单条 Seam

```python
img    = load_image("input.jpg")
energy = compute_energy(img)
M      = compute_cumulative_energy(energy)
seam   = find_seam(M)
vis    = visualize_seam(img, seam)   # Seam 以红色标注
```

---

## 输出结果

程序运行后将：

1. 在终端打印原始尺寸与缩放后尺寸
2. 弹出对比窗口（原始图像 vs Seam Carving 结果）
3. 保存结果至 `output_seam_carved.jpg`（JPEG 质量 80）

---

## 性能说明

- 每删除一条 Seam 需重新计算一次能量图，时间复杂度为 $O(H \times W)$
- 删除 $k$ 条 Seam 总复杂度为 $O(k \times H \times W)$
- 对于大尺寸图像或大缩减量，建议使用 GPU 加速版本或考虑批量预计算优化

---

## 参考文献

Avidan, S., & Shamir, A. (2007). **Seam carving for content-aware image resizing**. *ACM Transactions on Graphics*, 26(3), 10.
