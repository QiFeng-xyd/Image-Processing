# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:10:48 2026

@author: 徐逸东
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path: str) -> np.ndarray:
    """加载图像并转换为 float64 RGB 数组"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
    return img.astype(np.float64)

img = load_image("./Nikki_image.jpg")
H, W, _ = img.shape     #Height, Width


def compute_energy(img: np.ndarray) -> np.ndarray:
    """
    计算每像素的梯度能量 (e1 energy)
    输入: img  shape (H, W, 3) float64
    输出: energy shape (H, W) float64
    """
    energy = np.zeros(img.shape[:2], dtype=np.float64)  #img.shape[:2]取前两个元素,即H,W

    for c in range(3):           # 对 R, G, B 三通道分别计算
        channel = img[:, :, c]

        # 水平方向梯度 (Sobel x)
        dx = cv2.Sobel(channel, cv2.CV_64F,
                       dx=1, dy=0, ksize=3)
        # 垂直方向梯度 (Sobel y)
        dy = cv2.Sobel(channel, cv2.CV_64F,
                       dx=0, dy=1, ksize=3)

        energy += np.abs(dx) + np.abs(dy)

    return energy

def compute_cumulative_energy(energy: np.ndarray) -> np.ndarray:
    """
    动态规划计算垂直 Seam 的累积最小能量矩阵
    输入: energy  (H, W)
    输出: M       (H, W)  每格存储到达该点的最低路径代价
    """
    H, W = energy.shape
    M = energy.copy()

    for i in range(1, H):
        # 三列偏移：左、中、右邻居
        # 将上一行（i-1 行）的数组向右滚动 1 列
        left   = np.roll(M[i-1],  1);  left[0]  = float('inf')  
        # 将上一行数组向左滚动 1 列
        right  = np.roll(M[i-1], -1);  right[-1] = float('inf')
        center = M[i-1]

        # 取三个方向的最小值，加上当前格能量
        M[i] += np.minimum(np.minimum(left, center), right)

    return M

# 从底行 M 值最小的位置出发，每行向上选择三个相邻格中 M 最小的， 记录每行的列索引，形成长度为 H 的整数数组——即一条完整的 Seam。
def find_seam(M: np.ndarray) -> np.ndarray:
    """
    回溯累积能量矩阵，找出能量最小的垂直 Seam
    输出: seam  shape (H,)  每行对应的列索引
    """
    H, W = M.shape
    seam = np.zeros(H, dtype=np.int32)

    # 从最后一行开始，找能量最小的列
    seam[-1] = np.argmin(M[-1])

    for i in range(H - 2, -1, -1):     # 从倒数第二行向上
        j = seam[i + 1]              # 下一行的列位置

        # 三个候选列（注意边界处理）
        lo = max(j - 1, 0)
        hi = min(j + 2, W)

        # 在窗口内找 M 最小的偏移
        offset = np.argmin(M[i, lo:hi])
        seam[i] = lo + offset

    return seam

# ── 可视化 Seam ────────────────────────────────
def visualize_seam(img, seam):
    vis = img.copy().astype(np.uint8)
    for i, j in enumerate(seam):
        vis[i, j] = [255, 0, 0]   # 红色标注 Seam 路径
    return vis


def remove_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """
    从图像中删除一条垂直 Seam
    输入: img (H, W, 3),  seam (H,) 每行的列索引
    输出: (H, W-1, 3) 的新图像
    """
    H, W, C = img.shape
    output = np.zeros((H, W - 1, C), dtype=img.dtype)

    for i in range(H):
        j = seam[i]
        # 拼接：Seam 左边 + Seam 右边（跳过 Seam 列）
        output[i] = np.concatenate([img[i, :j],
                                     img[i, j+1:]], axis=0)
    return output


# ── 向量化版（更快）──────────────────────────────
def remove_seam_fast(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    H, W, C = img.shape
    # 构建掩码：每行标记要保留的列
    mask = np.ones((H, W), dtype=bool)
    mask[np.arange(H), seam] = False        # 将 Seam 位置设为 False

    # 扩展到三通道并重塑
    mask3 = np.stack([mask] * C, axis=-1)
    return img[mask3].reshape(H, W - 1, C)  # img[mask3]：用三维掩码索引原图像，只保留掩码为 True 的像素，返回一个一维数组（所有保留像素按顺序排列）.reshape(H, W - 1, C)：将一维数组重塑为 (H, W-1, 3) 的图像形状，得到最终结果。


def seam_carving(
    img: np.ndarray,
    n_cols: int = 0,
    n_rows: int = 0,
    verbose: bool = True
) -> np.ndarray:
    """
    完整 Seam Carving 流程
    n_cols: 要减少的宽度（列）像素数
    n_rows: 要减少的高度（行）像素数
    """
    result = img.copy()

    # ── 1. 缩减宽度（删除垂直 Seam）─────────────────
    for k in range(n_cols):
        energy = compute_energy(result)
        M      = compute_cumulative_energy(energy)
        seam   = find_seam(M)
        result = remove_seam_fast(result, seam)


    # ── 2. 缩减高度（转置后删除垂直 Seam 再转回）─────
    if n_rows > 0:
        result = result.transpose(1, 0, 2)        # (W, H, C)
        for k in range(n_rows):
            energy = compute_energy(result)
            M      = compute_cumulative_energy(energy)
            seam   = find_seam(M)
            result = remove_seam_fast(result, seam)
        result = result.transpose(1, 0, 2)        # (H', W', C)

    return result


# ════════════════════════════════════════════
# ── 主程序入口 ────────────────────────────────
# ════════════════════════════════════════════
if __name__ == "__main__":
    # 加载图像
    img = load_image("./Nikki_image.jpg")
    print(f"原始尺寸: {img.shape[1]}×{img.shape[0]}")

    # 执行 Seam Carving（宽度减 150px，高度减 80px）
    result = seam_carving(img, n_cols=233, n_rows=108)
    print(f"缩放后: {result.shape[1]}×{result.shape[0]}")

    # 对比显示
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img.astype(np.uint8)); axes[0].set_title("原始图像")
    axes[1].imshow(result.astype(np.uint8)); axes[1].set_title("Seam Carving 结果")
    plt.tight_layout(); plt.show()

    # 保存结果
    out = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_seam_carved.jpg", out, 
            [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("✓ 已保存: output_seam_carved.jpg")












