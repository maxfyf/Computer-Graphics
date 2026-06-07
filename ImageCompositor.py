"""基于马尔科夫随机场的光照一致图像合成方法"""
import numpy as np
from collections import Counter
from typing import Tuple
                          
# RGB与Lab色彩空间转换器 (基于sRGB和D65白点)
class ColorSpaceConverter:
    Xn, Yn, Zn = 0.95047, 1.00000, 0.95583
    
    # sRGB到XYZ的转换矩阵 (D65)
    RGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    # XYZ到sRGB的转换矩阵 (D65)
    XYZ_TO_RGB = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    @classmethod
    def rgb_to_lab(cls, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将RGB图像转换为Lab色彩空间

        参数:
            rgb: RGB图像, shape (H, W, 3), 范围 [0, 1]
        返回:
            L: 亮度分量, shape (H, W), 范围 0~100
            a: 绿-红分量, shape (H, W), 范围约 -128~127
            b: 蓝-黄分量, shape (H, W), 范围约 -128~127
        """
        rgb = np.asarray(rgb)
        original_shape = rgb.shape
        rgb_flat = rgb.reshape(-1, 3)
        
        # RGB -> 线性RGB (Gamma解码)
        linear_mask = rgb_flat <= 0.04045
        rgb_linear = np.empty_like(rgb_flat)
        rgb_linear[linear_mask] = rgb_flat[linear_mask] / 12.92
        rgb_linear[~linear_mask] = ((rgb_flat[~linear_mask] + 0.055) / 1.055) ** 2.4
        
        # 线性RGB -> XYZ
        xyz = rgb_linear @ cls.RGB_TO_XYZ.T
        
        # XYZ -> Lab
        xyz_norm = xyz / np.array([cls.Xn, cls.Yn, cls.Zn])
        
        delta = 6 / 29
        threshold = delta ** 3
        
        f = np.where(xyz_norm > threshold, 
                     xyz_norm ** (1/3),
                     xyz_norm / (3 * delta**2) + 4/29)
        
        L = 116 * f[:, 1] - 16
        a = 500 * (f[:, 0] - f[:, 1])
        b = 200 * (f[:, 1] - f[:, 2])
        
        L = L.reshape(original_shape[:2])
        a = a.reshape(original_shape[:2])
        b = b.reshape(original_shape[:2])
        
        return L, a, b
    
    @classmethod
    def lab_to_rgb(cls, L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        将Lab图像转换回RGB色彩空间
        
        参数:
            L: 亮度分量, shape (H, W), 范围 0~100
            a: 绿-红分量, shape (H, W), 范围约 -128~127
            b: 蓝-黄分量, shape (H, W), 范围约 -128~127
        返回:
            rgb: RGB图像, shape (H, W, 3), 范围 [0, 1]
        """
        L = np.asarray(L)
        a = np.asarray(a)
        b = np.asarray(b)
        original_shape = L.shape
        
        L_flat = L.flatten()
        a_flat = a.flatten()
        b_flat = b.flatten()
        
        # Lab -> XYZ
        delta = 6 / 29
        fY = (L_flat + 16) / 116
        fX = fY + a_flat / 500
        fZ = fY - b_flat / 200
        
        threshold = delta
        Y_norm = np.where(fY > threshold, 
                          fY ** 3,
                          3 * delta**2 * (fY - 4/29))
        X_norm = np.where(fX > threshold,
                          fX ** 3,
                          3 * delta**2 * (fX - 4/29))
        Z_norm = np.where(fZ > threshold,
                          fZ ** 3,
                          3 * delta**2 * (fZ - 4/29))
        
        X = X_norm * cls.Xn
        Y = Y_norm * cls.Yn
        Z = Z_norm * cls.Zn
        
        xyz = np.stack([X, Y, Z], axis=1)
        
        # XYZ -> 线性RGB
        rgb_linear = xyz @ cls.XYZ_TO_RGB.T
        
        # 线性RGB -> RGB (Gamma编码)
        linear_mask = rgb_linear <= 0.0031308
        rgb = np.empty_like(rgb_linear)
        rgb[linear_mask] = rgb_linear[linear_mask] * 12.92
        rgb[~linear_mask] = 1.055 * (rgb_linear[~linear_mask] ** (1/2.4)) - 0.055
        
        rgb = np.clip(rgb, 0, 1)
        rgb = rgb.reshape(*original_shape, 3)
        
        return rgb

# 加权泊松克隆的梯度权重计算器
class GradientWeightCalculator:
    @staticmethod
    def compute_sigma_from_boundary(L_source_boundary: np.ndarray,
                                     L_target_boundary: np.ndarray,
                                     sigma0: float = 25.5) -> float:
        """
        根据合成边界亮度差异的标准差计算参数 σ = max(255 - 3 * σ^D, σ_0)
        
        参数:
            L_source_boundary: 源图像在边界像素点的亮度值数组
            L_target_boundary: 目标图像在边界像素点的亮度值数组
            sigma0: 避免σ过小的常数
        返回:
            sigma: 权重变化范围参数
        """
        diff = L_target_boundary - L_source_boundary
        sigma_D = np.std(diff)
        sigma = max(255 - 3 * sigma_D, sigma0)
        return sigma

    @staticmethod
    def compute_weights(L_source: np.ndarray, 
                        sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算四邻域的梯度保持权重 w_{pq} = exp(-||L_p^S - L_q^S||^2 / (2 * σ^2))
        
        参数:
            L_source: 源图像亮度, shape (H, W)
            sigma: 权重变化范围参数，由狄利克雷边界条件的满足程度决定
        返回:
            w_pq: 四邻域权重数组, shape (H, W, 4) 分别对应上、右、下、左
            w_p: 归一化因子, shape (H, W), 即四个方向权重之和
        """
        H, W = L_source.shape
        w_pq = np.zeros((H, W, 4))
        
        inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
        if H > 1:
            # 上方
            grad_up = L_source[1:, :] - L_source[:-1, :]
            w_pq[1:, :, 0] = np.exp(-grad_up**2 * inv_2sigma2)
            # 下方
            grad_down = L_source[:-1, :] - L_source[1:, :]
            w_pq[:-1, :, 2] = np.exp(-grad_down**2 * inv_2sigma2) 
        if W > 1:
            # 右方
            grad_right = L_source[:, :-1] - L_source[:, 1:]
            w_pq[:, :-1, 1] = np.exp(-grad_right**2 * inv_2sigma2)
            # 左方
            grad_left = L_source[:, 1:] - L_source[:, :-1]
            w_pq[:, 1:, 3] = np.exp(-grad_left**2 * inv_2sigma2)
        
        w_p = np.sum(w_pq, axis=2)
        w_p = np.maximum(w_p, 1e-8)
        
        return w_pq, w_p


# 直方图对齐的光照一致约束
class HistogramAligner:
    @staticmethod
    def compute_histogram_mean(L_boundary: np.ndarray, L_max: int = 255) -> float:
        """
        计算合成边界亮度直方图的均值 M = Σ_{l=1}^{L_max} l * h_l (亮度主轴)
        
        参数:
            L_boundary: 边界像素点的亮度值数组
            L_max: 最大亮度值
        返回:
            mean: 亮度直方图均值
        """
        hist, _ = np.histogram(L_boundary, bins=np.arange(L_max + 2))
        l_values = np.arange(L_max + 1)
        weighted_sum = np.sum(l_values * hist)
        
        total_pixels = np.sum(hist)
        if total_pixels == 0:
            return 0.0
        
        return weighted_sum / total_pixels
    
    @staticmethod
    def compute_k_means(L: np.ndarray, k: int = 6, max_iters: int = 100, tol: float = 1e-4) -> np.ndarray:
        """
        对面部像素点光强L构成的一维浮点数组进行K-means聚类，并返回包含最多元素的簇的所有数据点。
        
        参数:
            L: np.ndarray, 一维浮点型数组
            k: int, 聚类数量
            max_iters: int, 最大迭代次数
            tol: float, 质心变化容忍度，用于判断收敛
        
        返回:
            majority_cluster: np.ndarray, 元素最多的聚簇中的所有数据点
        """
        if L.size <= k:
            return L

        np.random.seed(42)
        centroids = np.zeros(k, dtype=np.float64)
        centroids[0] = L[np.random.randint(len(L))]
        
        for i in range(1, k):
            distances = np.min(np.abs(L[:, np.newaxis] - centroids[:i]), axis=1)
            distances_squared = distances ** 2
            
            if np.sum(distances_squared) == 0:
                remaining_indices = list(set(range(len(L))) - set(np.where(distances == 0)[0]))
                if remaining_indices:
                    centroids[i] = L[np.random.choice(remaining_indices)]
                else:
                    centroids[i] = L[np.random.randint(len(L))]
            else:
                probabilities = distances_squared / np.sum(distances_squared)
                centroids[i] = L[np.random.choice(len(L), p=probabilities)]
        
        for _ in range(max_iters):
            distances = np.abs(L[:, np.newaxis] - centroids)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros(k, dtype=np.float64)
            for i in range(k):
                cluster_points = L[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points)
                else:
                    new_centroids[i] = L[np.random.randint(len(L))]
            
            if np.all(np.abs(new_centroids - centroids) < tol):
                break
            
            centroids = new_centroids
        
        final_distances = np.abs(L[:, np.newaxis] - centroids)
        final_labels = np.argmin(final_distances, axis=1)
        
        label_counts = Counter(final_labels)
        majority_label = max(label_counts, key=lambda i: label_counts[i])
        majority_cluster = L[final_labels == majority_label]
        
        return majority_cluster
    
    @staticmethod
    def compute_prior_difference(L_source: np.ndarray,
                                  L_target: np.ndarray,
                                  boundary_mask: np.ndarray,
                                  M_diff: float) -> np.ndarray:
        """
        计算亮度差值的先验观测值 Y
        对于边界像素: Y = L_p^T - L_p^S
        对于内部像素: Y = M^T - M^S
        
        参数:
            L_source: 源图像亮度, shape (H, W)
            L_target: 目标图像亮度, shape (H, W)
            boundary_mask: 边界掩码, shape (H, W), True表示边界像素
            M_diff: 亮度主轴差 M^T - M^S
        返回:
            Y: 先验观测值数组, shape (H, W)
        """
        Y = np.zeros_like(L_source)
        Y[boundary_mask] = L_target[boundary_mask] - L_source[boundary_mask]
        Y[~boundary_mask] = M_diff
        return Y

# 自适应正则化系数计算器
class AdaptiveWeightCalculator:
    @staticmethod
    def compute_mu(weight_map: np.ndarray,
                   boundary_mask: np.ndarray,
                   foreground_mask: np.ndarray,
                   default_mu: float = 0.5) -> float:
        """
        计算自适应正则化系数 μ = Σ_{p∈Ω, q∈N_p∩Ω} w_{pq} / Σ_{p∈Ω, q∈N_p∩Ω} δ_{pq}
        
        参数:
            weight_map: 梯度权重 w_{pq}, shape (H, W, 4)
            boundary_mask: 边界掩码, shape (H, W), True表示边界像素
            foreground_mask: 前景掩码, shape (H, W), True表示前景像素
            default_mu: 默认μ值
        返回:
            mu: 正则化系数，范围 [0, 1]
        """
        H, W = boundary_mask.shape
        numerator = 0.0
        denominator = 0.0
        
        for i in range(H):
            for j in range(W):
                if not foreground_mask[i, j]:
                    continue
                
                # 上方
                if i > 0 and boundary_mask[i - 1, j]:
                    denominator += 1
                    numerator += weight_map[i, j, 0]
                # 右方
                if j < W - 1 and boundary_mask[i, j + 1]:
                    denominator += 1
                    numerator += weight_map[i, j, 1]
                # 下方
                if i < H - 1 and boundary_mask[i + 1, j]:
                    denominator += 1
                    numerator += weight_map[i, j, 2]
                # 左方
                if j > 0 and boundary_mask[i, j - 1]:
                    denominator += 1
                    numerator += weight_map[i, j, 3]
        
        if denominator == 0:
            return default_mu
        
        mu = numerator / denominator
        return mu


# 基于马尔科夫随机场的光照一致图像合成器
class MRFImageCompositor:            
    def __init__(self, max_iter: int = 100, tolerance: float = 1e-6):
        """
        初始化合成器
        
        参数:
            max_iter: LLGC最大迭代次数
            tolerance: 收敛容差
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        self.color_converter = ColorSpaceConverter()
        self.gradient_calc = GradientWeightCalculator()
        self.hist_aligner = HistogramAligner()
        self.adaptive_calc = AdaptiveWeightCalculator()
    
    def compute_L_smooth_matrix(self, w_pq: np.ndarray, w_p: np.ndarray) -> np.ndarray:
        """
        构建光滑项仿射矩阵 S = D^{-1} W
        
        参数:
            w_pq: 四邻域权重, shape (H, W, 4)
            w_p: 归一化因子, shape (H, W)
        返回:
            S: 仿射矩阵, shape (N, N), N = H * W
        """
        H, W = w_pq.shape[:2]
        N = H * W
        
        from scipy.sparse import lil_matrix, csr_matrix
        W_sparse = lil_matrix((N, N))
        
        def idx(i, j):
            return i * W + j
        
        for i in range(H):
            for j in range(W):
                current_idx = idx(i, j)
                # 上方
                if i > 0:
                    neighbor_idx = idx(i - 1, j)
                    weight = w_pq[i, j, 0] / w_p[i, j]
                    W_sparse[current_idx, neighbor_idx] = weight
                # 右方
                if j < W - 1:
                    neighbor_idx = idx(i, j + 1)
                    weight = w_pq[i, j, 1] / w_p[i, j]
                    W_sparse[current_idx, neighbor_idx] = weight
                # 下方
                if i < H - 1:
                    neighbor_idx = idx(i + 1, j)
                    weight = w_pq[i, j, 2] / w_p[i, j]
                    W_sparse[current_idx, neighbor_idx] = weight
                # 左方
                if j > 0:
                    neighbor_idx = idx(i, j - 1)
                    weight = w_pq[i, j, 3] / w_p[i, j]
                    W_sparse[current_idx, neighbor_idx] = weight
        
        # 转换为CSR格式以提高计算效率
        W_csr = W_sparse.tocsr()
        D_diag = np.array(W_csr.sum(axis=1)).flatten()
        D_diag = np.maximum(D_diag, 1e-8)
        D_inv = np.diag(1.0 / D_diag)
        S = D_inv @ W_csr

        return S
    
    def llgc_iteration(self, L_prev: np.ndarray, S, mu: float, Y: np.ndarray) -> np.ndarray:
        """
        执行一次LLGC迭代 L_p^D(t) = μ_p * S_p * L^D(t-1) + (1 - mu_p) * Y_p
        
        参数:
            L_prev: 上一轮迭代的亮度差值估计值, shape (N,)
            S: 仿射矩阵, shape (N, N)
            mu: 正则化系数
            Y: 先验观测值, shape (N,)
        返回:
            L_new: 更新后的亮度差值, shape (N,)
        """
        S_L = S @ L_prev
        L_new = mu * S_L + (1 - mu) * Y
        return L_new

    def _rgb_samples_to_luminance(self, rgb_samples: np.ndarray) -> np.ndarray:
        """
        将任意形状的 RGB 样本数组转换为扁平的 Lab 亮度分量数组。

        参数:
            rgb_samples: RGB 样本数组, shape 可以是 (N, 3) 或 (H, W, 3)
        返回:
            L_values: 亮度数组, shape (num_samples,)
        """
        rgb_samples = np.asarray(rgb_samples)
        if rgb_samples.size == 0:
            return np.array([], dtype=float)

        if rgb_samples.ndim == 1:
            rgb_samples = rgb_samples.reshape(1, 1, 3)
        elif rgb_samples.ndim == 2:
            if rgb_samples.shape[1] != 3:
                raise ValueError("source_face_samples/target_face_samples 必须为 (N, 3) 或 (H, W, 3) 形式的 RGB 样本数组")
            rgb_samples = rgb_samples.reshape(-1, 1, 3)
        elif rgb_samples.ndim == 3:
            if rgb_samples.shape[2] != 3:
                raise ValueError("RGB 样本数组的最后一维必须为 3")
        else:
            raise ValueError("RGB 样本数组维度必须为 1、2 或 3")

        L_values, _, _ = self.color_converter.rgb_to_lab(rgb_samples)
        return L_values.flatten()

    def compose(self,
                source_rgb: np.ndarray,
                target_rgb: np.ndarray,
                foreground_mask: np.ndarray,
                boundary_mask: np.ndarray,
                source_face_rgb: np.ndarray,
                target_face_rgb: list[np.ndarray],
                optimize_sampling: bool = True,
                apply_k_means: bool = False) -> np.ndarray:
        """
        执行光照一致图像合成
        
        参数:
            source_rgb: 源图像RGB, shape (H, W, 3), 范围 [0, 1]
            target_rgb: 目标图像RGB, shape (H, W, 3), 范围 [0, 1]
            foreground_mask: 前景掩码, shape (H, W), True表示前景像素
            boundary_mask: 边界掩码, shape (H, W), True表示边界像素
            source_face_samples: 源图像中人脸采样像素点的 RGB 数组
            target_face_samples: 目标图像中人脸采样像素点的 RGB 数组
        返回:
            result_rgb: 合成图像RGB, shape (H, W, 3), 范围 [0, 1]
        """
        # RGB -> Lab 转换
        L_source, a_source, b_source = self.color_converter.rgb_to_lab(source_rgb)
        L_target, _, _ = self.color_converter.rgb_to_lab(target_rgb)

        L_source_face_lab = self.color_converter.rgb_to_lab(source_face_rgb)[0].flatten()
        L_target_face_lab = [self.color_converter.rgb_to_lab(target_rgb)[0].flatten() for target_rgb in target_face_rgb]
        # print(f"Average Lightness: target {np.mean(L_target_face_lab[0])}, source {np.mean(L_source_face_lab)}")

        boundary_indices = np.where(boundary_mask)
        L_source_boundary = L_source[boundary_indices]
        L_target_boundary = L_target[boundary_indices]

        # Step 1: 计算亮度差异均值 M^T - M^S
        if optimize_sampling:
            if apply_k_means:
                L_source_face = self.hist_aligner.compute_k_means(L_source_face_lab)
                L_target_face = np.concatenate([self.hist_aligner.compute_k_means(L_target_lab) for L_target_lab in L_target_face_lab])
                M_S = self.hist_aligner.compute_histogram_mean(L_source_face) / L_source_face.size / 100 * 255
                M_T = self.hist_aligner.compute_histogram_mean(L_target_face) / L_target_face.size / 100 * 255
            else:
                L_source_face = L_source_face_lab.flatten()
                L_target_face = np.concatenate([L_target_lab.flatten() for L_target_lab in L_target_face_lab])
                M_S = np.mean(L_source_face) / 100 * 255
                M_T = np.mean(L_target_face) / 100 * 255
        else:
            M_S = self.hist_aligner.compute_histogram_mean(L_source_boundary)
            M_T = self.hist_aligner.compute_histogram_mean(L_target_boundary)
        # print(f"M_T = {M_T}, M_S = {M_S}")
        M_diff = M_T - M_S
        
        # Step 2-3: 计算参数σ
        sigma = self.gradient_calc.compute_sigma_from_boundary(
            L_source_boundary, L_target_boundary
        )
        
        # Step 4: 计算梯度保持权重矩阵 w_pq 和归一化向量 w_p
        w_pq, w_p = self.gradient_calc.compute_weights(L_source, sigma)
        
        # Step 5-6: 计算仿射矩阵 S
        S = self.compute_L_smooth_matrix(w_pq, w_p)
        
        # Step 7: 计算先验观测值 Y
        Y = self.hist_aligner.compute_prior_difference(
            L_source, L_target, boundary_mask, M_diff
        )
        
        # Step 8: 计算自适应正则化系数 μ
        mu = self.adaptive_calc.compute_mu(w_pq, boundary_mask, foreground_mask)
        
        # Step 9: LLGC迭代求解 L^D
        H, W = L_source.shape
        N = H * W
        Y_flat = Y.flatten()
        
        L_D_flat = np.zeros(N)
        
        for iteration in range(self.max_iter):
            L_D_prev = L_D_flat.copy()
            L_D_flat = self.llgc_iteration(L_D_flat, S, mu, Y_flat)
            
            diff_norm = np.linalg.norm(L_D_flat - L_D_prev)
            if diff_norm < self.tolerance:
                print(f"LLGC收敛于第 {iteration} 次迭代")
                break
        
        # Step 10: 计算最终亮度值 L^C = L^D + L^S
        L_D = L_D_flat.reshape(H, W)
        L_composite = L_D + L_source
        
        # 裁剪亮度到有效范围 [0, 100]
        L_composite = np.clip(L_composite, 0, 100)
        
        # Lab -> RGB 转换
        result_rgb = self.color_converter.lab_to_rgb(L_composite, a_source, b_source)
        
        return result_rgb