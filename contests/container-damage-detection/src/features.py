"""特征提取：GLCM纹理 + HSV颜色矩 + Sobel边缘 + HOG梯度直方图 + LBP。"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

from .config import (
    GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS,
    HOG_CELL_SIZE, HOG_BLOCK_SIZE, HOG_N_BINS,
    HOG_PCA_COMPONENTS,
    RANDOM_SEED, MODELS_DIR, IMAGE_SIZE,
)


def extract_glcm_features(gray: np.ndarray) -> np.ndarray:
    """提取 GLCM 纹理特征（4维）。

    Args:
        gray: 灰度图 (H, W)，uint8。

    Returns:
        np.ndarray: shape (4,)，[contrast, homogeneity, energy, correlation]
    """
    gray_quantized = (gray.astype(np.float64) / 256 * GLCM_LEVELS).astype(np.uint8)
    glcm = graycomatrix(
        gray_quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy = graycoprops(glcm, "energy").mean()
    correlation = graycoprops(glcm, "correlation").mean()

    feats = np.array([contrast, homogeneity, energy, correlation], dtype=np.float32)
    # 处理 NaN（全零 GLCM 时相关性可能为 NaN）
    feats = np.nan_to_num(feats, nan=0.0)
    return feats


def extract_color_moments(rgb: np.ndarray) -> np.ndarray:
    """提取 HSV 颜色矩特征（12维）。

    Args:
        rgb: RGB图像 (H, W, 3)，uint8。

    Returns:
        np.ndarray: shape (12,)，[H_mu, H_sigma, H_skew, H_kurt,
                                    S_mu, S_sigma, S_skew, S_kurt,
                                    V_mu, V_sigma, V_skew, V_kurt]
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64) / 255.0
    feats = []
    for ch in range(3):
        data = hsv[:, :, ch].ravel()
        mu = data.mean()
        sigma = data.std()
        # 偏度
        if sigma > 1e-8:
            skew = np.mean(((data - mu) / sigma) ** 3)
        else:
            skew = 0.0
        # 峰度
        if sigma > 1e-8:
            kurt = np.mean(((data - mu) / sigma) ** 4)
        else:
            kurt = 0.0
        feats.extend([mu, sigma, skew, kurt])
    return np.array(feats, dtype=np.float32)


def extract_sobel_features(gray: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """提取 Sobel 边缘统计特征（2维）。

    Args:
        gray: 灰度图 (H, W)，uint8。
        threshold: 边缘判定阈值。

    Returns:
        np.ndarray: shape (2,)，[edge_density, mean_gradient]
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    edge_density = np.mean(magnitude > threshold)
    mean_grad = magnitude.mean()
    return np.array([edge_density, mean_grad], dtype=np.float32)


def extract_hog_features(gray: np.ndarray, pca: PCA = None) -> np.ndarray:
    """提取 HOG 特征，PCA 降维至指定维度。

    Args:
        gray: 灰度图 (H, W)，uint8。
        pca: 已拟合的 PCA 对象。若为 None，返回原始 HOG 特征（不降维）。

    Returns:
        np.ndarray: 若 pca 已拟合则 shape (HOG_PCA_COMPONENTS,)，
                    否则 shape (n_hog_raw,)
    """
    hog_feat = hog(
        gray,
        orientations=HOG_N_BINS,
        pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
        cells_per_block=(HOG_BLOCK_SIZE, HOG_BLOCK_SIZE),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    if pca is not None:
        hog_feat = pca.transform(hog_feat.reshape(1, -1)).ravel()
    return hog_feat.astype(np.float32)


def extract_all_features(rgb: np.ndarray, hog_pca: PCA = None) -> np.ndarray:
    """提取全部 33 维特征。

    Args:
        rgb: RGB图像 (H, W, 3)，uint8。
        hog_pca: 已拟合的 HOG PCA 对象。

    Returns:
        np.ndarray: shape (33,) 若 hog_pca 已拟合，否则 HOG 为原始维度。
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    glcm = extract_glcm_features(gray)     # 4
    color = extract_color_moments(rgb)      # 12
    sobel = extract_sobel_features(gray)    # 2
    hog_feat = extract_hog_features(gray, hog_pca)  # 15 (or raw)

    return np.concatenate([glcm, color, sobel, hog_feat])


def fit_hog_pca(hog_features: np.ndarray, n_components: int = HOG_PCA_COMPONENTS) -> PCA:
    """拟合 HOG 特征的 PCA。

    Args:
        hog_features: shape (n_samples, n_hog_raw)，原始 HOG 特征。
        n_components: 降维目标维度。

    Returns:
        已拟合的 PCA 对象。
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(hog_features)
    return pca


def fit_standard_scaler(features: np.ndarray) -> StandardScaler:
    """拟合并返回 StandardScaler。

    Args:
        features: shape (n_samples, d)，原始特征。

    Returns:
        已拟合的 StandardScaler。
    """
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def extract_features_batch(images: np.ndarray, desc: str = "") -> np.ndarray:
    """批量提取原始特征（GLCM+颜色矩+Sobel+原始HOG），返回 (N, 18+hog_dim)。

    Args:
        images: (N, H, W, 3) RGB uint8。
        desc: tqdm 进度条描述。

    Returns:
        np.ndarray: shape (N, 18 + n_hog_raw)。
    """
    from skimage.feature import hog as _hog
    from tqdm import tqdm

    n = len(images)
    # 预计算 HOG 维度
    gray0 = images[0][:, :, 0] if images[0].ndim == 3 else images[0]
    sample_hog = _hog(
        gray0,
        orientations=HOG_N_BINS,
        pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
        cells_per_block=(HOG_BLOCK_SIZE, HOG_BLOCK_SIZE),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    hog_dim = len(sample_hog)
    lbp_dim = 18  # uniform LBP: n_points + 2
    total_dim = 4 + 12 + 2 + hog_dim + lbp_dim  # glcm + color + sobel + hog + lbp

    features = np.zeros((n, total_dim), dtype=np.float32)

    for i in tqdm(range(n), desc=desc):
        rgb = images[i]
        gray = rgb[:, :, 0] if rgb.ndim == 3 else rgb

        # GLCM (4)
        glcm_feat = extract_glcm_features(gray)
        features[i, 0:4] = glcm_feat

        # 颜色矩 (12)
        if rgb.ndim == 3:
            color_feat = extract_color_moments(rgb)
        else:
            color_feat = np.zeros(12, dtype=np.float32)
        features[i, 4:16] = color_feat

        # Sobel (2)
        sobel_feat = extract_sobel_features(gray)
        features[i, 16:18] = sobel_feat

        # HOG (raw)
        hog_feat = _hog(
            gray,
            orientations=HOG_N_BINS,
            pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
            cells_per_block=(HOG_BLOCK_SIZE, HOG_BLOCK_SIZE),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        features[i, 18:18+hog_dim] = hog_feat.astype(np.float32)

        # LBP (18)
        lbp_feat = extract_lbp_features(gray)
        features[i, 18+hog_dim:18+hog_dim+lbp_dim] = lbp_feat

    return features


def save_preprocessor(obj, name: str, save_dir: Path = None):
    """保存预处理器到 models/ 目录。"""
    if save_dir is None:
        save_dir = MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / name
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  已保存: {path}")


def load_preprocessor(name: str, save_dir: Path = None):
    """从 models/ 目录加载预处理器。"""
    if save_dir is None:
        save_dir = MODELS_DIR
    path = save_dir / name
    with open(path, "rb") as f:
        return pickle.load(f)


# ── LBP 特征 ──

def extract_lbp_features(gray: np.ndarray, radius: int = 2, n_points: int = 16) -> np.ndarray:
    """提取旋转不变 uniform LBP 直方图（维度 = n_points + 2）。

    Args:
        gray: 灰度图 (H, W)，uint8。
        radius: LBP 半径。
        n_points: 邻域采样点数。

    Returns:
        np.ndarray: shape (n_points + 2,)，归一化直方图。
    """
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def compute_iou(box_a: dict, box_b: dict, img_size: int = IMAGE_SIZE) -> float:
    """计算两个 bbox 的 IoU（归一化坐标）。

    Args:
        box_a, box_b: dict with x_center, y_center, width, height (normalized).
        img_size: 图像尺寸（用于 area 计算，实际 IoU 与 img_size 无关但保持一致）。

    Returns:
        float: IoU ∈ [0, 1]。
    """
    def _to_xyxy(box):
        x_c, y_c, w, h = box["x_center"], box["y_center"], box["width"], box["height"]
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return x1, y1, x2, y2

    x1_a, y1_a, x2_a, y2_a = _to_xyxy(box_a)
    x1_b, y1_b, x2_b, y2_b = _to_xyxy(box_b)

    # 交集
    xi1 = max(x1_a, x1_b)
    yi1 = max(y1_a, y1_b)
    xi2 = min(x2_a, x2_b)
    yi2 = min(y2_a, y2_b)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h

    # 并集
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union
