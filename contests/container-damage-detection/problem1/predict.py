#!/usr/bin/env python3
"""问题1 推理：判断集装箱图像是否存在残损。

Usage:
    python problem1/predict.py <image_path> [image_path ...]
    python problem1/predict.py --dir <directory>
    python problem1/predict.py --dir data/raw/dataset/images/test/ --json r.json

依赖模型（models/）：logistic_regression.pkl, standard_scaler.pkl, hog_pca.pkl
阈值默认读取 models/threshold_info.json（贝叶斯 τ），可用 --tau 覆盖。

特征提取在脚本内自包含（不依赖 src/），推理只需 numpy + cv2 + skimage。
"""

import sys
import json
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
GLOBAL_INPUT_SIZE = 448

# ── 与训练时完全一致的参数（改动任一参数都会破坏模型输入分布）──
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, 0.785398, 1.570796, 2.356194]
GLCM_LEVELS = 64
HOG_CELL_SIZE = 32
HOG_BLOCK_SIZE = 2
HOG_N_BINS = 9
HOG_PCA_COMPONENTS = 15
LBP_RADIUS = 2
LBP_N_POINTS = 16
LBP_DIM = LBP_N_POINTS + 2  # uniform 模式 58 bin 归并为 16+2 维直方图


def _extract_glcm(gray):
    # GLCM 用 64 级灰度量化：既控制共生矩阵尺寸（64×64 可算），
    # 又保留足够灰度层次以区分金属表面纹理与锈蚀/凹陷区域。
    from skimage.feature import graycomatrix, graycoprops
    quantized = (gray.astype(np.float64) / 256 * GLCM_LEVELS).astype(np.uint8)
    m = graycomatrix(quantized, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                     levels=GLCM_LEVELS, symmetric=True, normed=True)
    feats = [graycoprops(m, p).mean() for p in ("contrast", "homogeneity", "energy", "correlation")]
    return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0)


def _extract_color_moments(rgb):
    # HSV 而非 RGB：色相/饱和度与亮度解耦，锈蚀的偏红色相在 H/S 上更可分。
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64) / 255.0
    feats = []
    for ch in range(3):
        data = hsv[:, :, ch].ravel()
        mu, sigma = data.mean(), data.std()
        skew = np.mean(((data - mu) / sigma) ** 3) if sigma > 1e-8 else 0.0
        kurt = np.mean(((data - mu) / sigma) ** 4) if sigma > 1e-8 else 0.0
        feats.extend([mu, sigma, skew, kurt])
    return np.array(feats, dtype=np.float32)


def _extract_sobel(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # 边缘密度（mag>50 的像素占比）捕获凹陷/破洞的强边缘，均值梯度捕获整体粗糙度。
    return np.array([np.mean(mag > 50), mag.mean()], dtype=np.float32)


def _extract_hog_raw(gray):
    return hog(gray, orientations=HOG_N_BINS,
               pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
               cells_per_block=(HOG_BLOCK_SIZE, HOG_BLOCK_SIZE),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def _extract_lbp(gray):
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_DIM, range=(0, LBP_DIM), density=True)
    return hist.astype(np.float32)


def extract_features(rgb):
    """从单张 RGB 原图提取 51 维特征（结构与训练时 extract_features_batch 一致）。"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    return (_extract_glcm(gray), _extract_color_moments(rgb),
            _extract_sobel(gray), _extract_hog_raw(gray), _extract_lbp(gray))


def load_models():
    clf = pickle.load(open(MODELS_DIR / "logistic_regression.pkl", "rb"))
    scaler = pickle.load(open(MODELS_DIR / "standard_scaler.pkl", "rb"))
    hog_pca = pickle.load(open(MODELS_DIR / "hog_pca.pkl", "rb"))
    return clf, scaler, hog_pca


def default_threshold():
    """从训练产物读取贝叶斯阈值；缺失时回退 τ=0.3333（λ=0.5）。"""
    info_path = MODELS_DIR / "threshold_info.json"
    if info_path.exists():
        with open(info_path) as f:
            return float(json.load(f)["bayesian_threshold"])
    return 0.3333


def predict_single(rgb, clf, scaler, hog_pca, tau):
    glcm, color, sobel, hog_raw, lbp = extract_features(rgb)
    hog_pca_feat = hog_pca.transform(hog_raw.reshape(1, -1)).ravel()
    feat = np.concatenate([glcm, color, sobel, hog_pca_feat, lbp])  # 51 维
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    prob = clf.predict_proba(feat_scaled)[0, 1]
    return int(prob >= tau), float(prob)


def predict_batch(image_paths, clf, scaler, hog_pca, tau, verbose=True):
    results = []
    for path in image_paths:
        t0 = time.perf_counter()
        img = cv2.imread(str(path))
        if img is None:
            print(f"  ✗ 无法读取: {path}")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (GLOBAL_INPUT_SIZE, GLOBAL_INPUT_SIZE))
        label, prob = predict_single(rgb, clf, scaler, hog_pca, tau)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.append((str(path), label, prob, elapsed_ms))
        if verbose:
            print(f"  {'有损伤' if label == 1 else '无损伤'}  prob={prob:.4f}  "
                  f"{Path(path).name}  ({elapsed_ms:.0f}ms)")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="问题1 推理：判断集装箱图像是否存在残损",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python problem1/predict.py data/raw/dataset/images/train/1001.jpg
  python problem1/predict.py --dir data/raw/dataset/images/test/
  python problem1/predict.py --tau 0.2 1001.jpg    # 降低阈值以减少漏检
        """)
    parser.add_argument("images", nargs="*", help="图像文件路径（可多个）")
    parser.add_argument("--dir", "-d", help="批量处理目录下所有 jpg/png 图像")
    parser.add_argument("--tau", type=float, default=None,
                        help="贝叶斯决策阈值（默认读取 threshold_info.json）")
    parser.add_argument("--json", "-j", help="结果导出 JSON 路径")
    parser.add_argument("--quiet", "-q", action="store_true", help="不逐张打印")
    args = parser.parse_args()

    if not args.images and not args.dir:
        parser.print_help()
        sys.exit(1)

    image_paths = [Path(p) for p in args.images]
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"错误：目录不存在: {args.dir}")
            sys.exit(1)
        image_paths += sorted(dir_path.glob("*.jpg")) + sorted(dir_path.glob("*.png"))
        image_paths = list(dict.fromkeys(image_paths))
    if not image_paths:
        print("未找到任何图像文件。")
        sys.exit(1)

    print(">>> 加载模型...")
    try:
        clf, scaler, hog_pca = load_models()
    except FileNotFoundError as e:
        print(f"错误：模型文件缺失 — {e}")
        print("请先运行 `python problem1/train.py` 训练并保存模型。")
        sys.exit(1)

    tau = args.tau if args.tau is not None else default_threshold()
    print(f"  模型: logistic_regression (C={clf.C:.4f})")
    print(f"  阈值: τ={tau:.4f} (λ={tau/(1-tau):.2f})")
    print(f"  图像数: {len(image_paths)}")

    print("\n>>> 推理...")
    t_start = time.perf_counter()
    results = predict_batch(image_paths, clf, scaler, hog_pca, tau,
                            verbose=not args.quiet)
    total_time = time.perf_counter() - t_start

    n_total = len(results)
    n_pos = sum(1 for _, label, _, _ in results if label == 1)
    avg_ms = np.mean([r[3] for r in results]) if results else 0
    print(f"\n{'='*50}")
    print(f"  总计: {n_total} 张, 有损伤: {n_pos}, 无损伤: {n_total - n_pos}")
    print(f"  平均耗时: {avg_ms:.0f}ms/张, 总耗时: {total_time:.1f}s")

    if args.json:
        output = [{"path": p, "label": l, "prob": pr, "elapsed_ms": round(ms, 1)}
                  for p, l, pr, ms in results]
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  结果已导出至 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
