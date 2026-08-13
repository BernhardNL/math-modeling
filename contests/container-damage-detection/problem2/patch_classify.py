#!/usr/bin/env python3
"""问题2 传统路线（基线）：72 维增强手工特征 + LightGBM 四分类 + 滑动窗口检测。

这是问题2 的探索性方案：patch 级分类性能尚可，但接入滑动窗口后
因训练-推理分布偏移而失效（Prec/Rec < 15%，详见论文 4.5 节）。
最终检测方案为 YOLOv8s-P2（train_yolo.py），本脚本作为方法论对照保留。

Usage:
    python problem2/patch_classify.py train            # 训练四分类模型
    python problem2/patch_classify.py detect           # 验证集滑动窗口检测
    python problem2/patch_classify.py detect --demo    # 演示模式（20 张）
    python problem2/patch_classify.py classify <img>   # 单张图检测
"""

import sys
import time
import random
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from lightgbm import LGBMClassifier
from skimage.feature import hog as _hog_func

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_all_labels
from src.dataset import build_problem2_dataset, split_train_val
from src.config import (
    LABELS_TRAIN, IMAGES_TRAIN, MODELS_DIR,
    RANDOM_SEED, HOG_PCA_COMPONENTS, PATCH_SIZE,
    NMS_IOU_THRESHOLD, NMS_GLOBAL_IOU_THRESHOLD,
    MAX_DETECTIONS_PER_IMAGE, SEVERITY_WEIGHTS,
    HOG_CELL_SIZE, HOG_BLOCK_SIZE, HOG_N_BINS,
    SLIDING_WINDOW_SCALES,
)
from src.features import (
    extract_glcm_features, extract_color_moments, extract_sobel_features,
    extract_lbp_features, fit_hog_pca, fit_standard_scaler,
)
from src.classifier import save_model, load_model

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_NAMES = {0: "Dent", 1: "Hole", 2: "Rusty", 3: "Background"}


# ══════════════════════════════════════════════════════════════
# 增强颜色特征（问题1 的 51 维体系 + 21 维锈蚀/颜色增强）
# ══════════════════════════════════════════════════════════════

def extract_rgb_stats(rgb):
    """RGB 三通道均值与标准差 — 6维。"""
    feats = []
    for ch in range(3):
        data = rgb[:, :, ch].ravel().astype(np.float64)
        feats.append(data.mean())
        feats.append(data.std())
    return np.array(feats, dtype=np.float32)


def extract_lab_color(rgb):
    """LAB 颜色空间统计 — 6维。

    LAB 而非 HSV：a* 轴直接度量红-绿对抗，锈蚀的红色在 a* 上单向偏移，
    单维 a* 均值即成为锈蚀的最强颜色判据（论文 4.2.2）。
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    feats = []
    for ch in range(3):
        data = lab[:, :, ch].ravel()
        feats.append(data.mean())
        feats.append(data.std())
    return np.array(feats, dtype=np.float32)


def extract_rust_ratio(rgb):
    """锈蚀色彩像素占比 — 1维。

    把「铁锈呈红褐色」的物理先验编码为数值特征：
    R 显著高于 G 和 B（阈值取自锈蚀样本的颜色分布统计）。
    """
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    rust_mask = (r > g * 1.1) & (r > b * 1.2) & (r > 80)
    return np.array([rust_mask.mean()], dtype=np.float32)


def extract_r_histogram(rgb, bins=8):
    """R 通道直方图 — 8维。用分布而非矩统计：锈蚀区 R 分布右偏/双峰，
    均值方差无法区分，直方图可以。"""
    r = rgb[:, :, 0].ravel()
    hist, _ = np.histogram(r, bins=bins, range=(0, 256), density=True)
    return hist.astype(np.float32)


def extract_all_raw_features(images, desc=""):
    """提取全部原始特征模块，返回 dict of (N, dim) arrays。"""
    n = len(images)
    gray0 = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
    hog_dim = len(_hog_func(gray0, orientations=HOG_N_BINS,
                            pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
                            cells_per_block=(HOG_BLOCK_SIZE, HOG_BLOCK_SIZE),
                            block_norm="L2-Hys", feature_vector=True))
    lbp_dim = len(extract_lbp_features(gray0))
    r_hist_dim = len(extract_r_histogram(images[0]))

    feats = {
        "glcm":       np.zeros((n, 4), dtype=np.float32),
        "hsv":        np.zeros((n, 12), dtype=np.float32),
        "sobel":      np.zeros((n, 2), dtype=np.float32),
        "hog_raw":    np.zeros((n, hog_dim), dtype=np.float32),
        "lbp":        np.zeros((n, lbp_dim), dtype=np.float32),
        "rgb":        np.zeros((n, 6), dtype=np.float32),
        "lab":        np.zeros((n, 6), dtype=np.float32),
        "rust_ratio": np.zeros((n, 1), dtype=np.float32),
        "r_hist":     np.zeros((n, r_hist_dim), dtype=np.float32),
    }

    for i in tqdm(range(n), desc=desc):
        rgb = images[i]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        feats["glcm"][i] = extract_glcm_features(gray)
        feats["hsv"][i] = extract_color_moments(rgb) if rgb.ndim == 3 else 0
        feats["sobel"][i] = extract_sobel_features(gray)
        feats["hog_raw"][i] = _hog_func(
            gray, orientations=HOG_N_BINS,
            pixels_per_cell=(HOG_CELL_SIZE, HOG_CELL_SIZE),
            cells_per_block=(HOG_BLOCK_SIZE, HOG_BLOCK_SIZE),
            block_norm="L2-Hys", feature_vector=True,
        ).astype(np.float32)
        feats["lbp"][i] = extract_lbp_features(gray)
        feats["rgb"][i] = extract_rgb_stats(rgb)
        feats["lab"][i] = extract_lab_color(rgb)
        feats["rust_ratio"][i] = extract_rust_ratio(rgb)
        feats["r_hist"][i] = extract_r_histogram(rgb)

    return feats


def build_features(raw, hog_pca):
    """拼接为 72 维特征矩阵。"""
    hog_pca_feat = hog_pca.transform(raw["hog_raw"])
    return np.concatenate([
        raw["glcm"], raw["hsv"], raw["sobel"], hog_pca_feat, raw["lbp"],
        raw["rgb"], raw["lab"], raw["rust_ratio"], raw["r_hist"],
    ], axis=1)


# ══════════════════════════════════════════════════════════════
# NMS
# ══════════════════════════════════════════════════════════════

def compute_iou(box_a, box_b):
    """两个 YOLO 归一化 bbox 的 IoU。"""
    def _xyxy(box):
        xc, yc, w, h = box
        return xc - w/2, yc - h/2, xc + w/2, yc + h/2
    x1a, y1a, x2a, y2a = _xyxy(box_a)
    x1b, y1b, x2b, y2b = _xyxy(box_b)
    xi1, yi1 = max(x1a, x1b), max(y1a, y1b)
    xi2, yi2 = min(x2a, x2b), min(y2a, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = max(0, x2a - x1a) * max(0, y2a - y1a)
    area_b = max(0, x2b - x1b) * max(0, y2b - y1b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections, iou_threshold):
    """非极大值抑制：按置信度降序贪心保留，抑制 IoU 超阈值的重叠框。"""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d[2], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if compute_iou(best[0], d[0]) < iou_threshold]
    return keep


# ══════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════

def train():
    print("=" * 60)
    print("  问题2 传统路线训练：72维增强特征 + LightGBM 四分类")
    print("=" * 60)

    labels = load_all_labels(LABELS_TRAIN)
    all_ids = list(labels.keys())
    train_ids, val_ids = split_train_val(all_ids, labels, val_ratio=0.20)
    print(f"  训练图: {len(train_ids)}, 验证图: {len(val_ids)}")

    # neg_mode="hard_mixed"：负样本中混入高纹理/边界区 patch，
    # 让分类器见过「像损伤但不是损伤」的难负样本，抑制误报（论文 4.3.3）。
    X_train, y_train, _, _ = build_problem2_dataset(
        train_ids, labels, IMAGES_TRAIN, neg_per_image=2, verbose=True,
        neg_mode="hard_mixed")
    X_val, y_val, _, _ = build_problem2_dataset(
        val_ids, labels, IMAGES_TRAIN, neg_per_image=2, verbose=True,
        neg_mode="hard_mixed")

    print(f"\n>>> 提取增强特征（GLCM+HSV+Sobel+HOG+LBP+RGB+LAB+锈色）...")
    raw_tr = extract_all_raw_features(X_train, desc="训练集")
    raw_va = extract_all_raw_features(X_val, desc="验证集")

    hog_pca = fit_hog_pca(raw_tr["hog_raw"], n_components=HOG_PCA_COMPONENTS)
    print(f"  HOG: {raw_tr['hog_raw'].shape[1]} → {HOG_PCA_COMPONENTS}")

    X_tr_feat = build_features(raw_tr, hog_pca)
    X_va_feat = build_features(raw_va, hog_pca)
    print(f"  特征维度: {X_tr_feat.shape[1]} "
          f"(GLCM4+HSV12+Sobel2+HOG_PCA{HOG_PCA_COMPONENTS}+LBP18+RGB6+LAB6+Rust1+Rhist8)")

    scaler = fit_standard_scaler(X_tr_feat)
    X_tr_s = scaler.transform(X_tr_feat)
    X_va_s = scaler.transform(X_va_feat)

    # 网格搜索：类不平衡（Hole 仅 ~950 样本），class_weight="balanced"
    # 让少数类梯度不被多数类淹没。
    print("\n>>> 训练 LightGBM（网格搜索 n_estimators × num_leaves × lr）...")
    best_acc, best_clf = 0.0, None
    for n in [200, 500, 1000]:
        for leaves in [31, 63, 127]:
            for lr in [0.05, 0.1]:
                clf = LGBMClassifier(
                    n_estimators=n, num_leaves=leaves, learning_rate=lr,
                    objective="multiclass", num_class=4,
                    class_weight="balanced", random_state=RANDOM_SEED,
                    verbose=-1)
                clf.fit(X_tr_s, y_train)
                acc = clf.score(X_va_s, y_val)
                print(f"    LGB(n={n}, leaves={leaves:>3d}, lr={lr:.2f}): "
                      f"train={clf.score(X_tr_s, y_train):.4f}  val={acc:.4f}")
                if acc > best_acc:
                    best_acc, best_clf = acc, clf

    print(f"\n  最佳: n={best_clf.n_estimators}, leaves={best_clf.num_leaves}, "
          f"lr={best_clf.learning_rate:.2f}, val_acc={best_acc:.4f}")

    y_pred = best_clf.predict(X_va_s)
    cm = confusion_matrix(y_val, y_pred)
    print(f"\n  验证集混淆矩阵:")
    print(f"              pred_Dent  pred_Hole  pred_Rusty  pred_Bg")
    for i, name in CLASS_NAMES.items():
        print(f"  true_{name:>12s}  {cm[i][0]:>9d}  {cm[i][1]:>9d}  {cm[i][2]:>9d}  {cm[i][3]:>9d}")
    print("\n  各类别性能:")
    for i, name in CLASS_NAMES.items():
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"    {name:>12s}: Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    save_model(best_clf, "rf_stage_c.pkl")
    save_model(hog_pca, "hog_pca_stage_c.pkl")
    save_model(scaler, "scaler_stage_c.pkl")
    print("\n训练完成，模型已保存至 models/")
    return best_acc


# ══════════════════════════════════════════════════════════════
# 滑动窗口检测
# ══════════════════════════════════════════════════════════════

def detect_single_image(rgb, clf, scaler, hog_pca, damage_threshold=0.6):
    """多尺度滑动窗口检测。

    返回 list of (box_yolo, class_id, confidence)。
    damage_threshold：损伤概率和（3 个损伤类之和）低于此值的窗口丢弃。
    """
    h, w = rgb.shape[:2]
    all_detections = []

    for scale in SLIDING_WINDOW_SCALES:
        win, stride = scale["window"], scale["stride"]
        patches, positions = [], []
        for y in range(0, h - win + 1, stride):
            for x in range(0, w - win + 1, stride):
                patch = rgb[y:y+win, x:x+win]
                patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                patches.append(patch)
                positions.append([(x + win/2) / w, (y + win/2) / h, win / w, win / h])

        if not patches:
            continue

        patches_arr = np.array(patches, dtype=np.uint8)
        raw = extract_all_raw_features(patches_arr, desc=f"  scale {win}")
        feats = build_features(raw, hog_pca)
        feats_s = scaler.transform(feats)
        probs = clf.predict_proba(feats_s)
        preds = np.argmax(probs, axis=1)
        damage_scores = probs[:, 0] + probs[:, 1] + probs[:, 2]

        for i in range(len(positions)):
            if preds[i] == 3 or damage_scores[i] < damage_threshold:
                continue
            all_detections.append((positions[i], int(preds[i]), float(probs[i, preds[i]])))

    return all_detections


def detect():
    """在验证集上运行滑动窗口检测并保存 YOLO 格式结果。"""
    use_demo = "--demo" in sys.argv
    print("=" * 60)
    print("  问题2 传统路线检测：多尺度滑动窗口 + LGBM + NMS")
    print("=" * 60)

    clf = load_model("rf_stage_c.pkl")
    hog_pca = load_model("hog_pca_stage_c.pkl")
    scaler = load_model("scaler_stage_c.pkl")

    labels = load_all_labels(LABELS_TRAIN)
    all_ids = list(labels.keys())
    _, val_ids = split_train_val(all_ids, labels, val_ratio=0.20)
    test_ids = val_ids[:20] if use_demo else val_ids
    print(f"  {'Demo ' if use_demo else ''}模式: {len(test_ids)} 张")

    results = {}
    t_start = time.perf_counter()
    for stem in tqdm(test_ids, desc="检测"):
        img = cv2.imread(str(IMAGES_TRAIN / f"{stem}.jpg"))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detect_single_image(rgb, clf, scaler, hog_pca)

        # 两级 NMS：类内抑制同类重复框，再全局抑制跨类重叠框
        # （锈蚀常与凹陷共存，跨类抑制防止同一区域被重复报告）。
        per_class = {0: [], 1: [], 2: []}
        for box, cls, conf in detections:
            per_class[cls].append((box, cls, conf))
        nms_results = []
        for cls in [0, 1, 2]:
            nms_results += nms(per_class[cls], NMS_IOU_THRESHOLD)
        nms_results = nms(nms_results, NMS_GLOBAL_IOU_THRESHOLD)

        # 严重程度 = 面积 × 类别权重；Hole 权重最高（结构完整性威胁最直接）
        nms_results.sort(
            key=lambda d: (d[0][2] * d[0][3]) * SEVERITY_WEIGHTS.get(d[1], 1.0),
            reverse=True)
        results[stem] = nms_results[:MAX_DETECTIONS_PER_IMAGE]

    elapsed = time.perf_counter() - t_start
    print(f"\n  检测完成: {len(results)} 张, 平均 {elapsed/len(results):.1f}s/张")

    output_dir = ROOT / "output" / "stage_c_final_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, dets in results.items():
        lines = [f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {conf:.4f}"
                 for box, cls, conf in dets]
        (output_dir / f"{stem}.txt").write_text("\n".join(lines))
    print(f"  结果已保存至 {output_dir}/")
    return results


def evaluate_detections(results, labels, iou_threshold=0.3):
    """检测评估：同类 bbox IoU ≥ threshold 视为命中。"""
    total_gt = total_det = total_hit = 0
    per_class_gt = {0: 0, 1: 0, 2: 0}
    per_class_hit = {0: 0, 1: 0, 2: 0}

    for stem, dets in results.items():
        gt_boxes = labels.get(stem, [])
        total_gt += len(gt_boxes)
        total_det += len(dets)
        for gt in gt_boxes:
            per_class_gt[gt["class"]] += 1
        for det_box, det_cls, _ in dets:
            for gt in gt_boxes:
                if gt["class"] != det_cls:
                    continue
                gt_box = [gt["x_center"], gt["y_center"], gt["width"], gt["height"]]
                if compute_iou(det_box, gt_box) >= iou_threshold:
                    total_hit += 1
                    per_class_hit[det_cls] += 1
                    break

    prec = total_hit / total_det if total_det > 0 else 0
    rec = total_hit / total_gt if total_gt > 0 else 0
    f2 = 5 * prec * rec / (4 * prec + rec) if (prec + rec) > 0 else 0
    print(f"\n  检测评估（IoU≥{iou_threshold}）:")
    print(f"  GT={total_gt}, 检测={total_det}, 命中={total_hit}")
    print(f"  Precision={prec:.4f}, Recall={rec:.4f}, F2={f2:.4f}")
    return {"precision": prec, "recall": rec, "f2": f2}


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "detect":
        results = detect()
        if "--demo" not in sys.argv:
            labels = load_all_labels(LABELS_TRAIN)
            evaluate_detections(results, labels)
    elif cmd == "classify":
        img_path = sys.argv[2] if len(sys.argv) > 2 else None
        if img_path is None:
            print("Usage: python problem2/patch_classify.py classify <image_path>")
            return 1
        clf = load_model("rf_stage_c.pkl")
        hog_pca = load_model("hog_pca_stage_c.pkl")
        scaler = load_model("scaler_stage_c.pkl")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read: {img_path}")
            return 1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detect_single_image(rgb, clf, scaler, hog_pca)
        print(f"\nDetections ({len(dets)}):")
        for box, cls, conf in sorted(nms(dets, 0.5), key=lambda d: -d[2]):
            print(f"  {CLASS_NAMES.get(cls)}: box=({box[0]:.4f},{box[1]:.4f},"
                  f"{box[2]:.4f},{box[3]:.4f}) conf={conf:.4f}")
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
