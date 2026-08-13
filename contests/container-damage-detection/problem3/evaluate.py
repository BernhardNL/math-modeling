#!/usr/bin/env python3
"""问题3 模型评估：对问题1、问题2 的最终模型做系统评估。

评估内容：
  问题1（图像级二分类）：验证集 F2/Precision/Recall/AUC + 混淆矩阵 + 阈值
  问题2 patch 分类（传统路线）：验证集逐类 Prec/Rec/F1 + 混淆矩阵
  问题2 检测（YOLO 最终方案）：验证集 mAP@0.5 / mAP@0.5:0.95 + 逐类 AP

Usage:
    python problem3/evaluate.py              # 全部评估（YOLO 需要 GPU）
    python problem3/evaluate.py --skip-yolo  # 跳过 YOLO（CPU 环境）
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    LABELS_TRAIN, IMAGES_TRAIN, MODELS_DIR, RANDOM_SEED,
    HOG_PCA_COMPONENTS,
)
from src.data_loader import load_all_labels
from src.dataset import build_problem1_dataset, build_problem2_dataset, split_train_val
from src.features import extract_features_batch, fit_hog_pca, fit_standard_scaler
from src.classifier import evaluate_classifier

np.random.seed(RANDOM_SEED)


def evaluate_problem1():
    """问题1：51 维特征 + 逻辑回归 + 贝叶斯阈值。"""
    print("=" * 60)
    print("  问题1 评估：图像级二分类")
    print("=" * 60)

    clf = pickle.load(open(MODELS_DIR / "logistic_regression.pkl", "rb"))
    scaler = pickle.load(open(MODELS_DIR / "standard_scaler.pkl", "rb"))
    hog_pca = pickle.load(open(MODELS_DIR / "hog_pca.pkl", "rb"))
    with open(MODELS_DIR / "threshold_info.json") as f:
        tau = float(json.load(f)["bayesian_threshold"])
    print(f"  模型: LR C={clf.C:.4f}, 贝叶斯阈值 τ={tau:.4f}")

    labels = load_all_labels(LABELS_TRAIN)
    _, val_ids = split_train_val(list(labels.keys()), labels, val_ratio=0.20)
    X_val, y_val, _, _ = build_problem1_dataset(
        val_ids, labels, IMAGES_TRAIN, neg_per_image=2, verbose=True)

    print("  提取特征...")
    raw = extract_features_batch(X_val, desc="验证集特征")
    hog_pca_feat = hog_pca.transform(raw[:, 18:-18])
    feat = np.concatenate([raw[:, :18], hog_pca_feat, raw[:, -18:]], axis=1)
    feat_s = scaler.transform(feat)
    scores = clf.predict_proba(feat_s)[:, 1]

    metrics = evaluate_classifier(y_val, (scores >= tau).astype(int), scores, tau)
    print(f"\n  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F2       : {metrics['f2']:.4f}")
    print(f"  AUC      : {metrics['auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  混淆矩阵 : TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    return metrics


def evaluate_problem2_patch():
    """问题2 传统路线：patch 四分类。"""
    print("\n" + "=" * 60)
    print("  问题2 评估：patch 四分类（传统路线基线）")
    print("=" * 60)

    clf = pickle.load(open(MODELS_DIR / "rf_stage_c.pkl", "rb"))
    scaler = pickle.load(open(MODELS_DIR / "scaler_stage_c.pkl", "rb"))
    hog_pca = pickle.load(open(MODELS_DIR / "hog_pca_stage_c.pkl", "rb"))

    labels = load_all_labels(LABELS_TRAIN)
    _, val_ids = split_train_val(list(labels.keys()), labels, val_ratio=0.20)
    X_val, y_val, _, _ = build_problem2_dataset(
        val_ids, labels, IMAGES_TRAIN, neg_per_image=2, verbose=True,
        neg_mode="hard_mixed")

    print("  提取 72 维特征...")
    from problem2.patch_classify import extract_all_raw_features, build_features
    raw = extract_all_raw_features(X_val, desc="验证集")
    feat = build_features(raw, hog_pca)
    feat_s = scaler.transform(feat)

    y_pred = clf.predict(feat_s)
    acc = (y_pred == y_val).mean()
    names = ["Dent", "Hole", "Rusty", "Background"]
    print(f"\n  Accuracy: {acc:.4f}")
    print(f"  {'类别':>12s}  {'Prec':>8s}  {'Rec':>8s}  {'F1':>8s}  {'n':>5s}")
    for c in range(4):
        tp = int(((y_pred == c) & (y_val == c)).sum())
        fp = int((y_pred == c).sum() - tp)
        fn = int((y_val == c).sum() - tp)
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        print(f"  {names[c]:>12s}  {prec:>8.4f}  {rec:>8.4f}  {f1:>8.4f}  {int((y_val == c).sum()):>5d}")
    return acc


def evaluate_problem2_yolo():
    """问题2 最终方案：YOLO 检测。"""
    print("\n" + "=" * 60)
    print("  问题2 评估：YOLOv8s-P2 检测（最终方案）")
    print("=" * 60)

    from ultralytics import YOLO
    data_yaml = ROOT / "data" / "yolo_dataset" / "data.yaml"
    if not data_yaml.exists():
        print("  跳过：data/yolo_dataset 未构建（运行 problem2/prepare_data.py）")
        return None

    model = YOLO(str(ROOT / "models" / "yolov8s_p2_improved.pt"))
    metrics = model.val(data=str(data_yaml), split="val", device=0)
    print(f"\n  mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    for i, name in enumerate(["Dent", "Hole", "Rusty"]):
        ap50 = metrics.box.ap50[i] if len(metrics.box.ap50) > i else 0
        print(f"  {name}: AP@0.5={ap50:.4f}")
    return metrics


def main():
    skip_yolo = "--skip-yolo" in sys.argv
    evaluate_problem1()
    evaluate_problem2_patch()
    if not skip_yolo:
        evaluate_problem2_yolo()
    print("\n评估完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
