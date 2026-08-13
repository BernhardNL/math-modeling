#!/usr/bin/env python3
"""问题1 训练：51 维手工特征 + 逻辑回归 + 贝叶斯最小风险阈值选择。

完整流程：
    构造正负样本 → 特征提取 → HOG-PCA + Z-score → L2 逻辑回归（网格搜索 C）
    → 贝叶斯阈值选择（λ 扫描）→ 评估 + 特征重要性 → 保存模型

Usage:
    python problem1/train.py

产出（models/）：
    logistic_regression.pkl   逻辑回归（L2 正则）
    standard_scaler.pkl       特征标准化器（仅用训练集拟合）
    hog_pca.pkl               HOG 6084→15 维 PCA
    threshold_info.json       贝叶斯阈值 τ 与代价比扫描记录
"""

import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    LABELS_TRAIN, IMAGES_TRAIN, MODELS_DIR, FIGURES_DIR,
    HOG_PCA_COMPONENTS, RANDOM_SEED,
)
from src.data_loader import load_all_labels
from src.dataset import build_problem1_dataset, split_train_val
from src.features import extract_features_batch, fit_hog_pca, fit_standard_scaler
from src.classifier import (
    train_logistic_regression, select_bayesian_threshold,
    evaluate_classifier, save_model,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(RANDOM_SEED)

# 特征向量结构: [glcm(4), color(12), sobel(2), hog_raw(6084), lbp(18)]
# hog_raw 位于中间、LBP 在尾部，拼接 51 维时以切片定位这两段。
GLCM_DIM, COLOR_DIM, SOBEL_DIM, LBP_DIM = 4, 12, 2, 18
HOG_RAW_START = GLCM_DIM + COLOR_DIM + SOBEL_DIM
HOG_RAW_END = -LBP_DIM

FEATURE_NAMES = (
    ["GLCM_contrast", "GLCM_homogeneity", "GLCM_energy", "GLCM_correlation"]
    + ["H_mu", "H_sigma", "H_skew", "H_kurt",
       "S_mu", "S_sigma", "S_skew", "S_kurt",
       "V_mu", "V_sigma", "V_skew", "V_kurt"]
    + ["Sobel_edge_density", "Sobel_mean_grad"]
    + [f"HOG_PC{i+1}" for i in range(HOG_PCA_COMPONENTS)]
    + [f"LBP_{i}" for i in range(LBP_DIM)]
)


def build_features_split(raw_feats, hog_pca):
    """把 (N, 51+6072) 原始特征拼接为 51 维：HOG 段过 PCA，其余段直连。"""
    hog_pca_feat = hog_pca.transform(raw_feats[:, HOG_RAW_START:HOG_RAW_END])
    return np.concatenate([
        raw_feats[:, :HOG_RAW_START],
        hog_pca_feat,
        raw_feats[:, HOG_RAW_END:],
    ], axis=1)


def plot_roc_curve(y_true, y_score):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Problem 1 (Image-level Classification)")
    plt.legend()
    plt.grid(alpha=0.3)
    save_path = FIGURES_DIR / "roc_curve_problem1.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ROC 曲线已保存至 {save_path}")


def main():
    print("=" * 60)
    print("  问题1 训练：51维特征 + 逻辑回归 + 贝叶斯阈值")
    print("=" * 60)

    # ── 数据：按图划分，再在每张图内采样 ──
    labels = load_all_labels(LABELS_TRAIN)
    all_ids = list(labels.keys())
    train_ids, val_ids = split_train_val(all_ids, labels, val_ratio=0.20)
    print(f"\n  训练图: {len(train_ids)}, 验证图: {len(val_ids)}")

    # neg_per_image=2：有损图补采 2 个非标注区负样本，平衡正负且贴近真实分布
    # （真负样本是「干净集装箱图」，但题目只给了有损图，故从无标注区域采样近似）。
    X_train_img, y_train, _, _ = build_problem1_dataset(
        train_ids, labels, IMAGES_TRAIN, neg_per_image=2, verbose=True)
    X_val_img, y_val, _, _ = build_problem1_dataset(
        val_ids, labels, IMAGES_TRAIN, neg_per_image=2, verbose=True)

    # ── 特征提取（约 10 分钟，HOG 6084 维是主要开销）──
    print(f"\n>>> 特征提取 — 训练集 ({len(X_train_img)} 张)...")
    feats_train_raw = extract_features_batch(X_train_img, desc="训练集特征")
    print(f">>> 特征提取 — 验证集 ({len(X_val_img)} 张)...")
    feats_val_raw = extract_features_batch(X_val_img, desc="验证集特征")

    # ── HOG-PCA + Z-score：只在训练集上拟合，防止验证信息泄漏 ──
    # HOG 6084 维中相邻 cell 高度相关；PCA 至 15 维保留判别信息同时降噪。
    hog_pca = fit_hog_pca(feats_train_raw[:, HOG_RAW_START:HOG_RAW_END],
                          n_components=HOG_PCA_COMPONENTS)
    X_train_feat = build_features_split(feats_train_raw, hog_pca)
    X_val_feat = build_features_split(feats_val_raw, hog_pca)
    print(f"  拼接后特征维度: {X_train_feat.shape[1]}")

    scaler = fit_standard_scaler(X_train_feat)
    X_train_scaled = scaler.transform(X_train_feat)
    X_val_scaled = scaler.transform(X_val_feat)

    # ── 逻辑回归：网格搜索 L2 系数 C，按验证集 F2 选择 ──
    # 用 F2（β=2）而非 accuracy 选 C：漏检代价是误报代价的 2 倍，
    # F2 给 Recall 4 倍权重，选出的 C 更符合港口场景的代价结构。
    print("\n>>> 训练逻辑回归（网格搜索 L2 正则化系数 C）...")
    from sklearn.metrics import fbeta_score
    best_f2, best_clf, best_C = 0.0, None, None
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = train_logistic_regression(X_train_scaled, y_train, C=C)
        y_val_pred = clf.predict(X_val_scaled)
        f2 = fbeta_score(y_val, y_val_pred, beta=2)
        print(f"    C={C:6.4f}  →  train_acc={clf.score(X_train_scaled, y_train):.4f},  "
              f"val_acc={clf.score(X_val_scaled, y_val):.4f},  val_f2={f2:.4f}")
        if f2 > best_f2:
            best_f2, best_clf, best_C = f2, clf, C
    print(f"\n  最佳 C = {best_C}")
    clf = best_clf

    # ── 贝叶斯最小风险阈值：τ = λ/(1+λ)，λ = C_FP/C_FN ──
    # λ=0.5 意味着「放行一个破损箱的损失 = 2 次误报复核成本」；
    # 扫描多个 λ 并按 F2 选优，同时保留全表供论文/分析引用。
    print("\n>>> 贝叶斯最小风险决策 — 代价比 λ 扫描...")
    scores_val = clf.predict_proba(X_val_scaled)[:, 1]
    bayes_info = select_bayesian_threshold(y_val, scores_val)
    tau_bayes = bayes_info["threshold_bayes"]
    lam_best = bayes_info["best_lambda"]
    print(f"    最优代价比 λ* = {lam_best:.4f} "
          f"(漏检代价 = {1/lam_best:.1f}× 误报代价)")
    print(f"    最优阈值 τ_Bayes = λ/(1+λ) = {tau_bayes:.4f}")
    print(f"    {'λ':>8s}  {'τ':>8s}  {'FPR':>7s}  {'FNR':>7s}  {'F2':>7s}  {'Acc':>7s}")
    for s in bayes_info["sweep"]:
        marker = "  <-- F2最优" if s["lambda"] == lam_best else ""
        print(f"    {s['lambda']:>8.4f}  {s['tau']:>8.4f}  {s['fpr']:>7.4f}  "
              f"{s['fnr']:>7.4f}  {s['f2']:>7.4f}  {s['acc']:>7.4f}{marker}")

    # ── 评估 ──
    print(f"\n>>> 评估（贝叶斯 τ={tau_bayes:.4f}）")
    y_val_pred = (scores_val >= tau_bayes).astype(int)
    metrics_val = evaluate_classifier(y_val, y_val_pred, scores_val, tau_bayes)
    for k in ("accuracy", "precision", "recall", "f1", "f2", "auc"):
        print(f"    {k:12s}: {metrics_val[k]:.4f}")
    cm = metrics_val["confusion_matrix"]
    print(f"    Confusion: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # ── 特征重要性（|w| 排序）──
    print("\n>>> 特征重要性 (|w| Top-8):")
    w = clf.coef_.ravel()
    for rank, idx in enumerate(np.argsort(np.abs(w))[::-1][:8]):
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"dim_{idx}"
        print(f"    #{rank+1}: {name:25s}  |w|={abs(w[idx]):.4f}")

    # ── 保存 ──
    print("\n>>> 保存模型...")
    save_model(clf, "logistic_regression.pkl")
    save_model(scaler, "standard_scaler.pkl")
    save_model(hog_pca, "hog_pca.pkl")
    with open(MODELS_DIR / "threshold_info.json", "w") as f:
        json.dump({
            "bayesian_threshold": float(tau_bayes),
            "bayesian_lambda": float(lam_best),
            "bayesian_f2": float(bayes_info["f2_bayes"]),
            "cost_ratio_sweep": [
                {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                 for k, v in s.items()}
                for s in bayes_info["sweep"]
            ],
        }, f, indent=2)

    plot_roc_curve(y_val, scores_val)
    print("\n训练完成。推理：python problem1/predict.py <image>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
