"""逻辑回归分类器 + 贝叶斯最小风险阈值选择 + 评估。"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from pathlib import Path
import pickle

from .config import RANDOM_SEED, MODELS_DIR


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0
) -> LogisticRegression:
    """训练 L2 正则化逻辑回归。

    Args:
        X_train: (n, d) 特征矩阵。
        y_train: (n,) 标签。
        C: 正则化系数（C=1/λ），越小正则化越强。

    Returns:
        已训练的 LogisticRegression。
    """
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def select_bayesian_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    cost_ratios: list[float] = None,
) -> dict:
    """贝叶斯最小风险决策：阈值 = lambda / (1+lambda), lambda = C_FP / C_FN.

    Args:
        y_val: (n,) 验证集标签。
        scores_val: (n,) 逻辑回归预测概率。
        cost_ratios: 待扫描的代价比列表。

    Returns:
        dict with keys: best_lambda, threshold_bayes, f2_bayes,
                        sweep (list of dicts with lambda, tau, fpr, fnr, f2, acc)
    """
    if cost_ratios is None:
        cost_ratios = [0.01, 0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 1.0, 2.0, 4.0, 9.0]

    best_f2 = 0.0
    best_lam = 0.5
    sweep = []

    for lam in cost_ratios:
        tau = lam / (1.0 + lam)
        y_pred = (scores_val >= tau).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
        acc = (tp + tn) / (tp + tn + fp + fn)

        sweep.append({
            "lambda": lam, "tau": tau, "fpr": fpr, "fnr": fnr,
            "f2": f2, "acc": acc, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })

        if f2 > best_f2:
            best_f2 = f2
            best_lam = lam

    tau_best = best_lam / (1.0 + best_lam)

    return {
        "best_lambda": best_lam,
        "threshold_bayes": tau_best,
        "f2_bayes": best_f2,
        "sweep": sweep,
    }


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    threshold: float = None,
) -> dict:
    """评估二分类器。

    Returns:
        dict with accuracy, precision, recall, f1, f2, auc, cm
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    if threshold is not None:
        metrics["threshold"] = threshold
    return metrics


def save_model(obj, name: str):
    """保存模型到 models/ 目录。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  已保存模型: {path}")


def load_model(name: str):
    """从 models/ 目录加载模型。"""
    path = MODELS_DIR / name
    with open(path, "rb") as f:
        return pickle.load(f)
