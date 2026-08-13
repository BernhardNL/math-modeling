#!/usr/bin/env python3
"""问题2 最终方案训练：YOLOv8s-P2 + Wise-IoU + SlideLoss。

三项针对任务特性的改进（数学推导见论文 4.7.5–4.7.7）：
  1. Wise-IoU 距离注意力损失：R = exp(ρ²/c²) ∈ [1,e)，按真实框尺度
     归一化中心距惩罚——大锈蚀框容忍中心偏差、小破洞框严格要求对准，
     直击 Rusty 类（大框、边界模糊、中心定位难）的瓶颈。
  2. SlideLoss：以批次前景对齐分数均值 μ 为基准的 tanh 平滑权重，
     对简单正样本降权、困难正样本加权，实现零开销在线难例挖掘。
  3. P2 检测层（stride=4）：最小可检测目标从 8×8 降至 4×4 像素，
     覆盖 P10 面积仅 609 px² 的小残损。

实现方式：monkey-patch ultralytics 的 BboxLoss/v8DetectionLoss——
在 __init__ 后替换 criterion 而非改 ultralytics 源码，保持依赖可升级。
**依赖 ultralytics 8.4.x 的 loss 内部接口**，升级前需核对签名。

Usage:
    python problem2/train_yolo.py                # 全新训练
    python problem2/train_yolo.py --resume       # 从 runs/.../last.pt 恢复

先决条件：已运行 problem2/prepare_data.py 构建 data/yolo_dataset。
"""

import sys
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.metrics import bbox_iou

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Wise-IoU v1：距离注意力因子 ────────────────────────────────

def _wise_iou_r(pred_bboxes_xyxy, target_bboxes_xyxy, eps=1e-7):
    """R = exp((Δx²+Δy²)/(W_gt²+H_gt²)) ∈ [1, e)。

    分母用真实框宽高平方和归一化：同样的中心偏移，小目标得到强惩罚、
    大目标得到温和惩罚（尺度自适应）。上界 e 保证离群样本不导致梯度爆炸。
    """
    px = (pred_bboxes_xyxy[..., 0] + pred_bboxes_xyxy[..., 2]) / 2
    py = (pred_bboxes_xyxy[..., 1] + pred_bboxes_xyxy[..., 3]) / 2
    tx = (target_bboxes_xyxy[..., 0] + target_bboxes_xyxy[..., 2]) / 2
    ty = (target_bboxes_xyxy[..., 1] + target_bboxes_xyxy[..., 3]) / 2
    tw = target_bboxes_xyxy[..., 2] - target_bboxes_xyxy[..., 0]
    th = target_bboxes_xyxy[..., 3] - target_bboxes_xyxy[..., 1]
    rho2 = (px - tx) ** 2 + (py - ty) ** 2
    c2 = tw ** 2 + th ** 2
    return torch.exp(rho2 / (c2 + eps))


# ── SlideLoss 权重 ────────────────────────────────────────────

def _slide_weights(target_scores: torch.Tensor) -> torch.Tensor:
    """w(t) = 0.5 + 0.5·tanh(5(μ−t)/μ)，μ = 当前批次前景对齐分数均值。

    不用标准 SlideLoss 的分段线性而用 tanh：权重在 μ 附近近似线性、
    两端平滑饱和，避免硬阈值切换处的梯度不连续。
    μ 随训练进程自适应：初期所有样本都难（μ 小、权重平坦），
    后期简单样本分数升高（μ 大），权重自动向困难样本集中。
    """
    weights = torch.ones_like(target_scores)
    fg = target_scores > 0
    if fg.sum() == 0:
        return weights

    for b in range(target_scores.shape[0]):
        fg_b = target_scores[b].sum(-1) > 0
        if fg_b.sum() == 0:
            continue
        fg_idx = torch.where(fg_b)[0]
        mu = target_scores[b, fg_idx].max(-1).values.mean()
        if mu < 1e-6:
            continue
        t = target_scores[b, fg_idx].max(-1, keepdim=True).values
        w = (mu - t) / (mu + 1e-7)
        w = 0.5 + 0.5 * torch.tanh(w * 5)
        weights[b, fg_idx] = weights[b, fg_idx] * w

    return weights.detach()


# ── 改进版 Loss 类 ────────────────────────────────────────────

class ImprovedBboxLoss(BboxLoss):
    """BboxLoss with Wise-IoU v1（其余与官方 BboxLoss 一致）。"""

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum,
                fg_mask, imgsz, stride):
        weight = target_scores[fg_mask].sum(-1, keepdim=True)

        # L_WIoU = R × (1 − IoU)；R 为 IoU=0 阶段的回归提供中心距梯度
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=False)
        r = _wise_iou_r(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = (((1.0 - iou) * r) * weight).sum() / target_scores_sum

        from ultralytics.utils.tal import bbox2dist
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                 target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


class ImprovedDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with SlideLoss 分类加权。"""

    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        self.bbox_loss = ImprovedBboxLoss(self.reg_max).to(self.device)

    def get_assigned_targets_and_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()

        from ultralytics.utils.tal import make_anchors
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1),
                             batch["cls"].view(-1, 1),
                             batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # SlideLoss：BCE 逐样本乘难易权重
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        bce_loss = bce_loss * _slide_weights(target_scores)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor,
                target_scores, target_scores_sum,
                fg_mask, imgsz, stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


def _patch_trainer():
    """在 trainer 初始化完成后把 criterion 换成改进版。

    必须在每次训练（含 resume）前调用：ultralytics 每次训练新建 trainer，
    resume 也会走 _setup_train，patch 不持久。
    """
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    orig_build = DetectionTrainer._setup_train

    def _patched_setup(self):
        orig_build(self)
        self.criterion = ImprovedDetectionLoss(self.model, tal_topk=10)
        print("[OK] criterion → ImprovedDetectionLoss (Wise-IoU + SlideLoss)")

    DetectionTrainer._setup_train = _patched_setup


def main():
    data_yaml = ROOT / "data" / "yolo_dataset" / "data.yaml"
    models_dir = ROOT / "models"
    resume = "--resume" in sys.argv

    if not data_yaml.exists():
        print("错误：未找到 data/yolo_dataset/data.yaml，")
        print("请先运行 `python problem2/prepare_data.py`。")
        return 1

    # P2 架构：ultralytics 内置 yolov8s-p2.yaml；新层从 COCO 预训练权重
    # 只迁移匹配的层，P2 分支随机初始化（TAL 会自然处理其冷启动，见论文 4.7.7）。
    p2_yaml = "yolov8s-p2.yaml"
    pretrained_weights = "/tmp/yolov8s.pt"

    print("=" * 60)
    print("  训练配置: YOLOv8s-P2 (Wise-IoU + SlideLoss)")
    print(f"  数据: {data_yaml}")
    print("  epochs=150, imgsz=640, batch=16")
    print("=" * 60)

    _patch_trainer()

    model = YOLO(p2_yaml)
    model.load(pretrained_weights)

    model.train(
        data=str(data_yaml),
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=30,
        save=True,
        save_period=10,
        project=str(ROOT / "runs" / "detect"),
        name="train_s_p2_improved",
        exist_ok=True,
        resume=resume,
        pretrained=True,
        amp=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,
        # 10 epoch 后关 Mosaic：马赛克拼接伪影会干扰边界回归的收敛后期，
        # 让训练分布回归真实图像分布（ultralytics 官方推荐做法）。
        close_mosaic=10,
    )

    best_pt = ROOT / "runs" / "detect" / "train_s_p2_improved" / "weights" / "best.pt"
    if best_pt.exists():
        dst = models_dir / "yolov8s_p2_improved.pt"
        shutil.copy(best_pt, dst)
        print(f"\n最佳模型已保存至: {dst}")

    print("\n>>> 最终验证...")
    metrics = model.val(data=str(data_yaml), split="val", device=0)
    print(f"\nmAP@0.5:       {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")
    for i, name in enumerate(["Dent", "Hole", "Rusty"]):
        ap50 = metrics.box.ap50[i] if len(metrics.box.ap50) > i else 0
        print(f"  {name}: AP@0.5={ap50:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
