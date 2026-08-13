"""正负样本构造 + 训练/验证集划分。"""

import random
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

from .config import (
    IMAGES_TRAIN, LABELS_TRAIN, GLOBAL_INPUT_SIZE, PATCH_SIZE,
    IMAGE_SIZE, RANDOM_SEED, NUM_CLASSES, MODELS_DIR,
    SLIDING_WINDOW_SCALES,
)
from .data_loader import parse_yolo_label, bbox_area_pixels


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _is_on_container(patch_bgr: np.ndarray, min_surface_ratio: float = 0.4) -> bool:
    """判断 patch 是否落在集装箱表面（而非天空/地面/杂物）。

    方法：
      1. Otsu 二值化（找出主要前景/背景区域）。
      2. Sobel 边缘密度过滤纯色区域（天空、纯色地面）。
      3. 前景像素占比 ≥ min_surface_ratio。

    Args:
        patch_bgr: BGR patch (H, W, 3)。
        min_surface_ratio: 集装箱表面像素的最低占比。

    Returns:
        bool: True 表示 patch 在集装箱表面上。
    """
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = np.mean(binary > 0)

    # 边缘密度
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    edge_density = np.mean(magnitude > 30)

    # 集装箱表面：有一定边缘纹理（不是纯天空）+ 前景占比足够
    return edge_density > 0.02 and fg_ratio >= min_surface_ratio


def _sample_window_negatives(
    image_bgr: np.ndarray,
    boxes: list[dict],
    num_patches: int,
    patch_size: int,
    img_size: int,
    max_attempts: int = 200,
    iou_threshold: float = 0.1,
) -> list[np.ndarray]:
    """用推理时的滑窗尺度采样负样本，缓解训练/推理分布偏移。

    对照 SLIDING_WINDOW_SCALES 的每个 (window, stride) 滑窗，
    IoU < iou_threshold 且落在集装箱表面的 patch 作为候选，
    从中随机采样 num_patches 个。
    """
    h, w = image_bgr.shape[:2]
    candidates = []

    for scale in SLIDING_WINDOW_SCALES:
        win, stride = scale["window"], scale["stride"]
        for y in range(0, h - win + 1, stride):
            for x in range(0, w - win + 1, stride):
                # 归一化坐标
                patch_box = {
                    "x_center": (x + win / 2) / w,
                    "y_center": (y + win / 2) / h,
                    "width": win / w,
                    "height": win / h,
                }
                # IoU 检查
                max_iou = 0.0
                for box in boxes:
                    iou = _compute_iou_patch(patch_box, box)
                    if iou > max_iou:
                        max_iou = iou
                if max_iou >= iou_threshold:
                    continue

                # 裁剪并 resize
                patch = image_bgr[y:y + win, x:x + win]
                if patch.shape[0] != win or patch.shape[1] != win:
                    continue

                patch_resized = cv2.resize(patch, (patch_size, patch_size))
                if not _is_on_container(patch_resized):
                    continue

                candidates.append(patch_resized)

    if len(candidates) <= num_patches:
        return candidates
    return random.sample(candidates, num_patches)


def _sample_negative_patches(
    image_bgr: np.ndarray,
    boxes: list[dict],
    num_patches: int = 2,
    patch_size: int = PATCH_SIZE,
    img_size: int = IMAGE_SIZE,
    max_attempts: int = 200,
    mode: str = "random",
) -> list[np.ndarray]:
    """从一张图的非 bbox 区域采样干净表面 patch。

    Args:
        image_bgr: 原图 (H, W, 3), BGR。
        boxes: bbox 列表，归一化坐标。
        num_patches: 目标采样数量。
        patch_size: patch 边长（像素）。
        img_size: 图像尺寸。
        max_attempts: 最大尝试次数。
        mode: "random" | "boundary" | "high_texture" | "hard_mixed" | "hard_window"

    Returns:
        list of np.ndarray: BGR patches。
    """
    if mode == "boundary":
        return _sample_boundary_negatives(image_bgr, boxes, num_patches, patch_size, img_size, max_attempts)
    elif mode == "high_texture":
        return _sample_high_texture_negatives(image_bgr, boxes, num_patches, patch_size, img_size, max_attempts)
    elif mode == "hard_mixed":
        half = max(1, num_patches // 2)
        patches = _sample_boundary_negatives(image_bgr, boxes, half, patch_size, img_size, max_attempts)
        patches += _sample_high_texture_negatives(image_bgr, boxes, num_patches - len(patches), patch_size, img_size, max_attempts)
        return patches
    elif mode == "hard_window":
        third = max(1, num_patches // 3)
        patches = _sample_boundary_negatives(image_bgr, boxes, third, patch_size, img_size, max_attempts)
        patches += _sample_high_texture_negatives(image_bgr, boxes, third, patch_size, img_size, max_attempts)
        need = num_patches - len(patches)
        if need > 0:
            patches += _sample_window_negatives(image_bgr, boxes, need, patch_size, img_size, max_attempts)
        return patches
    # default: random
    h, w = image_bgr.shape[:2]
    patches = []

    def _does_overlap(x, y):
        px, py = x / w, y / h
        pw, ph = patch_size / w, patch_size / h
        patch_box = {
            "x_center": px + pw / 2,
            "y_center": py + ph / 2,
            "width": pw,
            "height": ph,
        }
        for box in boxes:
            iou = _compute_iou_patch(patch_box, box)
            if iou > 0:
                return True
        return False

    for _ in range(max_attempts):
        if len(patches) >= num_patches:
            break
        x = random.randint(0, max(0, w - patch_size))
        y = random.randint(0, max(0, h - patch_size))

        if _does_overlap(x, y):
            continue

        patch = image_bgr[y:y + patch_size, x:x + patch_size]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue

        if not _is_on_container(patch):
            continue

        patches.append(patch)

    return patches


def _sample_boundary_negatives(
    image_bgr: np.ndarray,
    boxes: list[dict],
    num_patches: int,
    patch_size: int,
    img_size: int,
    max_attempts: int,
    margin_min: int = 0,
    margin_max: int = 120,
) -> list[np.ndarray]:
    """在 bbox 边界附近采样负样本（IoU=0 但紧贴损伤边缘）。

    采样区域：bbox 外扩 margin_min ~ margin_max 的环形区域。
    """
    h, w = image_bgr.shape[:2]
    patches = []

    def _patch_overlaps_bbox(px, py, pw, ph):
        patch_box = {
            "x_center": px + pw / 2, "y_center": py + ph / 2,
            "width": pw, "height": ph,
        }
        for box in boxes:
            if _compute_iou_patch(patch_box, box) > 0:
                return True
        return False

    def _distance_to_nearest_bbox(px, py, pw, ph):
        """patch 中心到最近 bbox 边界的距离（像素，负值表示内部）。"""
        pcx, pcy = (px + pw / 2) / w, (py + ph / 2) / h
        min_dist = float("inf")
        for box in boxes:
            bx1 = box["x_center"] - box["width"] / 2
            by1 = box["y_center"] - box["height"] / 2
            bx2 = box["x_center"] + box["width"] / 2
            by2 = box["y_center"] + box["height"] / 2
            # patch 中心到 bbox 的有符号距离
            dx = max(bx1 - pcx, 0, pcx - bx2) * w
            dy = max(by1 - pcy, 0, pcy - by2) * h
            dist = np.sqrt(dx**2 + dy**2)
            # 如果在 bbox 内部，距离为负
            if bx1 <= pcx <= bx2 and by1 <= pcy <= by2:
                dist = -dist
            min_dist = min(min_dist, dist)
        return min_dist

    for _ in range(max_attempts):
        if len(patches) >= num_patches:
            break
        x = random.randint(0, max(0, w - patch_size))
        y = random.randint(0, max(0, h - patch_size))

        pw_norm = patch_size / w
        ph_norm = patch_size / h

        # 必须与所有 bbox 的 IoU = 0
        if _patch_overlaps_bbox(x / w, y / h, pw_norm, ph_norm):
            continue

        # 必须在 margin_min ~ margin_max 范围内
        dist = _distance_to_nearest_bbox(x, y, patch_size, patch_size)
        if dist < margin_min or dist > margin_max:
            continue

        patch = image_bgr[y:y + patch_size, x:x + patch_size]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue

        if not _is_on_container(patch):
            continue

        patches.append(patch)

    # 如果边界采样不足，回退到随机采样
    if len(patches) < num_patches:
        remaining = num_patches - len(patches)
        fallback = _sample_random_negatives(image_bgr, boxes, remaining, patch_size, img_size, max_attempts)
        patches += fallback

    return patches


def _sample_random_negatives(
    image_bgr: np.ndarray,
    boxes: list[dict],
    num_patches: int,
    patch_size: int,
    img_size: int,
    max_attempts: int,
) -> list[np.ndarray]:
    """从非 bbox 区域随机采样干净表面 patch（无难例约束）。"""
    h, w = image_bgr.shape[:2]
    patches = []

    def _does_overlap(x, y):
        px, py = x / w, y / h
        pw, ph = patch_size / w, patch_size / h
        patch_box = {
            "x_center": px + pw / 2,
            "y_center": py + ph / 2,
            "width": pw,
            "height": ph,
        }
        for box in boxes:
            if _compute_iou_patch(patch_box, box) > 0:
                return True
        return False

    for _ in range(max_attempts):
        if len(patches) >= num_patches:
            break
        x = random.randint(0, max(0, w - patch_size))
        y = random.randint(0, max(0, h - patch_size))

        if _does_overlap(x, y):
            continue

        patch = image_bgr[y:y + patch_size, x:x + patch_size]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue

        if not _is_on_container(patch):
            continue

        patches.append(patch)

    return patches


def _sample_high_texture_negatives(
    image_bgr: np.ndarray,
    boxes: list[dict],
    num_patches: int,
    patch_size: int,
    img_size: int,
    max_attempts: int,
    edge_threshold: float = 0.10,
) -> list[np.ndarray]:
    """在高纹理非损伤区域采样负样本（铆钉、焊缝、波纹板折线等）。"""
    h, w = image_bgr.shape[:2]
    patches = []

    def _patch_overlaps_bbox(px, py, pw, ph):
        patch_box = {
            "x_center": px + pw / 2, "y_center": py + ph / 2,
            "width": pw, "height": ph,
        }
        for box in boxes:
            if _compute_iou_patch(patch_box, box) > 0:
                return True
        return False

    def _edge_density(patch_bgr):
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        return np.mean(mag > 50)

    for _ in range(max_attempts * 2):  # more attempts since hard to find
        if len(patches) >= num_patches:
            break
        x = random.randint(0, max(0, w - patch_size))
        y = random.randint(0, max(0, h - patch_size))

        pw_norm = patch_size / w
        ph_norm = patch_size / h

        if _patch_overlaps_bbox(x / w, y / h, pw_norm, ph_norm):
            continue

        patch = image_bgr[y:y + patch_size, x:x + patch_size]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue

        if not _is_on_container(patch):
            continue

        if _edge_density(patch) < edge_threshold:
            continue

        patches.append(patch)

    # 如果高纹理采样不足，回退到随机采样
    if len(patches) < num_patches:
        remaining = num_patches - len(patches)
        fallback = _sample_random_negatives(image_bgr, boxes, remaining, patch_size, img_size, max_attempts)
        patches += fallback

    return patches


def _compute_iou_patch(patch_box: dict, bbox: dict) -> float:
    """计算 patch（归一化坐标）与 bbox 的 IoU。"""
    px1 = patch_box["x_center"] - patch_box["width"] / 2
    py1 = patch_box["y_center"] - patch_box["height"] / 2
    px2 = patch_box["x_center"] + patch_box["width"] / 2
    py2 = patch_box["y_center"] + patch_box["height"] / 2

    bx1 = bbox["x_center"] - bbox["width"] / 2
    by1 = bbox["y_center"] - bbox["height"] / 2
    bx2 = bbox["x_center"] + bbox["width"] / 2
    by2 = bbox["y_center"] + bbox["height"] / 2

    xi1 = max(px1, bx1)
    yi1 = max(py1, by1)
    xi2 = min(px2, bx2)
    yi2 = min(py2, by2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area_p = (px2 - px1) * (py2 - py1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_p + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def build_problem1_dataset(
    image_ids: list[str],
    labels: dict[str, list[dict]],
    image_dir: Path,
    global_size: int = GLOBAL_INPUT_SIZE,
    patch_size: int = PATCH_SIZE,
    neg_per_image: int = 2,
    verbose: bool = True,
    neg_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """问题1：构造图像级正负样本。

    正样本：全图 resize → global_size×global_size
    负样本：非 bbox 区域 patch → resize → global_size×global_size

    Args:
        neg_mode: "random" | "boundary" | "high_texture" | "hard_mixed"
    """
    all_images = []
    all_labels = []
    all_stems = []
    all_types = []

    pos_count = 0
    neg_count = 0

    for stem in image_ids:
        img_path = image_dir / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = labels.get(stem, [])

        # 正样本：全图 resize
        pos = cv2.resize(img_rgb, (global_size, global_size))
        all_images.append(pos)
        all_labels.append(1)
        all_stems.append(stem)
        all_types.append("positive")
        pos_count += 1

        # 负样本：从非 bbox 区域采样 patch
        neg_patches = _sample_negative_patches(
            img, boxes, num_patches=neg_per_image, patch_size=patch_size,
            mode=neg_mode,
        )
        for p in neg_patches:
            p_rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            p_resized = cv2.resize(p_rgb, (global_size, global_size))
            all_images.append(p_resized)
            all_labels.append(0)
            all_stems.append(stem)
            all_types.append("negative")
            neg_count += 1

    if verbose:
        print(f"  正样本: {pos_count}, 负样本: {neg_count}, 总计: {len(all_images)}")

    return (
        np.array(all_images, dtype=np.uint8),
        np.array(all_labels, dtype=np.int32),
        all_stems,
        all_types,
    )


def build_problem2_dataset(
    image_ids: list[str],
    labels: dict[str, list[dict]],
    image_dir: Path,
    patch_size: int = PATCH_SIZE,
    neg_per_image: int = 2,
    verbose: bool = True,
    neg_mode: str = "random",
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """问题2：构造 patch 级多分类样本。

    正样本：bbox 区域扩展→ resize → class={0,1,2}
    负样本：非 bbox 区域→ resize → class=3

    Args:
        neg_mode: "random" | "boundary" | "high_texture" | "hard_mixed"

    Returns:
        images: (N, patch_size, patch_size, 3) RGB
        labels_arr: (N,) {0,1,2,3}
        stems: 来源图像标识
        types: "positive" / "negative"
    """
    all_images = []
    all_labels = []
    all_stems = []
    all_types = []

    counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for stem in image_ids:
        img_path = image_dir / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        boxes = labels.get(stem, [])

        # 正样本：每个 bbox 裁出 patch
        for box in boxes:
            cls = box["class"]
            area = bbox_area_pixels(box)
            # 扩展：小目标 30%，一般 10%
            expand = 0.30 if area < 900 else 0.10
            bw = box["width"] * (1 + expand)
            bh = box["height"] * (1 + expand)
            bx = box["x_center"]
            by = box["y_center"]

            x1 = int((bx - bw / 2) * w)
            y1 = int((by - bh / 2) * h)
            x2 = int((bx + bw / 2) * w)
            y2 = int((by + bh / 2) * h)

            # 边界裁剪
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            patch = img_rgb[y1:y2, x1:x2]
            patch = cv2.resize(patch, (patch_size, patch_size))
            all_images.append(patch)
            all_labels.append(cls)
            all_stems.append(stem)
            all_types.append("positive")
            counts[cls] += 1

        # 负样本
        neg_patches = _sample_negative_patches(
            img, boxes, num_patches=neg_per_image, patch_size=patch_size,
            mode=neg_mode,
        )
        for p in neg_patches:
            p_rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            p_resized = cv2.resize(p_rgb, (patch_size, patch_size))
            all_images.append(p_resized)
            all_labels.append(3)
            all_stems.append(stem)
            all_types.append("negative")
            counts[3] += 1

    if verbose:
        print(f"  Dent: {counts[0]}, Hole: {counts[1]}, Rusty: {counts[2]}, "
              f"Background: {counts[3]}, 总计: {len(all_images)}")

    return (
        np.array(all_images, dtype=np.uint8),
        np.array(all_labels, dtype=np.int32),
        all_stems,
        all_types,
    )


def split_train_val(
    image_ids: list[str],
    labels: dict[str, list[dict]],
    val_ratio: float = 0.20,
) -> tuple[list[str], list[str]]:
    """按图像 ID 做分层训练/验证划分。

    分层策略：对每张图的类别组合做分层抽样，保证 Hole 类不集中。

    Returns:
        train_ids, val_ids
    """
    # 计算每张图的"类别签名"
    def _class_signature(boxes):
        classes = tuple(sorted(set(b["class"] for b in boxes)))
        return classes

    # 按签名分组
    groups = defaultdict(list)
    for stem in image_ids:
        sig = _class_signature(labels.get(stem, []))
        groups[sig].append(stem)

    train_ids = []
    val_ids = []

    for sig, stems in groups.items():
        random.shuffle(stems)
        n_val = max(1, int(len(stems) * val_ratio))
        val_ids.extend(stems[:n_val])
        train_ids.extend(stems[n_val:])

    random.shuffle(train_ids)
    random.shuffle(val_ids)

    return train_ids, val_ids
