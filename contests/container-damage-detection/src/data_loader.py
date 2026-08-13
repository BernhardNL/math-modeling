"""数据加载与统计。"""

from pathlib import Path
import numpy as np
from collections import Counter

from .config import (
    LABELS_TRAIN, LABELS_TEST, IMAGES_TRAIN, IMAGES_TEST, CLASS_NAMES,
    CLASS_NAMES_CN,
)


def parse_yolo_label(filepath: str) -> list[dict]:
    """解析单个 YOLO 标注文件。

    Returns:
        list of dicts: [{"class": int, "x_center": float, "y_center": float,
                          "width": float, "height": float}, ...]
        空文件返回空列表。
    """
    boxes = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            boxes.append({
                "class": cls_id,
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "width": float(parts[3]),
                "height": float(parts[4]),
            })
    return boxes


def load_all_labels(labels_dir: Path) -> dict[str, list[dict]]:
    """加载目录下所有标注文件。

    Returns:
        {filename_stem: [boxes]}, 空标注对应空列表。
    """
    labels = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        labels[stem] = parse_yolo_label(str(label_path))
    return labels


def bbox_area_pixels(box: dict, img_size: int = 640) -> float:
    """计算 bbox 的绝对像素面积。"""
    return box["width"] * box["height"] * img_size * img_size


def bbox_aspect_ratio(box: dict) -> float:
    """bbox 宽高比（w/h）。"""
    h = box["height"]
    return box["width"] / h if h > 0 else 0.0


def compute_label_statistics(labels: dict[str, list[dict]]) -> dict:
    """计算标注数据的统计信息。"""
    # 按类别统计
    class_counts = Counter()
    all_areas = []
    all_aspect_ratios = []
    boxes_per_image = []
    class_areas = {0: [], 1: [], 2: []}

    for stem, boxes in labels.items():
        boxes_per_image.append(len(boxes))
        for box in boxes:
            cls = box["class"]
            class_counts[cls] += 1
            area = bbox_area_pixels(box)
            all_areas.append(area)
            class_areas[cls].append(area)
            all_aspect_ratios.append(bbox_aspect_ratio(box))

    areas = np.array(all_areas)
    aspect_ratios = np.array(all_aspect_ratios)
    bpi = np.array(boxes_per_image)

    stats = {
        "num_images": len(labels),
        "total_boxes": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "boxes_per_image": {
            "min": int(bpi.min()),
            "max": int(bpi.max()),
            "mean": float(bpi.mean()),
            "median": float(np.median(bpi)),
        },
        "area_stats": {
            "min": float(areas.min()),
            "max": float(areas.max()),
            "mean": float(areas.mean()),
            "median": float(np.median(areas)),
            "p10": float(np.percentile(areas, 10)),
            "p25": float(np.percentile(areas, 25)),
            "p75": float(np.percentile(areas, 75)),
            "p90": float(np.percentile(areas, 90)),
        },
        "aspect_ratio_stats": {
            "min": float(aspect_ratios.min()),
            "max": float(aspect_ratios.max()),
            "mean": float(aspect_ratios.mean()),
            "median": float(np.median(aspect_ratios)),
        },
        "class_area_stats": {
            cls: {
                "count": len(arr),
                "median": float(np.median(arr)) if arr else 0,
                "p10": float(np.percentile(arr, 10)) if arr else 0,
                "p90": float(np.percentile(arr, 90)) if arr else 0,
            }
            for cls, arr in class_areas.items()
        },
    }
    return stats


def print_statistics(stats: dict, title: str = "Dataset Statistics") -> str:
    """格式化打印统计信息，返回字符串便于写入文件。"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {title}")
    lines.append("=" * 60)
    lines.append(f"  图像总数:     {stats['num_images']}")
    lines.append(f"  标注框总数:   {stats['total_boxes']}")
    lines.append("")
    lines.append("  每类标注框数量:")
    for cls in sorted(stats["class_counts"]):
        name = CLASS_NAMES_CN.get(cls, f"Class {cls}")
        lines.append(f"    Class {cls} ({name}): {stats['class_counts'][cls]}")
    lines.append("")
    bpi = stats["boxes_per_image"]
    lines.append(f"  每图框数: min={bpi['min']}, max={bpi['max']}, "
                 f"mean={bpi['mean']:.1f}, median={bpi['median']:.0f}")
    lines.append("")
    a = stats["area_stats"]
    lines.append("  bbox 面积统计 (像素²):")
    lines.append(f"    min={a['min']:.0f}, max={a['max']:.0f}, "
                 f"mean={a['mean']:.0f}, median={a['median']:.0f}")
    lines.append(f"    P10={a['p10']:.0f}, P25={a['p25']:.0f}, "
                 f"P75={a['p75']:.0f}, P90={a['p90']:.0f}")
    lines.append("")
    ar = stats["aspect_ratio_stats"]
    lines.append("  bbox 宽高比统计:")
    lines.append(f"    min={ar['min']:.2f}, max={ar['max']:.2f}, "
                 f"mean={ar['mean']:.2f}, median={ar['median']:.2f}")
    lines.append("")
    lines.append("  各类别面积统计:")
    for cls in sorted(stats["class_area_stats"]):
        name = CLASS_NAMES_CN.get(cls, f"Class {cls}")
        s = stats["class_area_stats"][cls]
        lines.append(f"    Class {cls} ({name}): "
                     f"count={s['count']}, median={s['median']:.0f}, "
                     f"P10={s['p10']:.0f}, P90={s['p90']:.0f}")
    lines.append("=" * 60)
    return "\n".join(lines)
