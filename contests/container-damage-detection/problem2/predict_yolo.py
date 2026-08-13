#!/usr/bin/env python3
"""问题2 推理：用 YOLOv8s-P2 模型生成 test_result.csv（提交文件）。

对测试集逐图推理 → NMS（模型内置）→ 按严重程度排序 → 每图保留 Top-4。

严重程度 severity = 框面积 × 类别权重：
    Hole=1.5（破洞直接威胁结构完整性，最严重）
    Dent=1.0（凹陷影响密封与强度）
    Rusty=0.8（锈蚀是渐进性表面退化，紧急度最低）
面积用归一化 w×h，使排序与图像分辨率无关。

Usage:
    python problem2/predict_yolo.py                    # 全量测试集
    python problem2/predict_yolo.py --limit 20         # 前 20 张（冒烟测试）
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEVERITY_WEIGHTS = {0: 1.0, 1: 1.5, 2: 0.8}
MAX_PER_IMAGE = 4
CONF_THRESHOLD = 0.25


def main():
    from ultralytics import YOLO

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    model_path = ROOT / "models" / "yolov8s_p2_improved.pt"
    test_dir = ROOT / "data" / "raw" / "dataset" / "images" / "test"
    out_csv = ROOT / "test_result.csv"

    if not model_path.exists():
        print(f"错误：模型缺失 {model_path}")
        print("请先运行 `python problem2/train_yolo.py` 或将模型放入 models/。")
        return 1
    if not test_dir.exists():
        print(f"错误：测试集缺失 {test_dir}")
        print("请将竞赛数据集放置到 data/raw/dataset/（见 README）。")
        return 1

    print(f"模型: {model_path}")
    print(f"测试图: {test_dir}")

    model = YOLO(str(model_path))
    img_paths = sorted(test_dir.glob("*.jpg"))
    if limit:
        img_paths = img_paths[:limit]

    all_rows = []
    total_boxes = 0

    for i, img_path in enumerate(img_paths):
        results = model(str(img_path), imgsz=640, conf=CONF_THRESHOLD,
                        iou=0.5, verbose=False)
        r = results[0]
        boxes = r.boxes

        dets = []
        if boxes is not None and len(boxes) > 0:
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            xywh = boxes.xywhn.cpu().numpy()  # 归一化 x_center, y_center, w, h

            for c, cf, (xc, yc, w, h) in zip(cls, conf, xywh):
                severity = (w * h) * SEVERITY_WEIGHTS.get(c, 1.0)
                dets.append((severity, c, xc, yc, w, h, cf))

            # 严重程度降序，保证 Top-4 截断优先保留高严重度框（题目要求）
            dets.sort(key=lambda x: -x[0])
            total_boxes += len(dets)

            for severity, c, xc, yc, w, h, cf in dets[:MAX_PER_IMAGE]:
                all_rows.append([
                    img_path.stem,
                    c,
                    round(xc, 6),
                    round(yc, 6),
                    round(w, 6),
                    round(h, 6),
                    round(cf, 4),
                ])

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(img_paths)}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "class", "x_center", "y_center",
                         "width", "height", "confidence"])
        writer.writerows(all_rows)

    n_with_det = len(set(r[0] for r in all_rows))
    print(f"\n完成!")
    print(f"  总检测框: {total_boxes}")
    print(f"  有检测的图像: {n_with_det}/{len(img_paths)}")
    print(f"  输出行数: {len(all_rows)}")
    print(f"  保存至: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
