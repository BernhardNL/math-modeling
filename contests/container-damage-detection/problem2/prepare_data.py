#!/usr/bin/env python3
"""问题2 数据准备：从原始竞赛数据构建 YOLO 训练集。

按 data/splits/ 下的固定划分（与训练 YOLO 时完全一致）把
data/raw/dataset 的训练图复制到 data/yolo_dataset/images/{train,val}，
标签保持 YOLO 归一化格式（class x_center y_center w h），
并生成相对路径的 data.yaml。

为何复制而非符号链接：原始数据可能位于项目外，符号链接会让
项目在别的机器上失效；复制保证 data/yolo_dataset 自包含。

Usage:
    python problem2/prepare_data.py
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dataset"
SPLITS_DIR = ROOT / "data" / "splits"
YOLO_DIR = ROOT / "data" / "yolo_dataset"


def build_split(split: str, ids: list[str]):
    img_dst = YOLO_DIR / "images" / split
    lbl_dst = YOLO_DIR / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    copied, missing_labels = 0, 0
    for stem in ids:
        src_img = RAW / "images" / "train" / f"{stem}.jpg"
        if not src_img.exists():
            print(f"  ✗ 缺图: {src_img.name}")
            continue
        shutil.copy2(src_img, img_dst / f"{stem}.jpg")

        src_lbl = RAW / "labels" / "train" / f"{stem}.txt"
        if src_lbl.exists():
            shutil.copy2(src_lbl, lbl_dst / f"{stem}.txt")
        else:
            # 无标注 = 整图无损伤目标；YOLO 要求标签文件存在或缺失均可，
            # 写空文件显式表示「已核对、确实无目标」。
            (lbl_dst / f"{stem}.txt").write_text("")
            missing_labels += 1
        copied += 1

    print(f"  [{split}] 复制 {copied} 张（其中 {missing_labels} 张无标注）")


def main():
    print("=" * 60)
    print("  问题2 数据准备：构建 YOLO 训练/验证集")
    print("=" * 60)

    train_ids = json.load(open(SPLITS_DIR / "train_ids.json"))
    val_ids = json.load(open(SPLITS_DIR / "val_ids.json"))
    print(f"  划分: train={len(train_ids)}, val={len(val_ids)}")

    if YOLO_DIR.exists():
        # 幂等重建：清掉旧内容（含可能残留的符号链接）
        shutil.rmtree(YOLO_DIR)
    (YOLO_DIR / "images").mkdir(parents=True)
    (YOLO_DIR / "labels").mkdir(parents=True)

    build_split("train", [str(i) for i in train_ids])
    build_split("val", [str(i) for i in val_ids])

    # path 用 "."：ultralytics 相对 data.yaml 所在目录解析，项目可整体搬移。
    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "Dent", 1: "Hole", 2: "Rusty"},
        "nc": 3,
    }
    import yaml
    with open(YOLO_DIR / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"\n  完成: {YOLO_DIR}")
    print("  下一步: python problem2/train_yolo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
