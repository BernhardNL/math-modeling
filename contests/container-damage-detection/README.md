# 集装箱破损检测（选题 D）

基于计算机视觉的集装箱外表面破损（Dent 凹陷 / Hole 破洞 / Rusty 锈蚀）自动检测。
2026 数学建模竞赛选题 D 的完整代码仓库。

## 三个问题与最终方案

| 问题 | 任务 | 最终方案 | 性能 |
|---|---|---|---|
| 问题1 | 图像级二分类（有无残损） | 51 维手工特征（GLCM+HSV+Sobel+HOG-PCA+LBP）+ 逻辑回归 + 贝叶斯最小风险阈值 | F2=0.9985, AUC=1.0000 |
| 问题2 | 残损检测与分类 | **YOLOv8s-P2 + Wise-IoU + SlideLoss**（传统路线 patch 分类 + 滑动窗口作为探索性基线保留） | mAP@0.5=0.403 |
| 问题3 | 模型评估 | 多维度系统评估 + 论文 | 见 `output/论文_集装箱破损检测.docx` |

提交文件：`test_result.csv`（413 张测试图 775 个检测框，每图最多 4 框按严重程度排序）。

## 目录结构

```
container-damage-detection/
├── README.md
├── requirements.txt          # 依赖（Python 3.10）
├── data/
│   ├── raw/                  # 原始竞赛数据集（不入库，需自行放置）
│   ├── splits/               # YOLO 训练/验证划分（train 2642 / val 658）
│   └── yolo_dataset/         # prepare_data.py 生成（不入库）
├── models/                   # 训练好的模型（问题1 三个 pkl + 问题2 LGBM + YOLO）
├── output/                   # 论文 docx 与插图、统计表
├── src/                      # 共享库：特征提取/数据集构造/分类器/配置
├── problem1/
│   ├── train.py              # 训练：特征 + 逻辑回归 + 贝叶斯阈值
│   └── predict.py            # 推理：判断图像是否有残损
├── problem2/
│   ├── prepare_data.py       # 构建 YOLO 数据集（复制 + 相对路径 data.yaml）
│   ├── patch_classify.py     # 传统路线基线：72 维特征 + LightGBM 四分类 + 滑窗检测
│   ├── train_yolo.py         # 最终方案训练：YOLOv8s-P2 + Wise-IoU + SlideLoss
│   └── predict_yolo.py       # 最终方案推理：生成 test_result.csv
├── problem3/
│   └── evaluate.py           # 三个模型的系统评估
└── paper/
    └── generate_paper.py     # 生成论文 Word 文档
```

## 快速开始

```bash
pip install -r requirements.txt   # 或使用项目 .venv
```

### 数据准备

将竞赛原始数据集（含 `images/train`、`images/test`、`labels/train`、`labels/test`、`classes.txt`）
放置到 `data/raw/dataset/`，然后：

```bash
python problem2/prepare_data.py    # 按固定划分构建 data/yolo_dataset
```

### 问题1：图像级残损判定

```bash
python problem1/train.py           # 训练（特征提取约 10 分钟），产出 models/ 三个 pkl + threshold_info.json
python problem1/predict.py data/raw/dataset/images/train/1001.jpg
python problem1/predict.py --dir data/raw/dataset/images/test/
```

阈值默认读取 `models/threshold_info.json`（贝叶斯 τ，λ=0.5 → τ=1/3）；
`--tau` 可覆盖（阈值越低漏检越少、误报越多）。

### 问题2：残损检测

传统路线基线（论文 4.2–4.6，方法论对照）：

```bash
python problem2/patch_classify.py train
python problem2/patch_classify.py detect --demo
python problem2/patch_classify.py classify <image.jpg>
```

最终方案（论文 4.7，需 NVIDIA GPU）：

```bash
python problem2/train_yolo.py              # 训练 YOLOv8s-P2（~150 epoch）
python problem2/train_yolo.py --resume     # 中断恢复
python problem2/predict_yolo.py            # 生成 test_result.csv
python problem2/predict_yolo.py --limit 20 # 冒烟测试
```

### 问题3：评估

```bash
python problem3/evaluate.py --skip-yolo    # CPU：问题1 + patch 分类
python problem3/evaluate.py                # 含 YOLO 验证集 mAP（GPU）
```

### 论文

```bash
python paper/generate_paper.py             # 生成 output/论文_集装箱破损检测.docx
```

## 模型清单（models/）

| 文件 | 用途 | 来源脚本 |
|---|---|---|
| `logistic_regression.pkl` | 问题1 逻辑回归（C 网格搜索最优） | problem1/train.py |
| `standard_scaler.pkl` | 问题1 特征标准化器 | problem1/train.py |
| `hog_pca.pkl` | 问题1 HOG 6084→15 维 PCA | problem1/train.py |
| `threshold_info.json` | 问题1 贝叶斯阈值 τ 与 λ 扫描记录 | problem1/train.py |
| `rf_stage_c.pkl` | 问题2 patch 四分类（LightGBM，文件名保留早期命名） | problem2/patch_classify.py |
| `hog_pca_stage_c.pkl` / `scaler_stage_c.pkl` | 问题2 特征预处理器 | problem2/patch_classify.py |
| `yolov8s_p2_improved.pt` | 问题2 最终检测模型（epoch 133 best） | problem2/train_yolo.py |

## 注意事项

- **ultralytics 版本必须为 8.4.117**：`train_yolo.py` 通过 monkey-patch 修改
  `BboxLoss`/`v8DetectionLoss` 内部接口实现 Wise-IoU 与 SlideLoss，
  接口随版本变化，升级前需核对签名。
- `patch_classify.py` 当前训练 LightGBM（与 models/ 中保存的模型一致）；
  论文 4.3–4.4 节记录的 Random Forest（85.68%）为早期实验结果，两者结论一致：
  patch 分类可用、滑窗检测失效。
- `data/raw` 与 `data/yolo_dataset` 不入库；克隆后按上文放置数据并运行
  `prepare_data.py` 即可复现全部流程。
- 训练 YOLO 约需 8GB 显存（batch=16, imgsz=640）；推理 CPU 可用但较慢。
