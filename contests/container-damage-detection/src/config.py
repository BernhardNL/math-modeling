"""项目级常量与路径配置。"""

from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 数据路径
DATA_RAW = ROOT / "data" / "raw" / "dataset"
DATA_PROCESSED = ROOT / "data" / "processed"
IMAGES_TRAIN = DATA_RAW / "images" / "train"
IMAGES_TEST = DATA_RAW / "images" / "test"
LABELS_TRAIN = DATA_RAW / "labels" / "train"
LABELS_TEST = DATA_RAW / "labels" / "test"
CLASSES_FILE = DATA_RAW / "classes.txt"

# 输出路径
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = ROOT / "models"

# 检测结果输出
RESULT_CSV = ROOT / "test_result.csv"

# 类别定义
CLASS_NAMES = {0: "Dent", 1: "Hole", 2: "Rusty"}
CLASS_NAMES_CN = {0: "Dent（凹陷）", 1: "Hole（破洞）", 2: "Rusty（锈蚀）"}
NUM_CLASSES = 3

# 严重程度权重（用于按严重程度排序）
SEVERITY_WEIGHTS = {0: 1.0, 1: 1.5, 2: 0.8}

# 图像参数
IMAGE_SIZE = 640
GLOBAL_INPUT_SIZE = 448  # 问题1用：全局特征提取时的统一尺寸
PATCH_SIZE = 224         # 问题2用：窗口 resize 尺寸

# 滑动窗口参数（问题2）
SLIDING_WINDOW_SCALES = [
    {"window": 96, "stride": 32},   # 小缺陷
    {"window": 160, "stride": 48},  # 中等缺陷
    {"window": 288, "stride": 80},  # 大缺陷
    {"window": 448, "stride": 128}, # 超大缺陷
]

# NMS 参数
NMS_IOU_THRESHOLD = 0.5
NMS_GLOBAL_IOU_THRESHOLD = 0.6

# 最终输出每张图最多保留的检测框数
MAX_DETECTIONS_PER_IMAGE = 4

# GLCM 参数
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, 0.785398, 1.570796, 2.356194]  # 0°, 45°, 90°, 135° in radians
GLCM_LEVELS = 64

# HOG 参数
HOG_CELL_SIZE = 32
HOG_BLOCK_SIZE = 2
HOG_N_BINS = 9
HOG_PCA_COMPONENTS = 15

# 随机种子
RANDOM_SEED = 42
