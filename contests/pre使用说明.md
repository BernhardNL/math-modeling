# 单张图像
python predict.py data/raw/dataset/images/train/1001.jpg

# 多张图像
python predict.py 1001.jpg 1002.jpg 1003.jpg

# 批量处理整个目录
python predict.py --dir data/raw/dataset/images/test/

# 导出 JSON 结果
python predict.py --dir test_images/ --json results.json

# 自定义阈值（λ=0.25 漏检代价更大，τ=0.2）
python predict.py --tau 0.2 1001.jpg

# 安静模式（只输出汇总）
python predict.py --dir images/ --quiet
