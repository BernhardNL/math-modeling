git# math-modeling

数学建模学习与练习的 monorepo，收纳所有建模相关的子项目。

## 目录结构

```
math-modeling/
├── practices/       # 练习项目
├── contests/        # 真题 / 竞赛
├── summaries/       # 总结 / 笔记
├── templates/       # 项目模板
│   └── template/    # 新建子项目时复制此目录
└── .gitignore
```

## 使用方法

### 新建子项目

```bash
cp -r templates/template <目录>/<项目名>
```

### 子项目内部结构

```
<项目名>/
├── data/
│   ├── raw/          # 原始数据（只读）
│   └── processed/    # 处理后数据
├── models/           # 数学模型实现
├── notebooks/        # Jupyter Notebooks
├── src/              # 源码
├── output/
│   ├── figures/      # 图片输出
│   └── tables/       # 表格输出
├── main.py           # 主入口
└── requirements.txt  # 依赖
```
