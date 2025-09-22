
# 时序异常检测基准测试框架

时序异常检测算法的基准测试框架，支持多种模型和数据集的性能评估对比。

## 项目结构

```
fault_diagnosis_lib/
├── benchmark/
│   ├── src/                           # 核心代码
│   │   ├── config_manager.py          # 配置管理
│   │   ├── dataloaders.py             # 数据加载
│   │   ├── trainer.py                 # 模型训练
│   │   ├── metrics.py                 # 评估指标
│   │   ├── main.ipynb                 # 主实验脚本
│   │   └── main_script.py             # Python脚本版本
│   ├── models/                        # 模型实现
│   │   ├── base_model.py              # 基础模型类
│   │   ├── iforest.py                 # Isolation Forest
│   │   └── lstm_ae.py                 # LSTM AutoEncoder
│   ├── configs/                       # 实验配置
│   │   ├── default_experiment.yaml    # 默认配置
│   │   ├── swat_experiment.yaml       # SWAT数据集配置
│   │   ├── hyperparameter_tuning.yaml # 超参数调优配置
│   │   └── multi_dataset_experiment.yaml # 多数据集配置
│   ├── data/                          # 数据存储
│   └── results/                       # 实验结果
├── test.ipynb                         # 测试脚本
├── test.py
└── test_lstm_gpu.py                   # GPU测试脚本
```

## 快速开始

### 安装依赖

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pyyaml torch
```

### 运行实验

1. 打开 `benchmark/src/main.ipynb` 进行交互式实验
2. 或运行 Python 脚本：`python benchmark/src/main_script.py`

### 配置文件

- `default_experiment.yaml` - 默认实验设置
- `swat_experiment.yaml` - SWAT数据集实验
- `hyperparameter_tuning.yaml` - 超参数调优
- `multi_dataset_experiment.yaml` - 多数据集对比

## 支持的模型

- **Isolation Forest** - 基于随机森林的异常检测
- **LSTM AutoEncoder** - 深度学习序列重构

## 核心功能

- 模块化设计，易于扩展
- YAML配置驱动
- 统一的模型接口
- 多种评估指标
- 自动结果可视化

## 扩展开发

### 添加新模型

1. 继承 `BaseModel` 类
2. 实现 `fit()` 和 `predict()` 方法
3. 在 `ModelFactory` 中注册

### 添加新数据集

1. 在 `dataloaders.py` 中添加数据加载函数
2. 更新配置文件

## 许可证

MIT License