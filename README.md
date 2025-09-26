# 故障诊断基准测试框架

基于配置的多模型多数据集故障诊断基准测试框架，支持模块化组件和批量实验。

## 📁 项目结构

```
benchmark/
├── main.py                      # 程序入口
├── configs/                     # 配置文件
│   └── NASA_IMS.yaml
├── src/                         # 核心Loaders
│   ├── config_loader.py         # 配置加载器
│   ├── data_loader.py           # 数据加载器
│   ├── model_loader.py          # 模型加载器
│   ├── training_loader.py       # 训练加载器
│   ├── eval_loader.py           # 评估加载器
│   ├── epochinfo_loader.py      # Epoch信息加载器
│   └── result_manager.py        # 结果管理器
├── models/                      # 模型定义
│   ├── LSTM.py
│   └── CNN.py
├── trainers/                    # 训练器
│   └── supervised_trainer.py
├── evaluators/                  # 评估器
│   ├── f1.py
│   ├── sklearn_metrics.py
│   └── plot_label_distribution.py
├── preprocessors/               # 预处理器
│   ├── normalizers.py
│   ├── noise_processors.py
│   └── feature_engineering.py
├── data/                        # 数据目录
└── results/                     # 结果输出
```

## � 数据流向图

```
配置文件(YAML) → ConfigLoader
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
   DataLoader    ModelLoader   TrainingLoader
       ↓             ↓             ↓
   数据+预处理  →    模型实例   →   训练器
       ↓             ↓             ↓
       └─────────────┼─────────────┘
                     ↓
              SupervisedTrainer
              (训练循环 + 早停)
                     ↓
              EpochInfoLoader ←─→ EvalLoader
              (实时显示信息)      (训练中评估)
                     ↓
               训练完成模型
                     ↓
               EvalLoader
               (最终评估)
                     ↓
              ResultManager
              (保存结果+日志)
```

## �🔧 核心Loaders功能

### ConfigLoader
- 加载和验证YAML配置文件

### DataLoader  
- 加载数据文件 (train/test split)
- 执行预处理管道 (normalize → denoise → feature engineering)
- 返回处理后的数据和元信息

### ModelLoader
- 根据配置动态加载模型类
- 实例化模型并传入参数

### TrainingLoader
- 根据训练类型创建对应训练器
- 支持验证集划分和早停配置

### EvalLoader
- 动态加载评估函数
- 支持多指标组合评估
- 生成数值结果和可视化图表

### EpochInfoLoader
- 控制训练过程信息显示
- 支持实时评估指标显示
- 可配置日志等级 (minimal/normal/verbose)

### ResultManager
- 自动版本管理 (v1, v2, ...)
- 实时日志记录
- checkpoint保存和结果输出

## 🚀 快速开始

```bash
cd benchmark
python main.py configs/NASA_IMS.yaml
```

## 💡 配置示例

```yaml
# 定义模型
models:
  LSTM:
    hidden_dim: 64
    num_layers: 2

# 定义数据集
datasets:
  NPY_UCI_HAR:
    train_data: ./data/NPY_UCI_HAR/train_X.npy
    preprocessing:
      steps:
        - name: "normalize"
          file: "normalizers"
          function: "standard_normalize"

# 定义实验
experiments:
  - name: "LSTM_NPY_UCI_HAR_baseline"
    model: "LSTM"
    dataset: "NPY_UCI_HAR"
    training: "supervised_complete"
    evaluation: "default"
```