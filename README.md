# 故障诊断基准测试框架

基于配置的多模型多数据集故障诊断基准测试框架，支持模块化组件、网格搜索与可复现实验。

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

## 🔄 数据流向图

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

## 🧰 核心组件

### ConfigLoader
- 加载和验证YAML配置文件

### DataLoader  
- 加载数据文件 (train/test split)
- 执行预处理管道 (normalize → denoise → feature engineering)
- 返回处理后的数据和元信息

### ModelLoader
- 根据配置动态加载模型类
- 实例化模型并传入参数

### TrainingLoader / SupervisedTrainer
- 根据训练类型创建对应训练器
- 显式验证集策略（validation_split）与早停（默认基于 val_loss）
- 数据常驻 CPU，按 batch 搬运至 device，避免显存过载

### EvalLoader
- 动态加载评估函数
- 支持多指标组合评估
- 生成数值结果和可视化图表

### EpochInfoLoader
- 控制训练过程信息显示与训练期评估（依据训练模板的 `epochinfo`）
- 训练期评估默认使用验证集（`epochinfo_split: 'val'`），日志行尾标注 `split:val|test`

### ResultManager
- 自动版本管理 (v1, v2, ...)
- 实时日志记录
- checkpoint保存和结果输出

## 🚀 快速开始

```bash
cd benchmark
python main.py configs/NASA_IMS.yaml
```

## 💡 配置要点与示例

### 1) 全局设置（必看）
```yaml
global:
  seed: 42                  # 随机种子（Python/NumPy/PyTorch）
  deterministic: false      # 更强确定性（可能降低性能）
  device: 'cuda'            # 训练设备
  checkpoint_policy: 'best' # 'best' 仅保留最佳; 'all' 每个epoch都保存
  pre_test: true            # 训练前预检 evaluator 可用性（用2条样本）
```

### 2) 训练模板（显式验证与训练期评估）
```yaml
training_templates:
  supervised_debug_with_metrics:
    type: supervised
    batch_size: 64
    epochs: 3
    lr: 0.001
    patience: 2
    optimizer: 'adam'
    print_interval: 1

    # 验证集：须显式配置（不再默认0.2）
    validation_split: 0.2     # (0,1) 之间; 或 0.0 禁止切分
    # 若无验证集且确需用测试集充当验证集，需显式开启（自担风险）
    # use_test_as_val: true

    num_workers: 0            # DataLoader workers

    # 训练期评估：引用 evaluation_templates 下的模板
    epochinfo: 'train_acc'    # 轻量模板，仅 accuracy
    epochinfo_split: 'val'    # 训练期评估使用的 split（默认 val）

    # 最优 ckpt 监控：强约束，需显式指定
    monitor:
      metric: 'accuracy'
      mode: 'max'
      split: 'val'            # 一般用 val；如设为 test 将打印警告
```

### 3) 评估模板（扁平结构）
```yaml
evaluation_templates:
  # 训练期轻量模板（仅 accuracy）
  train_acc:
    accuracy:
      file: sklearn_metrics
      function: accuracy_evaluate

  # 最终评估模板（完整指标与可视化）
  default:
    f1: {}
    precision:
      file: sklearn_metrics
      function: precision_evaluate
    recall:
      file: sklearn_metrics
      function: recall_evaluate
    accuracy:
      file: sklearn_metrics
      function: accuracy_evaluate
    train_test_gap:
      file: sklearn_metrics
      function: train_test_gap_evaluate
    test_samples:
      file: plot_label_distribution
      function: evaluate
```

### 4) 模型（显式指定类名）
```yaml
models:
  LSTM:
    module: models/LSTM
    class: LSTM2one          # 或 LSTM2seq（序列输出）
    hidden_dim: 64
    num_layers: 2
    dropout: 0.2

  CNN:
    module: models/CNN
    class: CNN2one           # 或 CNN2seq（序列输出）
    num_filters: 64
    filter_sizes: [3,5,7]
    num_layers: 3
    dropout: 0.2
```

### 5) 实验示例
```yaml
experiments:
  - name: "LSTM_NPY_UCI_HAR_baseline"
    model: "LSTM"
    dataset: "NPY_UCI_HAR"
    training: "supervised_debug_with_metrics"
    evaluation: "default"
```

## 📒 日志与结果
- run.log：INFO 概览（包含全局配置、实验清单、训练/评估摘要）
- debug.log：DEBUG 细节（包含 traceback；已过滤 matplotlib findfont 噪声）
- error.log：错误与堆栈（预检/训练/评估异常时写入上下文+traceback）
- best.pth：在 checkpoints/ 下维护最佳模型（按 monitor 指标与 split）

## ✅ 预检（可选）
- 打开 `global.pre_test: true` 后，框架在正式训练前会用 2 条训练样本对：
  - 训练期模板（epochinfo）与最终模板（evaluation）中的每个 evaluator 做一次调用
  - 仅对 monitor 指标强制为数值；其余只需不抛错

## ⚠️ 常见提醒
- 如未配置 `validation_split` 且未显式 `use_test_as_val: true`，将直接报错（不再隐式用 0.2 或回退 test）
- 若 `epochinfo_split` 或 `monitor.split` 使用 `'test'`，训练开始时会打印警告，提示可能信息泄露
- 概率型指标（如 AUC/PR）需要模型概率输出；若只提供 argmax，相关指标将不可用或需在 evaluator 内自行转换
