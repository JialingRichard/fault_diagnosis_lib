# 时序异常检测基准测试框架

一个全面、模块化的时序异常检测算法基准测试框架，用于系统性地评估和对比不同异常检测方法在时序数据上的性能。

## 🎯 项目特色

- **🏗️ 模块化架构**: 五大核心组件，职责分离，易于扩展
- **📊 智能评估**: 基于数据元数据的自适应评估指标选择
- **🔧 统一接口**: 抽象基类设计，支持快速集成新算法
- **⚡ 高效数据流**: 流式数据处理，支持大规模数据集
- **📈 丰富可视化**: 多维度结果分析和性能对比图表
- **🧪 实验管理**: YAML配置驱动，支持批量实验和超参数调优

## 📁 项目结构

```
benchmark/
├── src/                        # 核心源代码
│   ├── __init__.py            # 包初始化
│   ├── config_manager.py      # 配置管理器
│   ├── dataloaders.py         # 数据管道
│   ├── trainer.py             # 模型训练器
│   ├── metrics.py             # 评估指标
│   ├── models/                # 模型实现
│   │   ├── __init__.py
│   │   ├── base_model.py      # 抽象基类和工厂
│   │   ├── iforest.py         # Isolation Forest
│   │   └── lstm_ae.py         # LSTM AutoEncoder
│   └── main.ipynb             # 实验运行主脚本
├── configs/                   # 实验配置文件
│   ├── swat_experiment.yaml   # SWAT数据集实验
│   ├── multi_dataset_experiment.yaml  # 多数据集对比
│   └── hyperparameter_tuning.yaml     # 超参数调优
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 预处理后数据
│   └── synthetic/             # 合成数据
├── results/                   # 实验结果
│   ├── reports/               # 实验报告
│   ├── plots/                 # 可视化图表
│   └── logs/                  # 训练日志
├── models/                    # 保存的模型
└── docs/                      # 文档
    ├── user_guide.md          # 用户指南
    ├── developer_guide.md     # 开发者指南
    └── api_reference.md       # API参考
```

## 🏗️ 核心架构

### 1. 配置管理器 (ConfigManager)
- **职责**: 实验配置的加载、验证、合并和保存
- **特性**: 
  - YAML格式配置文件
  - 配置模板和继承
  - 参数验证和类型检查
  - 配置版本管理

### 2. 数据管道 (DataPipeline)
- **职责**: 数据加载、预处理、元数据管理
- **特性**:
  - 多数据集支持 (SWAT, SMD, KDD等)
  - 自动元数据提取
  - 灵活的预处理流程
  - 流式数据处理

### 3. 模型中心 (ModelHub)
- **职责**: 模型注册、创建、管理
- **特性**:
  - 统一的模型接口
  - 工厂模式创建
  - 自动参数验证
  - 插件式扩展

### 4. 训练器 (Trainer)
- **职责**: 模型训练、验证、超参数调优
- **特性**:
  - 交叉验证支持
  - 网格/随机搜索
  - 训练过程监控
  - 模型保存/加载

### 5. 评估器 (Evaluator) 
- **职责**: 性能评估、指标计算、结果分析
- **特性**:
  - 时序专用指标
  - Point-Adjusted评估
  - 自适应指标选择
  - 统计显著性检验

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install numpy pandas scikit-learn matplotlib seaborn pyyaml
# 可选：深度学习支持
pip install torch torchvision
```

### 2. 基础使用

```python
# 导入框架
from src.config_manager import ConfigManager
from src.dataloaders import DataPipeline
from src.models.base_model import ModelFactory
from src.trainer import Trainer
from src.metrics import TimeSeriesEvaluator

# 加载配置
config_manager = ConfigManager()
config = config_manager.load_config('configs/swat_experiment.yaml')

# 加载数据
data_pipeline = DataPipeline()
train_data, train_labels, test_data, test_labels, metadata = \
    data_pipeline.load_dataset('swat', config['data'])

# 创建和训练模型
model = ModelFactory.create_model('iforest', config['models']['iforest'])
model.fit(train_data, metadata)

# 预测和评估
anomaly_scores = model.predict(test_data, metadata)
evaluator = TimeSeriesEvaluator()
results = evaluator.evaluate(test_labels, anomaly_scores, metadata)

print(f\"F1 Score: {results['f1']:.4f}\")
```

### 3. 使用Jupyter Notebook

打开 `src/main.ipynb` 获得完整的交互式实验体验：

- 📋 配置管理和实验设置
- 📊 数据加载和预处理
- 🤖 模型训练和评估  
- 📈 结果可视化和对比
- 💾 结果保存和报告生成

## 🔧 支持的模型

### 已实现模型

1. **Isolation Forest** (`iforest`)
   - 无监督异常检测
   - 适用于高维数据
   - 快速训练和推理

2. **LSTM AutoEncoder** (`lstm_ae`)
   - 深度学习方法
   - 序列建模能力强
   - 支持GPU加速

### 即将支持

- Local Outlier Factor (LOF)
- One-Class SVM
- Transformer-based AutoEncoder
- Variational AutoEncoder (VAE)
- 更多...

## 📊 支持的数据集

### 工业数据集
- **SWAT**: 安全水处理数据集
- **SMD**: 服务器机器数据集
- **WADI**: 水分配数据集

### 网络安全数据集  
- **KDD Cup 99**: 网络入侵检测
- **NSL-KDD**: KDD的改进版本

### 合成数据集
- 内置多种模式的合成数据生成器
- 支持不同异常模式和噪声水平

## ⚙️ 配置系统

### 配置文件结构

```yaml
experiment:
  name: \"swat_benchmark_01\"
  description: \"SWAT数据集基准测试\"

data:
  dataset: \"swat\"
  preprocessing:
    normalize: true
    fill_missing: true

models:
  iforest:
    n_estimators: 100
    contamination: 0.1
  lstm_ae:
    hidden_dim: 64
    epochs: 50

evaluation:
  metrics: [\"f1\", \"precision\", \"recall\", \"auc\"]
  auto_threshold: true

output:
  save_results: true
  generate_plots: true
```

### 配置继承和模板

```yaml
# 基础配置模板
base: &base_config
  evaluation:
    auto_threshold: true
    tolerance: 0

# 继承基础配置
swat_experiment:
  <<: *base_config
  experiment:
    name: \"swat_test\"
  data:
    dataset: \"swat\"
```

## 📈 评估指标

### 传统指标
- **Precision**: 精确率
- **Recall**: 召回率  
- **F1 Score**: F1分数
- **AUC**: ROC曲线下面积
- **Accuracy**: 准确率

### 时序专用指标
- **Point-Adjusted F1**: 考虑时序连续性的F1
- **Range-based Precision/Recall**: 基于异常区间的评估
- **Detection Delay**: 检测延迟
- **Early Detection Rate**: 早期检测率

### 自适应指标选择

框架根据数据元数据自动选择最适合的评估方式：

```python
# 根据标签粒度和故障类型自动选择
if metadata.label_granularity == \"point-wise\":
    if metadata.fault_type == \"binary\":
        # 使用Point-Adjusted指标
        return evaluate_binary_pointwise(...)
    else:
        # 使用多类别评估
        return evaluate_multiclass_pointwise(...)
```

## 🧪 高级实验功能

### 1. 超参数调优

```yaml
hyperparameter_search:
  method: \"grid_search\"  # grid_search, random_search
  param_grid:
    hidden_dim: [32, 64, 128]
    num_layers: [1, 2, 3]
    lr: [0.001, 0.01, 0.1]
  optimization_metric: \"f1\"
```

### 2. 交叉验证

```yaml
evaluation:
  cross_validation:
    enabled: true
    folds: 5
    method: \"time_series\"  # standard, time_series
```

### 3. 批量实验

```python
# 批量运行多个配置
configs = [
    'configs/swat_experiment.yaml',
    'configs/smd_experiment.yaml',
    'configs/kdd_experiment.yaml'
]

for config_file in configs:
    run_experiment(config_file)
```

## 🔌 扩展开发

### 添加新模型

1. **继承BaseModel**:

```python
from src.models.base_model import BaseModel

class MyAnomalyDetector(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化模型参数
    
    def fit(self, data, metadata):
        # 实现训练逻辑
        pass
    
    def predict(self, data, metadata):
        # 实现预测逻辑
        return anomaly_scores
```

2. **注册模型**:

```python
from src.models.base_model import ModelFactory

ModelFactory.register_model('my_detector', MyAnomalyDetector)
```

### 添加新数据集

```python
from src.dataloaders import BaseDataLoader

class MyDataLoader(BaseDataLoader):
    def load_data(self, config):
        # 实现数据加载逻辑
        return train_data, train_labels, test_data, test_labels, metadata
```

### 添加新评估指标

```python
from src.metrics import TimeSeriesEvaluator

class MyEvaluator(TimeSeriesEvaluator):
    def compute_my_metric(self, y_true, y_pred):
        # 实现自定义指标
        return metric_value
```

## 📊 实验结果示例

### 性能对比表格

| 模型 | F1 Score | Precision | Recall | AUC | Point-Adjusted F1 |
|------|----------|-----------|---------|-----|-------------------|
| Isolation Forest | 0.8234 | 0.7891 | 0.8621 | 0.9156 | 0.7983 |
| LSTM AutoEncoder | 0.8567 | 0.8234 | 0.8934 | 0.9287 | 0.8312 |

### 可视化分析

框架自动生成多种分析图表：

- 📊 **性能对比条形图**: 多模型指标对比
- 📈 **异常分数分布**: 正常vs异常样本分布  
- 🕒 **时序检测可视化**: 时间轴上的异常检测结果
- 🎯 **混淆矩阵热力图**: 分类结果详细分析
- 📉 **ROC/PR曲线**: 不同阈值下的性能曲线

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 贡献方式

1. **🐛 报告问题**: 提交bug报告或功能请求
2. **📝 改进文档**: 完善用户指南或API文档
3. **🔧 修复bug**: 提交代码修复
4. **✨ 新功能**: 添加新模型、数据集或评估指标
5. **🧪 测试用例**: 编写单元测试或集成测试

### 开发流程

1. Fork项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

### 代码规范

- 遵循PEP 8代码风格
- 添加详细的文档字符串
- 编写相应的测试用例
- 更新相关文档

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙋 FAQ

### Q: 如何添加自定义数据集？
A: 继承`BaseDataLoader`类并实现`load_data`方法，然后在配置文件中指定新的数据集名称。

### Q: 支持GPU训练吗？
A: 是的，LSTM AutoEncoder等深度学习模型支持GPU加速，在配置中设置`device: \"cuda\"`即可。

### Q: 如何处理大规模数据集？
A: 框架支持流式数据处理和批量训练，可以通过调整`batch_size`和启用数据采样来处理大规模数据。

### Q: 可以自定义评估指标吗？
A: 可以，继承`TimeSeriesEvaluator`类并添加新的指标计算方法，然后在配置中指定即可。

## 📞 联系我们

- **项目维护者**: Fault Diagnosis Benchmark Team
- **邮箱**: benchmark-team@example.com  
- **GitHub**: [时序异常检测基准测试框架](https://github.com/your-org/fault-diagnosis-benchmark)
- **文档站点**: [在线文档](https://your-org.github.io/fault-diagnosis-benchmark)

---

<div align=\"center\">
  <b>⭐ 如果这个项目对您有帮助，请给我们一个Star！</b><br>
  <sub>Built with ❤️ by the Fault Diagnosis Benchmark Team</sub>
</div>"
