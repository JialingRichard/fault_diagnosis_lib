# 故障诊断基准测试框架 Tutorial

本教程带你从 0 到 1 了解并使用本框架：快速运行、配置要点、如何新增数据集/模型/评估器，以及如何做网格搜索与集合数据集评测。

## 1. 快速开始

- 进入工程并运行默认示例：
  - `cd benchmark`
  - `python main.py configs/NASA_IMS.yaml`

运行后会在 `results/<配置文件名>/vN_时间戳/` 下生成独立版本目录，包含：
- `run.log`：控制台与关键信息（已 tee 输出）
- `debug.log`：DEBUG 详情（过滤 matplotlib findfont 噪声）
- `error.log`：错误与堆栈
- `实验名/checkpoints/`：ckpt 文件（另维护 `best.pth`）
- `实验名/plots/`：评估图像与训练期图像

## 2. 核心理念（配置驱动）

- 配置文件定义一切：数据集、模型、训练模板、评估模板与实验清单（可展开网格/集合）。
- 训练模板必须显式：
  - `validation_split`（或自担风险开启 `use_test_as_val`）
  - `epochinfo`（训练期评估所用的模板名）
  - `monitor.metric/mode/split`（最佳 ckpt 监控指标/规则/来源）

最小可用训练模板示例：

```yaml
training_templates:
  supervised_min:
    type: supervised
    batch_size: 64
    epochs: 10
    lr: 0.001
    patience: 3
    validation_split: 0.2
    epochinfo: 'train_acc'
    monitor:
      metric: 'accuracy'
      mode: 'max'
      split: 'val'
```

对应评估模板需在 `evaluation_templates` 中定义 `train_acc`：

```yaml
evaluation_templates:
  train_acc:
    accuracy:
      file: sklearn_metrics
      function: accuracy_evaluate
```

## 3. 运行不同配置

- 单数据集多模型对比：
  - `python main.py configs/NASA_IMS.yaml`
- 网格搜索（会自动展开为多实验）：
  - `python main.py configs/grid_search.yaml`
- 集合数据集（自动发现子数据集并展开）：
  - `python main.py configs/collection_mode_patchtst.yaml`

提示：设置 `global.pre_test: true` 可在大规模实验前做轻量“可用性预检”。

## 4. 配置结构要点

- `global`：随机种子、设备、ckpt 策略、预检与汇总规则。
- `datasets`：
  - 单数据集：`train_X.npy/train_y.npy/test_X.npy/test_y.npy`
  - 集合数据集：`collection_path` 指向根目录，自动发现子目录（需包含上述 4 个文件）。
  - 预处理：支持模块化 steps（`file/function/params`）与简化布尔枚举（如 `normalize:true|minmax|standard`）。
- `models`：每个模型提供 `module` 与 `class`，其它键值将直接作为构造参数传入；框架补充 `input_dim/output_dim/time_steps`。
- `training_templates`：见第 2 节最小模板；请显式 `validation_split/epochinfo/monitor`。
- `evaluation_templates`：扁平或带 `metrics` 均可，键为指标名；训练期与最终评估均从这里动态加载函数。
- `experiments`：列表；可用 `dataset` 或 `dataset_collection`。

## 5. 新增一个数据集

1) 将数据保存为 Numpy：
- `train_X.npy`: (N_train, L, C)
- `train_y.npy`: (N_train,) 或 (N_train, 1)
- `test_X.npy`:  (N_test, L, C)
- `test_y.npy`:  (N_test,) 或 (N_test, 1)

2) 在配置中添加：

```yaml
datasets:
  MYDATA:
    train_data: ./data/MYDATA/train_X.npy
    train_label: ./data/MYDATA/train_y.npy
    test_data: ./data/MYDATA/test_X.npy
    test_label: ./data/MYDATA/test_y.npy
    preprocessing:
      steps:
        - name: normalize
          file: normalizers
          function: standard_normalize
```

3) 在 `experiments` 中引用该数据集。

## 6. 新增一个模型

1) 在 `benchmark/models/` 下新增文件并实现类：例如 `MyModel2one`（输出 `(B, num_classes)`）。
2) 在配置 `models` 中声明：

```yaml
models:
  MyModel:
    module: models/MyModel
    class: MyModel2one
    hidden_dim: 64
    dropout: 0.2
```

3) 在某个实验中设定 `model: "MyModel"`。

注意：如果是序列输出 `(B, L, num_classes)`，Trainer 会在训练中自动转为末时间步或作平均；分类任务建议直接用 `*2one` 类。

## 7. 新增一个评估器

1) 在 `benchmark/evaluators/` 新建 `my_metric.py`，实现：

```python
def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    # 返回 float、str、Figure 或 (Figure, float)
    return 0.0
```

2) 在 `evaluation_templates` 中引用：

```yaml
evaluation_templates:
  default:
    my_metric:
      file: my_metric
      function: evaluate
```

## 8. 网格搜索与集合数据集

- 网格语法：`"{64, 128}"`、`"{[3,5,7], [3,5]}"`，框架会在运行时展开为多实验并自动命名。
- 集合数据集：在 `datasets` 下配置 `collection_path`，实验用 `dataset_collection` 字段，框架会自动枚举子数据集。

## 9. 结果与日志

- `run.log`：训练过程与摘要；`debug.log`：详细调试；`error.log`：异常追踪。
- 训练期评估图像位于 `plots/epochinfo/`，最终评估图像位于 `plots/`。
- 收尾阶段会按“按数据集/按模型”导出 Excel（安装 `pandas openpyxl`）。

## 10. 常见问题 (FAQ)

- 报错“lack of epochinfo and monitor”或“monitor.metric not found”：
  - 请在训练模板中显式设置 `epochinfo` 和 `monitor`；并确保 `epochinfo` 所指模板在 `evaluation_templates` 中存在，且包含 `monitor.metric`（如 `accuracy`）。
- 使用测试集做验证集安全吗？
  - 不建议。若必须如此可设置 `use_test_as_val: true`，框架会打印风险提示。
- AUC/PR 指标？
  - 当前示例评估器基于类别预测；若需 AUC/PR，请让模型输出概率（logits→softmax）并在评估器中使用概率。

## 11. 参考配置

- `configs/NASA_IMS.yaml`：推荐起点（包含严格的 `epochinfo` 与 `monitor` 示例）。
- `configs/collection_mode_patchtst.yaml`：分类任务 + PatchTST + 集合数据集 + 网格组合示例。

祝使用顺利，若需要我帮你生成一份最小可运行配置或对你的数据集做一次连通性检查，告诉我你的数据形状与类别数即可。

