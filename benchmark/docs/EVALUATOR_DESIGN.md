# 评估器设计规范

## 📐 设计原则

### 核心思想
1. **评估器只负责计算和返回结果**，不处理保存逻辑
2. **EvalLoader 负责根据上下文处理返回值**，特别是图像的保存位置
3. **移除评估器内部的全局状态依赖**，使评估器更纯粹和可测试

---

## 🎯 评估器返回值规范

评估器可以返回以下类型：

### 1. **数值型** (推荐)
```python
def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """计算准确率"""
    accuracy = (y_test_pred == y_test).mean()
    return accuracy  # float
```

### 2. **字符串型**
```python
def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """返回文本信息"""
    return f"准确率: {accuracy:.2%}"  # str
```

### 3. **图像型** (matplotlib Figure)
```python
def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """生成图表"""
    fig = plt.figure(figsize=(10, 6))
    # ... 绘图代码 ...
    plt.tight_layout()
    return fig  # matplotlib.figure.Figure
```

### 4. **图像+数值型** (推荐用于绘图评估器)
```python
def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """生成图表并返回指标"""
    fig = plt.figure(figsize=(10, 6))
    # ... 绘图代码 ...
    plt.tight_layout()
    
    metric_value = len(y_test)  # 或其他有意义的数值
    return (fig, metric_value)  # tuple: (Figure, float)
```

---

## 🔧 EvalLoader 自动处理逻辑

### 图像保存规则

#### **训练过程中** (epoch_info 不为 None)
```python
eval_loader.set_context(
    plots_dir='/path/to/plots',
    epoch_info={'epoch': 10},  # 第10轮
    logging_level='normal'
)
```
**保存位置**: `plots/epochinfo/metric_name_epoch_011.png`

#### **最终评估** (epoch_info 为 None)
```python
eval_loader.set_context(
    plots_dir='/path/to/plots',
    epoch_info=None,  # 最终评估
    logging_level='normal'
)
```
**保存位置**: `plots/metric_name.png`

---

## 📝 完整示例

### 示例1: 简单数值评估器
```python
# evaluators/accuracy.py
def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """计算测试集准确率"""
    return (y_test_pred == y_test).mean()
```

### 示例2: 绘图评估器
```python
# evaluators/confusion_matrix_plot.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """
    绘制混淆矩阵
    
    Returns:
        tuple: (Figure对象, 准确率)
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_test_pred)
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 计算准确率
    accuracy = (y_test_pred == y_test).mean()
    
    # 返回Figure和准确率
    # EvalLoader会自动保存图像到合适位置
    return (fig, accuracy)
```

### 示例3: 纯文本评估器
```python
# evaluators/dataset_summary.py
def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """返回数据集统计信息"""
    return f"Train: {len(y_train)}, Test: {len(y_test)}"
```

---

## 🚀 使用方法

### 在 main.py 中（最终评估）
```python
# 设置评估上下文
eval_loader.set_context(
    plots_dir=plots_dir,
    epoch_info=None,  # None表示最终评估
    logging_level='normal'
)

# 执行评估（图像会自动保存到 plots/ 目录）
eval_results = eval_loader.evaluate(
    config, 'basic_metrics',
    X_train, y_train, train_pred,
    X_test, y_test, test_pred
)
```

### 在 epochinfo_loader.py 中（训练过程）
```python
# 设置评估上下文（包含epoch信息）
eval_loader.set_context(
    plots_dir=plots_dir,
    epoch_info={'epoch': epoch_num},  # 指定当前epoch
    logging_level='minimal'
)

# 执行评估（图像会自动保存到 plots/epochinfo/ 目录）
eval_results = eval_loader.evaluate(
    config, 'basic_metrics',
    X_train, y_train, train_pred,
    X_test, y_test, test_pred
)
```

---

## ✅ 优势总结

### 旧设计的问题
- ❌ 评估器需要通过全局变量获取配置
- ❌ 需要手动调用 `set_plots_dir()`, `set_epoch_info()` 等函数
- ❌ 评估器和框架耦合度高
- ❌ 难以测试和复用

### 新设计的优势
- ✅ **评估器纯粹**：只负责计算，不关心保存逻辑
- ✅ **统一处理**：EvalLoader 统一处理所有类型的返回值
- ✅ **易于扩展**：新增评估器只需按规范返回结果
- ✅ **易于测试**：评估器是纯函数，方便单元测试
- ✅ **灵活配置**：通过 `set_context()` 集中管理上下文
- ✅ **自动化**：图像保存位置自动根据 epoch_info 判断

---

## 🔄 迁移指南

### 旧的评估器代码
```python
def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    fig = plt.figure()
    # ... 绘图 ...
    
    # 获取全局变量
    plots_dir = getattr(evaluate, '_plots_dir', None)
    epoch_info = getattr(evaluate, '_epoch_info', None)
    
    # 手动处理保存逻辑
    if plots_dir:
        if epoch_info:
            path = f"{plots_dir}/epochinfo/plot_{epoch_info['epoch']}.png"
        else:
            path = f"{plots_dir}/plot.png"
        plt.savefig(path)
    plt.close()
    
    return metric_value
```

### 新的评估器代码
```python
def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    fig = plt.figure()
    # ... 绘图 ...
    plt.tight_layout()
    
    # 直接返回Figure对象，让EvalLoader处理保存
    return (fig, metric_value)
```

**就这么简单！** 🎉

---

## 📚 相关文件

- **评估器接口**: `benchmark/src/eval_loader.py`
- **示例评估器**: `benchmark/evaluators/plot_label_distribution.py`
- **训练中调用**: `benchmark/src/epochinfo_loader.py`
- **最终评估调用**: `benchmark/main.py`
