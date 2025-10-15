"""
Plot Label Distribution 评估器
绘制测试集标签分布图

返回: (Figure对象, 测试样本总数)
由 EvalLoader 负责根据上下文保存图像
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """
    绘制测试集标签分布图
    
    Args:
        X_train, y_train, y_train_pred: 训练集数据和预测（未使用）
        X_test, y_test, y_test_pred: 测试集数据和预测
        
    Returns:
        tuple: (Figure对象, 测试样本总数)
        Figure对象会被EvalLoader自动保存到合适的位置
    """
    # 计算标签分布
    y_test_flat = y_test.flatten()
    label_counts = Counter(y_test_flat)
    
    # 创建图像
    fig = plt.figure(figsize=(10, 6))
    
    # 绘制条形图
    labels = sorted(label_counts.keys())
    counts = [label_counts[label] for label in labels]
    
    bars = plt.bar(labels, counts, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # 在每个条形上添加数量标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Class Label', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Test Set Label Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 返回Figure对象和测试样本总数
    # EvalLoader会自动处理图像保存（根据epoch_info判断保存位置）
    return (fig, len(y_test_flat))