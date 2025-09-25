"""
Plot Label Distribution 评估器
绘制测试集标签分布图并保存到plots目录
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os


def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """
    绘制测试集标签分布图
    
    Args:
        X_train, y_train, y_train_pred: 训练集数据和预测（未使用）
        X_test, y_test, y_test_pred: 测试集数据和预测
        
    Returns:
        总的测试样本数（作为metric返回值）
    """
    # 计算标签分布
    y_test_flat = y_test.flatten()
    label_counts = Counter(y_test_flat)
    
    # 创建图像
    plt.figure(figsize=(10, 6))
    
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
    
    # 这里我们需要获取当前实验的plots目录路径
    # 由于evaluate函数无法直接访问result_manager，我们使用一个全局变量来传递路径
    plots_dir = getattr(evaluate, '_plots_dir', None)
    if plots_dir:
        plot_path = os.path.join(plots_dir, 'test_label_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"标签分布图已保存: {plot_path}")
    else:
        print("警告: 无法获取plots目录路径，图像未保存")
    
    plt.close()  # 关闭图形以释放内存
    
    # 返回测试样本总数作为metric
    return len(y_test_flat)


def set_plots_dir(plots_dir):
    """设置plots目录路径的辅助函数"""
    evaluate._plots_dir = plots_dir