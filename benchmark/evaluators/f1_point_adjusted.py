"""
F1 Point Adjusted 评估器
"""

from sklearn.metrics import f1_score
import numpy as np


def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """
    计算点调整F1分数
    
    这是一个示例实现，具体的点调整逻辑需要根据实际需求定义
    
    Args:
        X_train, y_train, y_train_pred: 训练集数据和预测
        X_test, y_test, y_test_pred: 测试集数据和预测
        
    Returns:
        点调整F1分数
    """
    # 简单实现：基础F1分数 + 基于训练集性能的调整因子
    base_f1 = f1_score(y_test.flatten(), y_test_pred.flatten(), average='macro')
    train_f1 = f1_score(y_train.flatten(), y_train_pred.flatten(), average='macro')
    
    # 调整因子：如果测试集性能接近训练集，给予奖励
    adjustment = 1.0 - abs(base_f1 - train_f1) * 0.1
    
    return base_f1 * adjustment