"""
F1 Score 评估器
"""

from sklearn.metrics import f1_score
import numpy as np


def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """
    计算F1分数
    
    Args:
        X_train, y_train, y_train_pred: 训练集数据和预测（未使用）
        X_test, y_test, y_test_pred: 测试集数据和预测
        
    Returns:
        F1分数
    """
    return f1_score(y_test.flatten(), y_test_pred.flatten(), average='macro')