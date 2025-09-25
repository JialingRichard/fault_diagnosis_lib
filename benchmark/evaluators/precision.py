"""
Precision 评估器
"""

from sklearn.metrics import precision_score
import numpy as np


def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """
    计算精确率
    
    Args:
        X_train, y_train, y_train_pred: 训练集数据和预测（未使用）
        X_test, y_test, y_test_pred: 测试集数据和预测
        
    Returns:
        精确率
    """
    return precision_score(y_test.flatten(), y_test_pred.flatten(), average='macro')