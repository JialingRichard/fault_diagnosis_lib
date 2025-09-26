"""
标准化预处理器
============

提供各种数据标准化方法
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def minmax_normalize(X_train, X_test, **kwargs):
    """
    MinMax标准化（0-1范围）
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    scaler = MinMaxScaler()
    X_train_processed = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
    X_test_processed = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
    
    # 恢复原始形状
    X_train_processed = X_train_processed.reshape(X_train.shape)
    X_test_processed = X_test_processed.reshape(X_test.shape)
    
    return X_train_processed, X_test_processed


def standard_normalize(X_train, X_test, **kwargs):
    """
    标准正态化（Z-score标准化）
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
    X_test_processed = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
    
    # 恢复原始形状
    X_train_processed = X_train_processed.reshape(X_train.shape)
    X_test_processed = X_test_processed.reshape(X_test.shape)
    
    return X_train_processed, X_test_processed


def robust_normalize(X_train, X_test, **kwargs):
    """
    鲁棒标准化（使用中位数和IQR）
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    scaler = RobustScaler()
    X_train_processed = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
    X_test_processed = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
    
    # 恢复原始形状
    X_train_processed = X_train_processed.reshape(X_train.shape)
    X_test_processed = X_test_processed.reshape(X_test.shape)
    
    return X_train_processed, X_test_processed


def no_normalize(X_train, X_test, **kwargs):
    """
    不进行标准化
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    return X_train, X_test