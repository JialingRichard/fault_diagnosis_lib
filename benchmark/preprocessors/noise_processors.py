"""
噪声处理预处理器
==============

提供各种噪声处理方法
"""

import numpy as np
from scipy import signal


def add_gaussian_noise(X_train, X_test, noise_level=0.01, **kwargs):
    """
    添加高斯噪声
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        noise_level: 噪声强度
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    np.random.seed(42)  # 保证可重复性
    
    train_noise = np.random.normal(0, noise_level, X_train.shape)
    test_noise = np.random.normal(0, noise_level, X_test.shape)
    
    X_train_processed = X_train + train_noise
    X_test_processed = X_test + test_noise
    
    return X_train_processed, X_test_processed


def remove_outliers(X_train, X_test, threshold=3.0, **kwargs):
    """
    移除异常值（基于Z-score）
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        threshold: Z-score阈值
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    def clip_outliers(X, threshold):
        # 计算每个特征的均值和标准差
        mean = np.mean(X, axis=(0, 1), keepdims=True)
        std = np.std(X, axis=(0, 1), keepdims=True)
        
        # 计算Z-score
        z_scores = np.abs((X - mean) / (std + 1e-8))
        
        # 裁剪异常值
        X_clipped = np.where(z_scores > threshold, 
                           mean + threshold * std * np.sign(X - mean), 
                           X)
        return X_clipped
    
    X_train_processed = clip_outliers(X_train, threshold)
    X_test_processed = clip_outliers(X_test, threshold)
    
    return X_train_processed, X_test_processed


def smooth_data(X_train, X_test, window_size=5, **kwargs):
    """
    数据平滑（移动平均）
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        window_size: 窗口大小
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    def apply_smoothing(X, window_size):
        X_smoothed = np.copy(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                X_smoothed[i, :, j] = signal.savgol_filter(X[i, :, j], 
                                                         min(window_size, X.shape[1]), 
                                                         polyorder=1)
        return X_smoothed
    
    X_train_processed = apply_smoothing(X_train, window_size)
    X_test_processed = apply_smoothing(X_test, window_size)
    
    return X_train_processed, X_test_processed