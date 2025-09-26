"""
特征工程预处理器
==============

提供各种特征工程方法
"""

import numpy as np
from scipy.stats import skew, kurtosis


def add_statistical_features(X_train, X_test, **kwargs):
    """
    添加统计特征（均值、标准差、偏度、峰度）
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    def extract_stats(X):
        # 计算统计特征
        mean_feat = np.mean(X, axis=1, keepdims=True)
        std_feat = np.std(X, axis=1, keepdims=True)
        
        # 使用 warnings.catch_warnings() 抑制 scipy 的数值精度警告
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Precision loss occurred in moment calculation')
            
            # 计算偏度和峰度，并处理可能的数值问题
            skew_vals = skew(X, axis=1, nan_policy='omit')
            kurt_vals = kurtosis(X, axis=1, nan_policy='omit')
            
            # 将 NaN 和 Inf 替换为 0
            skew_vals = np.nan_to_num(skew_vals, nan=0.0, posinf=0.0, neginf=0.0)
            kurt_vals = np.nan_to_num(kurt_vals, nan=0.0, posinf=0.0, neginf=0.0)
            
            skew_feat = skew_vals.reshape(X.shape[0], 1, X.shape[2])
            kurt_feat = kurt_vals.reshape(X.shape[0], 1, X.shape[2])
        
        # 将统计特征添加到原始数据中
        X_extended = np.concatenate([X, mean_feat, std_feat, skew_feat, kurt_feat], axis=1)
        return X_extended
    
    X_train_processed = extract_stats(X_train)
    X_test_processed = extract_stats(X_test)
    
    return X_train_processed, X_test_processed


def sliding_window(X_train, X_test, window_size=10, stride=1, **kwargs):
    """
    滑动窗口特征提取
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        window_size: 窗口大小
        stride: 步长
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    def create_windows(X, window_size, stride):
        n_samples, seq_len, n_features = X.shape
        n_windows = (seq_len - window_size) // stride + 1
        
        if n_windows <= 0:
            return X  # 如果窗口大小大于序列长度，返回原数据
        
        windowed_data = np.zeros((n_samples, n_windows, window_size, n_features))
        
        for i in range(n_samples):
            for j in range(n_windows):
                start_idx = j * stride
                end_idx = start_idx + window_size
                windowed_data[i, j] = X[i, start_idx:end_idx]
        
        # 重新整形为 (n_samples, n_windows * window_size, n_features)
        windowed_data = windowed_data.reshape(n_samples, n_windows * window_size, n_features)
        return windowed_data
    
    X_train_processed = create_windows(X_train, window_size, stride)
    X_test_processed = create_windows(X_test, window_size, stride)
    
    return X_train_processed, X_test_processed


def pca_transform(X_train, X_test, n_components=0.95, **kwargs):
    """
    PCA降维
    
    Args:
        X_train: 训练数据
        X_test: 测试数据
        n_components: 保留的主成分数量或比例
        **kwargs: 额外参数
        
    Returns:
        (X_train_processed, X_test_processed)
    """
    from sklearn.decomposition import PCA
    
    # 重塑数据以适应PCA
    original_train_shape = X_train.shape
    original_test_shape = X_test.shape
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # 应用PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    
    # 重塑回3D格式（假设时间步为1）
    X_train_processed = X_train_pca.reshape(X_train_pca.shape[0], 1, X_train_pca.shape[1])
    X_test_processed = X_test_pca.reshape(X_test_pca.shape[0], 1, X_test_pca.shape[1])
    
    return X_train_processed, X_test_processed