"""
Data Loaders for Time Series Anomaly Detection Benchmark
========================================================

This module provides data loading and preprocessing functionality for various
time series anomaly detection datasets.

Author: Fault Diagnosis Benchmark Team
Date: 2025-01-11
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, Optional, Tuple, List
import logging
import os
from pathlib import Path

from .models.base_model import DataMetadata

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """数据加载器抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器
        
        Args:
            config: 数据加载配置
        """
        self.config = config
        self.metadata = None
        
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, DataMetadata]:
        """
        加载数据
        
        Returns:
            (X, y, metadata): 特征数据，标签数据，元数据
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> DataMetadata:
        """获取数据元数据"""
        pass
    
    def _validate_data_path(self, path: str) -> bool:
        """验证数据路径是否存在"""
        return os.path.exists(path)


class SwatDataLoader(BaseDataLoader):
    """
    SWAT数据集加载器
    
    SWAT (Secure Water Treatment) 是一个工业控制系统安全数据集，
    包含正常运行和网络攻击场景下的传感器数据。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 默认配置
        self.default_config = {
            'path': './data/swat',
            'train_file': 'SWaT_Dataset_Normal_v1.csv',
            'test_file': 'SWaT_Dataset_Attack_v0.csv',
            'use_mock_data': False,
            'mock_samples': 10000,
            'mock_features': 51,
            'contamination_ratio': 0.1
        }
        
        self.data_config = {**self.default_config, **config}
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, DataMetadata]:
        """加载SWAT数据集"""
        
        if self.data_config['use_mock_data'] or not self._check_real_data_available():
            logger.info("使用模拟SWAT数据")
            return self._generate_mock_data()
        else:
            logger.info("加载真实SWAT数据集")
            return self._load_real_data()
    
    def _check_real_data_available(self) -> bool:
        """检查真实数据是否可用"""
        data_path = Path(self.data_config['path'])
        train_file = data_path / self.data_config['train_file']
        test_file = data_path / self.data_config['test_file']
        
        return train_file.exists() and test_file.exists()
    
    def _load_real_data(self) -> Tuple[np.ndarray, np.ndarray, DataMetadata]:
        """加载真实SWAT数据"""
        data_path = Path(self.data_config['path'])
        train_file = data_path / self.data_config['train_file']
        test_file = data_path / self.data_config['test_file']
        
        try:
            logger.info(f"正在加载SWAT数据集...")
            logger.info(f"训练文件: {train_file}")
            logger.info(f"测试文件: {test_file}")
            
            # 加载数据
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            
            # 预处理
            X_train, y_train = self._preprocess_swat_data(train_df, is_train=True)
            X_test, y_test = self._preprocess_swat_data(test_df, is_train=False)
            
            # 合并数据
            X = np.vstack([X_train, X_test])
            y = np.hstack([y_train, y_test])
            
            # 创建元数据
            metadata = DataMetadata(
                label_granularity="point-wise",
                fault_type="binary",
                num_classes=2,
                sequence_length=len(X),
                feature_dim=X.shape[1],
                dataset_name="swat"
            )
            
            logger.info(f"✅ SWAT数据集加载成功:")
            logger.info(f"   - 总样本数: {X.shape[0]}")
            logger.info(f"   - 特征维度: {X.shape[1]}")
            logger.info(f"   - 正常样本: {np.sum(y == 0)}")
            logger.info(f"   - 异常样本: {np.sum(y == 1)}")
            
            self.metadata = metadata
            return X, y, metadata
            
        except Exception as e:
            logger.error(f"加载真实SWAT数据失败: {e}")
            logger.info("回退到模拟数据")
            return self._generate_mock_data()
    
    def _preprocess_swat_data(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """预处理SWAT数据"""
        
        # 移除不需要的列
        columns_to_remove = ['Timestamp']
        if 'Normal/Attack' in df.columns:
            label_col = 'Normal/Attack'
            columns_to_remove.append(label_col)
        else:
            label_col = None
        
        # 提取特征
        feature_cols = [col for col in df.columns if col not in columns_to_remove]
        X = df[feature_cols].values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        
        # 创建标签
        if is_train:
            # 训练数据通常全为正常
            y = np.zeros(len(X))
        else:
            # 测试数据根据标签列判断
            if label_col and label_col in df.columns:
                y = (df[label_col] == 'Attack').astype(int)
            else:
                y = np.zeros(len(X))
        
        return X, y
    
    def _generate_mock_data(self) -> Tuple[np.ndarray, np.ndarray, DataMetadata]:
        """生成模拟SWAT数据"""
        
        n_samples = self.data_config['mock_samples']
        n_features = self.data_config['mock_features']
        contamination = self.data_config['contamination_ratio']
        
        # 生成正常数据（多元高斯分布）
        np.random.seed(42)
        
        # 生成相关特征
        mean = np.random.randn(n_features) * 0.5
        cov = np.random.randn(n_features, n_features)
        cov = np.dot(cov, cov.T) + np.eye(n_features) * 0.1
        
        X_normal = np.random.multivariate_normal(mean, cov, int(n_samples * (1 - contamination)))
        
        # 生成异常数据（偏移的高斯分布）
        n_anomalies = int(n_samples * contamination)
        anomaly_shift = np.random.randn(n_features) * 3
        X_anomaly = np.random.multivariate_normal(mean + anomaly_shift, cov, n_anomalies)
        
        # 合并数据
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
        
        # 打乱数据
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # 创建元数据
        metadata = DataMetadata(
            label_granularity="point-wise",
            fault_type="binary",
            num_classes=2,
            sequence_length=len(X),
            feature_dim=n_features,
            dataset_name="swat_mock"
        )
        
        logger.info(f"✅ 模拟SWAT数据生成完成:")
        logger.info(f"   - 总样本数: {X.shape[0]}")
        logger.info(f"   - 特征维度: {X.shape[1]}")
        logger.info(f"   - 正常样本: {np.sum(y == 0)}")
        logger.info(f"   - 异常样本: {np.sum(y == 1)}")
        
        self.metadata = metadata
        return X, y, metadata
    
    def get_metadata(self) -> DataMetadata:
        """获取数据元数据"""
        return self.metadata


class SMDDataLoader(BaseDataLoader):
    """
    SMD (Server Machine Dataset) 数据集加载器
    
    SMD数据集包含服务器性能监控数据，用于检测服务器异常。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # TODO: 实现SMD数据加载逻辑
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, DataMetadata]:
        raise NotImplementedError("SMD data loader not implemented yet")
    
    def get_metadata(self) -> DataMetadata:
        return self.metadata


class DataPipeline:
    """
    统一数据管道
    
    负责协调数据加载、预处理和数据分割。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据管道
        
        Args:
            config: 包含数据集和预处理配置的字典
        """
        self.config = config
        self.metadata = None
        
        # 默认预处理配置
        self.preprocessing_config = {
            'normalization': 'minmax',  # 'minmax', 'standard', 'none'
            'test_size': 0.3,
            'random_state': 42,
            'stratify': True
        }
        
        # 更新预处理配置
        if 'preprocessing' in config:
            self.preprocessing_config.update(config['preprocessing'])
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataMetadata]:
        """
        准备数据的主要入口
        
        Returns:
            (X_train, X_test, y_train, y_test, metadata)
        """
        
        # 1. 加载原始数据
        logger.info("开始数据加载和预处理...")
        loader = self._get_data_loader()
        X, y, self.metadata = loader.load_data()
        
        # 2. 数据预处理
        X_processed = self._preprocess_features(X)
        
        # 3. 数据分割
        X_train, X_test, y_train, y_test = self._split_data(X_processed, y)
        
        logger.info("数据准备完成")
        return X_train, X_test, y_train, y_test, self.metadata
    
    def _get_data_loader(self) -> BaseDataLoader:
        """根据配置创建数据加载器"""
        dataset_name = self.config['dataset']['name'].lower()
        dataset_config = self.config['dataset']
        
        # 数据加载器映射
        loader_map = {
            'swat': SwatDataLoader,
            'smd': SMDDataLoader,
            # 可以添加更多数据集
        }
        
        if dataset_name not in loader_map:
            available_datasets = list(loader_map.keys())
            raise ValueError(f"不支持的数据集: {dataset_name}. "
                           f"可用数据集: {available_datasets}")
        
        loader_class = loader_map[dataset_name]
        return loader_class(dataset_config)
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """特征预处理"""
        
        normalization = self.preprocessing_config['normalization']
        
        if normalization == 'minmax':
            scaler = MinMaxScaler()
            X_processed = scaler.fit_transform(X)
            logger.info("应用MinMax标准化")
            
        elif normalization == 'standard':
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X)
            logger.info("应用标准标准化")
            
        else:
            X_processed = X.copy()
            logger.info("未应用特征标准化")
        
        return X_processed
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """数据分割"""
        
        test_size = self.preprocessing_config['test_size']
        random_state = self.preprocessing_config['random_state']
        stratify = self.preprocessing_config['stratify']
        
        # 判断是否需要分层采样
        stratify_param = y if stratify and len(np.unique(y)) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        logger.info(f"数据分割完成:")
        logger.info(f"  - 训练集: {X_train.shape[0]} 样本")
        logger.info(f"  - 测试集: {X_test.shape[0]} 样本")
        logger.info(f"  - 训练集异常率: {np.mean(y_train):.3f}")
        logger.info(f"  - 测试集异常率: {np.mean(y_test):.3f}")
        
        return X_train, X_test, y_train, y_test


# 数据加载器工厂
class DataLoaderFactory:
    """数据加载器工厂"""
    
    _loaders = {
        'swat': SwatDataLoader,
        'smd': SMDDataLoader
    }
    
    @classmethod
    def create_loader(cls, dataset_name: str, config: Dict[str, Any]) -> BaseDataLoader:
        """创建数据加载器"""
        dataset_name = dataset_name.lower()
        
        if dataset_name not in cls._loaders:
            available_loaders = list(cls._loaders.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available datasets: {available_loaders}")
        
        loader_class = cls._loaders[dataset_name]
        return loader_class(config)
    
    @classmethod
    def list_available_datasets(cls) -> List[str]:
        """列出所有可用的数据集"""
        return list(cls._loaders.keys())
