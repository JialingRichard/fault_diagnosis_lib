"""
Data Loaders for Time Series Anomaly Detection Benchmark
========================================================

This module provides a unified and configuration-driven data loading
and preprocessing pipeline.

Author: Fault Diagnosis Benchmark Team
Date: 2025-09-17
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, Tuple
from pathlib import Path


logger = logging.getLogger(__name__)


# Metadata

class DataMetadata:
    """
    A flexible container for dataset metadata.
    Behaves like an open attribute bag.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        items = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"DataMetadata({items})"

    def __repr__(self):
        return self.__str__()

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

class DataLoader:
    """
    统一数据加载器，负责协调数据加载和预处理。
    """
    def __init__(self):
        """初始化数据加载器。"""
        pass

    def prepare_data(self, config: Dict[str, Any], dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataMetadata]:
        """
        准备数据的主入口。
        
        Args:
            config: 完整配置字典
            dataset_name: 数据集名称
            
        Returns:
            (X_train, X_test, y_train, y_test, metadata)
        """
        logger.info("开始数据加载和预处理...")

        dataset_config = config['datasets'][dataset_name]
        preprocessing_config = dataset_config.get('preprocessing', {'normalization': 'none'})

        X_train, y_train, X_test, y_test, metadata = self._load_data(dataset_name, dataset_config)
        logger.info("加载器已提供预分割数据。")

        # 对数据进行标准化
        scaler = self._get_scaler(preprocessing_config)
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logger.info(f"已对数据应用 '{preprocessing_config['normalization']}' 标准化。")
        else:
            print("No normalization applied.")
        logger.info("数据准备完成。")
        return X_train, X_test, y_train, y_test, metadata


    def _get_scaler(self, preprocessing_config: Dict[str, Any]):
        """根据配置获取一个 scaler 实例。"""
        normalization = preprocessing_config.get('normalization', 'none').lower()
        if normalization == 'minmax':
            return MinMaxScaler()
        elif normalization == 'standard':
            return StandardScaler()
        return None
    
    def _load_file(self, filename: str) -> np.ndarray:
        project_root = Path(__file__).parent.parent
        file_path = project_root / filename
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        if file_path.suffix == '.npy':
            return np.load(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def _load_data(self, dataset_name: str, dataset_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataMetadata]:
        """加载预分割的数据文件。"""
        logger.info("加载预分割数据集...")

        # 从配置中读取文件名
        train_X = self._load_file(dataset_config['train_data'])
        train_y = self._load_file(dataset_config['train_label'])
        test_X = self._load_file(dataset_config['test_data'])
        test_y = self._load_file(dataset_config['test_label'])

        # 创建元数据
        metadata = DataMetadata(
            dataset_name=dataset_name,
            fault_type='binary' if len(np.unique(train_y)) <= 2 else 'multi-class',
            feature_dim=train_X.shape[2],
            num_classes=len(np.unique(np.concatenate((train_y.flatten(), test_y.flatten())))),
            timesteps=train_X.shape[1],
            number_train_samples=train_X.shape[0],
            number_test_samples=test_X.shape[0],
        )
        
        return train_X, train_y, test_X, test_y, metadata