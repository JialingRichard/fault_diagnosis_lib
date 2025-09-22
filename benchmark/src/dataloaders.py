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

# 假设 models.base_model.DataMetadata 已正确定义
from models.base_model import DataMetadata

logger = logging.getLogger(__name__)







class DataPipeline:
    """
    统一数据管道，负责协调数据加载和预处理。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据管道。
        
        Args:
            config: 包含 'data' 和 'preprocessing' 的配置字典。
        """
        self.config = config
        self.metadata = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # 默认预处理配置
        self.preprocessing_config = {
            'normalization': 'none',  # 'minmax', 'standard', or 'none'
        }
        
        # 从主配置更新预处理配置
        if 'preprocessing' in config:
            self.preprocessing_config.update(config['preprocessing'])
        else:
            logger.info("未提供预处理配置，使用默认值。")

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataMetadata]:
        """
        准备数据的主入口。
        
        Returns:
            (X_train, X_test, y_train, y_test, metadata)
        """
        logger.info("开始数据加载和预处理...")

        self.X_train, self.y_train, self.X_test, self.y_test, self.metadata = self.load_data()
        logger.info("加载器已提供预分割数据。")

        # 对数据进行标准化
        # 注意：用训练集拟合（fit）scaler，然后分别转换（transform）训练集和测试集
        scaler = self._get_scaler()
        if scaler:
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            logger.info(f"已对数据应用 '{self.preprocessing_config['normalization']}' 标准化。")
        else:
            print("No normalization applied.")
        logger.info("数据准备完成。")
        return self.X_train, self.X_test, self.y_train, self.y_test, self.metadata


    def _get_scaler(self):
        """根据配置获取一个 scaler 实例。"""
        normalization = self.preprocessing_config.get('normalization', 'none').lower()
        if normalization == 'minmax':
            return MinMaxScaler()
        elif normalization == 'standard':
            return StandardScaler()
        return None
    
    def _load_file(self, filename: str) -> np.ndarray:
        # get project root path
        project_root = Path(__file__).parent.parent
        print(f"Project root path: {project_root}")
        # concat project root with filename
        file_path = project_root / filename
        print(f"Loading file: {file_path}")
        # 直接根据路径加载 .npy 文件
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        if file_path.suffix == '.npy':
            # show path for debugging
            logger.debug(f"正在加载文件: {file_path}")
            return np.load(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataMetadata]:
        """加载预分割的数据文件。"""
        logger.info("加载预分割数据集...")

        # 从配置中读取文件名
        train_X = self._load_file(self.config['data']['train_data'])
        train_y = self._load_file(self.config['data']['train_label'])
        test_X = self._load_file(self.config['data']['test_data'])
        test_y = self._load_file(self.config['data']['test_label'])

        # 创建元数据
        metadata = DataMetadata(
            dataset_name=self.config['data']['dataset'],
            fault_type='binary' if len(np.unique(train_y)) <= 2 else 'multi-class',
            feature_dim=train_X.shape[2],
            num_classes=len(np.unique(np.concatenate((train_y.flatten(), test_y.flatten())))),
            timesteps=train_X.shape[1],
            number_train_samples=train_X.shape[0],
            number_test_samples=test_X.shape[0],
        )
        
        # print metadata for debugging
        print(f"Metadata: {metadata}")
        
        return train_X, train_y, test_X, test_y, metadata