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
import importlib


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
        self.preprocessor_registry = {}  # 缓存已加载的预处理函数

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
        preprocessing_config = dataset_config.get('preprocessing', {})

        X_train, y_train, X_test, y_test, metadata = self._load_data(dataset_name, dataset_config)
        logger.info("加载器已提供预分割数据。")

        # 使用模块化预处理器
        try:
            X_train, X_test = self._apply_preprocessing(X_train, X_test, preprocessing_config)
            logger.info("模块化预处理完成。")
        except Exception as e:
            logger.warning(f"模块化预处理失败，回退到传统预处理: {e}")
            # 回退到传统预处理方法
            scaler = self._get_scaler(preprocessing_config)
            if scaler:
                original_shape_train = X_train.shape
                original_shape_test = X_test.shape
                X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
                X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
                X_train = X_train.reshape(original_shape_train)
                X_test = X_test.reshape(original_shape_test)
                
                if isinstance(scaler, MinMaxScaler):
                    scaler_type = "MinMax"
                elif isinstance(scaler, StandardScaler):
                    scaler_type = "Standard"
                else:
                    scaler_type = "Unknown"
                
                logger.info(f"已对数据应用 '{scaler_type}' 标准化。")
            else:
                print("No normalization applied.")
        
        logger.info("数据准备完成。")
        return X_train, X_test, y_train, y_test, metadata

    def _apply_preprocessing(self, X_train: np.ndarray, X_test: np.ndarray, 
                           preprocessing_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用模块化预处理步骤
        
        Args:
            X_train, X_test: 训练和测试数据
            preprocessing_config: 预处理配置
            
        Returns:
            (X_train_processed, X_test_processed)
        """
        if not preprocessing_config:
            logger.info("未配置预处理步骤，返回原始数据")
            return X_train, X_test
        
        # 记录原始数据形状
        logger.info(f"开始预处理 | 原始数据: train{X_train.shape}, test{X_test.shape}")
        
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # 按顺序应用每个预处理步骤
        for step_name, step_config in preprocessing_config.items():
            if step_name == 'steps':
                # 新的步骤列表格式
                for step in step_config:
                    step_name = step['name']
                    step_params = step.get('params', {})
                    file_name = step.get('file', None)
                    function_name = step.get('function', step_name)
                    
                    # 记录处理前的特征数
                    features_before = X_train_processed.shape[-1]
                    
                    preprocessor_func = self._load_preprocessor(function_name, file_name)
                    if preprocessor_func:
                        X_train_processed, X_test_processed = preprocessor_func(
                            X_train_processed, X_test_processed, **step_params)
                        
                        # 记录处理后的特征数
                        features_after = X_train_processed.shape[-1]
                        
                        # 构建详细的日志信息
                        param_str = ", ".join([f"{k}={v}" for k, v in step_params.items()]) if step_params else "无参数"
                        feature_change = f"{features_before}→{features_after}" if features_before != features_after else f"{features_before}"
                        
                        logger.info(f"✓ {step_name}: {file_name}.{function_name}({param_str}) | 特征数: {feature_change}")
            else:
                # 简化格式支持
                if self._is_simple_config(step_name, step_config):
                    # 记录处理前的特征数
                    features_before = X_train_processed.shape[-1]
                    
                    preprocessor_func = self._load_preprocessor_simple(step_name, step_config)
                    if preprocessor_func:
                        X_train_processed, X_test_processed = preprocessor_func(
                            X_train_processed, X_test_processed)
                        
                        # 记录处理后的特征数
                        features_after = X_train_processed.shape[-1]
                        feature_change = f"{features_before}→{features_after}" if features_before != features_after else f"{features_before}"
                        
                        logger.info(f"✓ {step_name}: {step_config} | 特征数: {feature_change}")
        
        # 记录最终数据形状
        logger.info(f"预处理完成 | 最终数据: train{X_train_processed.shape}, test{X_test_processed.shape}")
        
        return X_train_processed, X_test_processed
    
    def _is_simple_config(self, step_name: str, step_config: Any) -> bool:
        """判断是否为简化配置格式"""
        simple_configs = {
            'normalize': [True, False, 'minmax', 'standard', 'robust'],
            'add_noise': [True, False],
            'remove_outliers': [True, False],
            'smooth': [True, False]
        }
        return step_name in simple_configs and step_config in simple_configs[step_name]
    
    def _load_preprocessor_simple(self, step_name: str, step_config: Any):
        """加载简化配置的预处理器"""
        if step_name == 'normalize':
            if step_config is True or step_config == 'standard':
                return self._load_preprocessor('standard_normalize', 'normalizers')
            elif step_config == 'minmax':
                return self._load_preprocessor('minmax_normalize', 'normalizers')
            elif step_config == 'robust':
                return self._load_preprocessor('robust_normalize', 'normalizers')
            elif step_config is False:
                return self._load_preprocessor('no_normalize', 'normalizers')
        elif step_name == 'add_noise' and step_config is True:
            return self._load_preprocessor('add_gaussian_noise', 'noise_processors')
        elif step_name == 'remove_outliers' and step_config is True:
            return self._load_preprocessor('remove_outliers', 'noise_processors')
        elif step_name == 'smooth' and step_config is True:
            return self._load_preprocessor('smooth_data', 'noise_processors')
        
        return None
    
    def _load_preprocessor(self, function_name: str, module_name: str = None):
        """动态加载预处理函数"""
        if module_name is None:
            # 尝试从常见模块中查找
            common_modules = ['normalizers', 'noise_processors', 'feature_engineering']
            for mod in common_modules:
                try:
                    func = self._load_from_module(mod, function_name)
                    if func:
                        return func
                except:
                    continue
            logger.error(f"无法找到预处理器函数: {function_name}")
            return None
        
        return self._load_from_module(module_name, function_name)
    
    def _load_from_module(self, module_name: str, function_name: str):
        """从指定模块加载函数"""
        cache_key = f"{module_name}.{function_name}"
        
        # 检查缓存
        if cache_key in self.preprocessor_registry:
            return self.preprocessor_registry[cache_key]
        
        try:
            # 动态导入模块
            module = importlib.import_module(f"preprocessors.{module_name}")
            
            # 获取函数
            if hasattr(module, function_name):
                preprocessor_func = getattr(module, function_name)
                self.preprocessor_registry[cache_key] = preprocessor_func
                logger.debug(f"预处理器函数 '{function_name}' 加载成功 (模块: {module_name})")
                return preprocessor_func
            else:
                logger.error(f"模块 '{module_name}' 中未找到函数 '{function_name}'")
                return None
                
        except ImportError:
            logger.error(f"无法导入预处理器模块: preprocessors.{module_name}")
            return None
        except Exception as e:
            logger.error(f"加载预处理器函数失败: {module_name}.{function_name}, 错误: {str(e)}")
            return None

    def _get_scaler(self, preprocessing_config: Dict[str, Any]):
        """根据配置获取一个 scaler 实例。"""
        # 支持两种配置格式：
        # 1. normalize: true/false (简化格式)
        # 2. normalization: 'minmax'/'standard'/'none' (详细格式)
        
        # 优先检查简化格式
        if 'normalize' in preprocessing_config:
            normalize = preprocessing_config.get('normalize', False)
            if normalize is True:
                # 默认使用 standard 标准化
                return StandardScaler()
            elif normalize is False:
                return None
            elif isinstance(normalize, str):
                # 如果 normalize 是字符串，作为 normalization 类型处理
                normalization = normalize.lower()
            else:
                return None
        else:
            # 检查详细格式
            normalization = preprocessing_config.get('normalization', 'none').lower()
        
        # 处理标准化类型
        if normalization == 'minmax':
            return MinMaxScaler()
        elif normalization in ['standard', 'z-score']:
            return StandardScaler()
        elif normalization in ['none', 'false']:
            return None
        else:
            logger.warning(f"未知的标准化类型: {normalization}，跳过标准化")
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