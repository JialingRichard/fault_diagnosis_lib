"""
Isolation Forest Model Implementation
=====================================

This module provides an Isolation Forest wrapper for the benchmark framework.

Author: Fault Diagnosis Benchmark Team  
Date: 2025-01-11
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
import logging

from .base_model import BaseModel, DataMetadata

logger = logging.getLogger(__name__)


class IsolationForestModel(BaseModel):
    """
    Isolation Forest 异常检测模型包装器
    
    这是一个基于sklearn的Isolation Forest模型的包装器，
    适用于无监督异常检测任务。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Isolation Forest模型
        
        Args:
            config: 模型配置参数
        """
        super().__init__(config, "IsolationForest")
        
        # 默认参数
        self.default_config = {
            'contamination': 0.1,
            'n_estimators': 100,
            'max_samples': 'auto',
            'max_features': 1.0,
            'bootstrap': False,
            'random_state': 42,
            'n_jobs': -1,
            'normalize': True
        }
        
        # 合并配置
        self.model_config = {**self.default_config, **config}
        
        # 预处理器
        self.scaler = None if not self.model_config['normalize'] else StandardScaler()
        
        logger.info(f"IsolationForest配置: {self.model_config}")
    
    @property
    def requires_training_loop(self) -> bool:
        """Isolation Forest不需要复杂训练循环"""
        return False
    
    @property 
    def supports_online_learning(self) -> bool:
        """Isolation Forest不支持在线学习"""
        return False
    
    def build_model(self, input_shape: tuple, metadata: Optional[DataMetadata] = None) -> None:
        """
        构建Isolation Forest模型
        
        Args:
            input_shape: 输入数据形状
            metadata: 数据元数据
        """
        # 移除自定义配置项
        sklearn_config = {k: v for k, v in self.model_config.items() 
                         if k not in ['normalize']}
        
        self.model = SklearnIsolationForest(**sklearn_config)
        
        logger.info(f"Isolation Forest模型构建完成，输入形状: {input_shape}")
        
        if metadata:
            logger.info(f"数据集信息: {metadata}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None, 
            metadata: Optional[DataMetadata] = None) -> None:
        """
        训练Isolation Forest模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签（用于选择正常样本，可选）
            metadata: 数据元数据
        """
        if not self.model:
            self.build_model(X_train.shape, metadata)
        
        # 数据预处理
        X_processed = self._preprocess_data(X_train, fit_scaler=True)
        
        # 对于无监督模型，可以选择只用正常数据训练
        if y_train is not None:
            # 如果提供了标签，只使用正常数据进行训练
            normal_mask = (y_train == 0)
            if np.any(normal_mask):
                X_processed = X_processed[normal_mask]
                logger.info(f"使用 {np.sum(normal_mask)} 个正常样本训练模型")
            else:
                logger.warning("没有找到正常样本标签，使用所有数据训练")
        
        # 训练模型
        logger.info("开始训练Isolation Forest...")
        self.model.fit(X_processed)
        
        self.is_trained = True
        
        # 记录训练信息
        training_record = {
            'timestamp': np.datetime64('now').astype(str),
            'training_samples': len(X_processed),
            'contamination': self.model_config['contamination'],
            'n_estimators': self.model_config['n_estimators']
        }
        self._add_training_record(training_record)
        
        logger.info("Isolation Forest训练完成")
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数
        
        Args:
            X: 输入数据
            
        Returns:
            异常分数数组，值越高表示越异常
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit()方法")
        
        # 数据预处理
        X_processed = self._preprocess_data(X, fit_scaler=False)
        
        # 获取异常分数
        # Isolation Forest的decision_function返回负的异常分数
        # 我们将其转换为正值，值越大越异常
        scores = -self.model.decision_function(X_processed)
        
        return scores
    
    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        预测异常标签
        
        Args:
            X: 输入数据
            threshold: 异常检测阈值（可选）
            
        Returns:
            预测标签，0为正常，1为异常
        """
        if threshold is not None:
            # 使用自定义阈值
            scores = self.predict_anomaly_score(X)
            return (scores > threshold).astype(int)
        else:
            # 使用模型内置的预测方法
            X_processed = self._preprocess_data(X, fit_scaler=False)
            predictions = self.model.predict(X_processed)
            # sklearn的Isolation Forest返回1为正常，-1为异常
            # 我们将其转换为0正常，1异常
            return (predictions == -1).astype(int)
    
    def _preprocess_data(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """
        数据预处理
        
        Args:
            X: 输入数据
            fit_scaler: 是否拟合标准化器
            
        Returns:
            预处理后的数据
        """
        if self.scaler is not None:
            if fit_scaler:
                X_processed = self.scaler.fit_transform(X)
                logger.info("数据标准化器已拟合")
            else:
                X_processed = self.scaler.transform(X)
        else:
            X_processed = X.copy()
        
        return X_processed
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性（基于异常路径长度）
        
        Returns:
            特征重要性字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # Isolation Forest没有直接的特征重要性
        # 这里返回一个占位符
        n_features = len(self.model.estimators_[0].tree_.feature)
        
        return {f'feature_{i}': 1.0/n_features for i in range(n_features)}
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        if not self.model:
            return self.model_config
        
        return {
            **self.model_config,
            'n_features': getattr(self.model, 'n_features_in_', None),
            'offset': getattr(self.model, 'offset_', None)
        }


# 注册模型到工厂
from .base_model import ModelFactory
ModelFactory.register_model('isolation_forest', IsolationForestModel)
ModelFactory.register_model('iforest', IsolationForestModel)  # 别名
