"""
Base Model Abstract Class for Time Series Anomaly Detection
============================================================

This module provides the abstract base class for all anomaly detection models
used in the benchmark framework.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMetadata:
    """
    一个灵活的数据元数据容器，其行为类似一个开放的属性包。
    可以在初始化时传入任意关键字参数，这些参数都会成为它的属性。
    """
    def __init__(self, **kwargs):
        # 将所有传入的关键字参数动态地设置为实例的属性
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        # 创建一个可读的字符串表示，显示所有属性
        items = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"DataMetadata({items})"

    def __repr__(self):
        return self.__str__()

    def get(self, key: str, default: Any = None) -> Any:
        """提供类似字典的 .get() 方法，用于安全地访问属性。"""
        return getattr(self, key, default)


class BaseModel(ABC):
    """统一模型抽象基类
    
    这个基类定义了所有异常检测模型必须实现的接口，
    确保不同类型的模型（传统ML和深度学习）能够统一处理。
    """
    
    def __init__(self, config: Dict[str, Any], model_name: str = None):
        """
        初始化模型
        
        Args:
            config: 模型配置参数字典
            model_name: 模型名称（可选）
        """
        self.config = config
        self.model_name = model_name or self.__class__.__name__
        self.model = None
        self.is_trained = False
        self._training_history = []
        
        logger.info(f"初始化模型: {self.model_name}")
        
    @property
    @abstractmethod
    def requires_training_loop(self) -> bool:
        """指示是否需要复杂的训练循环（深度学习模型为True）"""
        pass
    
    @property
    @abstractmethod
    def supports_online_learning(self) -> bool:
        """指示是否支持在线学习"""
        pass
    
    @abstractmethod
    def build_model(self, input_shape: tuple, metadata: Optional[DataMetadata] = None) -> None:
        """
        构建模型
        
        Args:
            input_shape: 输入数据形状
            metadata: 数据元数据（可选）
        """
        pass
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None, 
            metadata: Optional[DataMetadata] = None) -> None:
        """
        训练模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签（可选，用于有监督模型）
            metadata: 数据元数据（可选）
        """
        pass
    
    @abstractmethod
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数
        
        Args:
            X: 输入数据
            
        Returns:
            异常分数数组，值越高表示越异常
        """
        pass
    
    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        预测异常标签
        
        Args:
            X: 输入数据
            threshold: 异常检测阈值
            
        Returns:
            预测标签，0为正常，1为异常
        """
        scores = self.predict_anomaly_score(X)
        
        if threshold is None:
            # 如果没有提供阈值，使用默认策略（如95分位数）
            threshold = np.percentile(scores, 95)
            
        return (scores > threshold).astype(int)
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        import joblib
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_history': self._training_history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self._training_history = model_data.get('training_history', [])
        
        logger.info(f"模型已从 {filepath} 加载")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.model_name,
            'class_name': self.__class__.__name__,
            'requires_training_loop': self.requires_training_loop,
            'supports_online_learning': self.supports_online_learning,
            'is_trained': self.is_trained,
            'config': self.config,
            'training_history': self._training_history
        }
    
    def get_training_history(self) -> list:
        """获取训练历史"""
        return self._training_history
    
    def _add_training_record(self, record: Dict[str, Any]) -> None:
        """添加训练记录"""
        self._training_history.append(record)
    
    def __str__(self):
        return f"{self.model_name}(trained={self.is_trained})"
    
    def __repr__(self):
        return self.__str__()


class ModelFactory:
    """模型工厂类，用于动态创建模型实例"""
    
    _models = {}
    
    @classmethod
    def register_model(cls, model_name: str, model_class: type):
        """
        注册新的模型类
        
        Args:
            model_name: 模型名称
            model_class: 模型类
        """
        cls._models[model_name.lower()] = model_class
        logger.info(f"注册模型: {model_name}")
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> BaseModel:
        """
        创建指定类型的模型
        
        Args:
            model_type: 模型类型
            config: 模型配置
            
        Returns:
            模型实例
        """
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available models: {available_models}")
        
        model_class = cls._models[model_type]
        return model_class(config)
    
    @classmethod
    def list_available_models(cls) -> list:
        """列出所有可用的模型类型"""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """获取模型类信息"""
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        
        return {
            'name': model_type,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'doc': model_class.__doc__
        }
