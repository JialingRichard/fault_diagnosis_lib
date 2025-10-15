"""
训练加载器 (TrainingLoader)
========================

负责根据配置动态加载不同的训练策略

"""

import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TrainingLoadError(Exception):
    """训练加载异常"""
    pass


class TrainingLoader:
    """
    训练加载器
    
    根据配置动态选择训练策略和创建训练器
    """
    
    def __init__(self, trainers_dir: Path = None):
        """初始化训练加载器"""
        if trainers_dir is None:
            trainers_dir = Path(__file__).parent.parent / "trainers"
        
        self.trainers_dir = Path(trainers_dir)
        self.trainer_registry = {}
        
        logger.debug(f"训练加载器初始化完成，训练器目录: {self.trainers_dir}")
    
    def create_trainer(self, config: Dict[str, Any], training_template_name: str, 
                      model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray):
        """
        创建训练器
        
        Args:
            config: 完整配置字典
            training_template_name: 训练模板名称
            model: 待训练的模型
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据（用于验证）
            
        Returns:
            训练器实例
        """
        # 获取训练配置
        if 'training_templates' not in config:
            raise TrainingLoadError("配置中缺少 'training_templates'")
        
        if training_template_name not in config['training_templates']:
            available = list(config['training_templates'].keys())
            raise TrainingLoadError(f"训练模板 '{training_template_name}' 不存在，可用: {available}")
        
        training_config = config['training_templates'][training_template_name]
        training_type = training_config.get('type', 'supervised')
        
        # 根据训练类型加载对应的训练器
        trainer_class = self._load_trainer_class(training_type, training_config)
        
        # 创建训练器实例
        # 从全局配置获取设备
        device = (config.get('global') or {}).get('device', config.get('device', 'cpu'))
        trainer = trainer_class(
            model=model,
            config=training_config,
            device=device,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            full_config=config  # 传递完整配置
        )
        
        logger.debug(f"训练器创建成功，类型: {training_type}")
        return trainer
    
    def _load_trainer_class(self, training_type: str, training_config: Dict[str, Any]):
        """动态加载训练器类"""
        trainer_key = training_type.lower()
        
        if trainer_key not in self.trainer_registry:
            try:
                # 确定模块名：配置中的trainer_file字段 或 默认使用training_type
                module_name = training_config.get('trainer_file', f"{training_type}_trainer")
                
                # 确定类名：配置中的trainer_class字段 或 默认使用类型名+Trainer
                class_name = training_config.get('trainer_class', f"{training_type.title()}Trainer")
                
                # 动态导入训练器模块
                module = importlib.import_module(f"trainers.{module_name}")
                trainer_class = getattr(module, class_name)
                self.trainer_registry[trainer_key] = trainer_class
                logger.debug(f"训练器 '{training_type}' 加载成功 (模块: {module_name}, 类: {class_name})")
            except ImportError:
                raise TrainingLoadError(f"无法导入训练器模块: trainers.{module_name}")
            except AttributeError:
                raise TrainingLoadError(f"训练器模块 trainers.{module_name} 中没有 '{class_name}' 类")
        
        return self.trainer_registry[trainer_key]
