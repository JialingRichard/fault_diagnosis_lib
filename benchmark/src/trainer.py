"""
模型训练器 (Trainer)
===================

统一的模型训练和验证接口，支持：
- 自动化训练流程管理
- 训练过程监控
- 模型保存和加载

Author: Fault Diagnosis Benchmark Team
Date: 2025-01-11
"""

import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import pickle
import time
from datetime import datetime

from models.base_model import BaseModel, ModelFactory, DataMetadata
from src.metrics import TimeSeriesEvaluator

logger = logging.getLogger(__name__)


class TrainingConfig:
    """训练配置类"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        
        # 保存配置
        self.save_models = config_dict.get('output', {}).get('save_models', False)
        self.models_dir = Path(config_dict.get('output', {}).get('models_dir', '../models'))
        
        # 日志配置
        self.verbose = config_dict.get('resources', {}).get('verbose', True)
        
        # 随机种子
        self.random_seed = config_dict.get('resources', {}).get('random_seed', 42)


class Trainer:
    """
    统一的模型训练器
    
    提供完整的训练流程管理
    """
    
    def __init__(self, config: TrainingConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.evaluator = TimeSeriesEvaluator()
        
        # 创建保存目录
        if self.config.save_models:
            self.config.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史记录
        self.training_history = []
        
        logger.info("训练器初始化完成")
    
    def train_single_model(self, model_name: str, model_config: Dict[str, Any],
                          train_data: np.ndarray, train_labels: np.ndarray,
                          val_data: Optional[np.ndarray] = None,
                          val_labels: Optional[np.ndarray] = None,
                          metadata: Optional[DataMetadata] = None) -> Dict[str, Any]:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            model_config: 模型配置
            train_data: 训练数据
            train_labels: 训练标签
            val_data: 验证数据（可选）
            val_labels: 验证标签（可选）
            metadata: 数据元数据
            
        Returns:
            训练结果字典
        """
        
        logger.info(f"开始训练模型: {model_name}")
        start_time = time.time()
        
        # 创建模型
        model = ModelFactory.create_model(model_name, model_config)
        
        # 训练模型
        try:
            model.fit(train_data, metadata)
            training_success = True
            error_message = None
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            training_success = False
            error_message = str(e)
            model = None
        
        training_time = time.time() - start_time
        
        # 验证（如果提供了验证集）
        val_results = {}
        if training_success and val_data is not None and val_labels is not None:
            try:
                val_scores = model.predict(val_data, metadata)
                val_results = self.evaluator.evaluate(val_labels, val_scores, metadata)
                logger.info(f"验证F1 Score: {val_results.get('f1', 'N/A'):.4f}")
                
            except Exception as e:
                logger.warning(f"验证失败: {e}")
                val_results = {'error': str(e)}
        
        # 保存模型（如果需要）
        model_path = None
        if self.config.save_models and training_success:
            model_path = self._save_model(model, model_name)
        
        # 记录训练结果
        result = {
            'model_name': model_name,
            'model_config': model_config,
            'training_success': training_success,
            'training_time': training_time,
            'error_message': error_message,
            'validation_results': val_results,
            'model_path': str(model_path) if model_path else None,
            'model_object': model if training_success else None,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(result)
        
        logger.info(f"模型训练完成: {model_name}, 耗时: {training_time:.2f}秒")
        return result
    
    def _save_model(self, model: BaseModel, model_name: str) -> Path:
        """保存模型"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.config.models_dir / model_filename
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"模型已保存到: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结"""
        
        if not self.training_history:
            return {'message': '没有训练记录'}
        
        total_models = len(self.training_history)
        successful_models = sum(1 for result in self.training_history 
                              if result['training_success'])
        
        total_time = sum(result['training_time'] for result in self.training_history)
        
        summary = {
            'total_models_trained': total_models,
            'successful_models': successful_models,
            'success_rate': successful_models / total_models if total_models > 0 else 0,
            'total_training_time': total_time,
            'average_training_time': total_time / total_models if total_models > 0 else 0,
            'training_history': self.training_history
        }
        
        return summary


# 便捷函数
def create_trainer(config_dict: Dict[str, Any]) -> Trainer:
    """创建训练器的便捷函数"""
    training_config = TrainingConfig(config_dict)
    return Trainer(training_config)