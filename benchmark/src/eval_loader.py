"""
评估加载器 (EvalLoader)
====================

负责动态加载和执行评估函数

"""

import importlib
import numpy as np
from typing import Dict, Any, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EvalLoadError(Exception):
    """评估加载异常"""
    pass


class EvalLoader:
    """
    评估加载器
    
    动态加载评估函数并执行评估
    """
    
    def __init__(self, evaluators_dir: Path = None):
        """初始化评估加载器"""
        if evaluators_dir is None:
            evaluators_dir = Path(__file__).parent.parent / "evaluators"
        
        self.evaluators_dir = Path(evaluators_dir)
        self.evaluator_registry = {}
        
        logger.info(f"评估加载器初始化完成，评估器目录: {self.evaluators_dir}")
    
    def evaluate(self, config: Dict[str, Any], eval_template_name: str,
                X_train: np.ndarray, y_train: np.ndarray, y_train_pred: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray, y_test_pred: np.ndarray) -> Dict[str, float]:
        """
        执行评估
        
        Args:
            config: 完整配置字典
            eval_template_name: 评估模板名称
            X_train, y_train, y_train_pred: 训练集数据和预测
            X_test, y_test, y_test_pred: 测试集数据和预测
            
        Returns:
            评估结果字典 {'metric_name': score}
        """
        # 获取评估配置
        if 'evaluation_templates' not in config:
            raise EvalLoadError("配置中缺少 'evaluation_templates'")
        
        if eval_template_name not in config['evaluation_templates']:
            available = list(config['evaluation_templates'].keys())
            raise EvalLoadError(f"评估模板 '{eval_template_name}' 不存在，可用: {available}")
        
        eval_config = config['evaluation_templates'][eval_template_name]
        metrics = eval_config.get('metrics', {})
        
        if not metrics:
            raise EvalLoadError(f"评估模板 '{eval_template_name}' 中没有定义指标")
        
        # 执行所有评估指标
        results = {}
        for metric_name, metric_config in metrics.items():
            evaluator_func = self._load_evaluator(metric_name, metric_config)
            try:
                score = evaluator_func(X_train, y_train, y_train_pred, 
                                     X_test, y_test, y_test_pred)
                results[metric_name] = float(score)
                logger.info(f"评估指标 '{metric_name}': {score:.4f}")
            except Exception as e:
                logger.error(f"评估指标 '{metric_name}' 执行失败: {e}")
                results[metric_name] = None
        
        return results
    
    def _load_evaluator(self, metric_name: str, metric_config: Dict[str, Any]) -> Callable:
        """动态加载评估函数"""
        evaluator_key = metric_name.lower()
        
        if evaluator_key not in self.evaluator_registry:
            try:
                # 确定模块名：配置中的file字段 或 默认使用metric_name
                module_name = metric_config.get('file', metric_name) if metric_config else metric_name
                
                # 确定函数名：配置中的function字段 或 默认使用'evaluate'
                function_name = metric_config.get('function', 'evaluate') if metric_config else 'evaluate'
                
                # 动态导入评估器模块
                module = importlib.import_module(f"evaluators.{module_name}")
                evaluator_func = getattr(module, function_name)
                self.evaluator_registry[evaluator_key] = evaluator_func
                logger.info(f"评估器 '{metric_name}' 加载成功 (模块: {module_name}, 函数: {function_name})")
            except ImportError:
                raise EvalLoadError(f"无法导入评估器模块: evaluators.{module_name}")
            except AttributeError:
                raise EvalLoadError(f"评估器模块 evaluators.{module_name} 中没有 '{function_name}' 函数")
        
        return self.evaluator_registry[evaluator_key]