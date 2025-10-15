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
        
        # 评估上下文信息
        self.plots_dir = None
        self.epoch_info = None  # {'epoch': 10} 表示在训练的第10轮，None表示最终评估
        self.logging_level = 'normal'
        
        logger.debug(f"评估加载器初始化完成，评估器目录: {self.evaluators_dir}")
    
    def set_context(self, plots_dir=None, epoch_info=None, logging_level='normal'):
        """
        设置评估上下文信息
        
        Args:
            plots_dir: 图片保存目录
            epoch_info: epoch信息字典，如 {'epoch': 10}，None表示最终评估
            logging_level: 日志等级 ('minimal', 'normal', 'verbose')
        """
        self.plots_dir = Path(plots_dir) if plots_dir else None
        self.epoch_info = epoch_info
        self.logging_level = logging_level
    
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
                result = evaluator_func(X_train, y_train, y_train_pred, 
                                      X_test, y_test, y_test_pred)
                
                # 处理不同类型的返回值
                processed_result = self._process_evaluator_result(
                    result, metric_name, metric_config
                )
                results[metric_name] = processed_result
                
                # 根据结果类型决定日志输出
                if isinstance(processed_result, (int, float)):
                    logger.info(f"评估指标 '{metric_name}': {processed_result:.4f}")
                elif isinstance(processed_result, str):
                    logger.info(f"评估指标 '{metric_name}': {processed_result}")
                else:
                    logger.info(f"评估指标 '{metric_name}': 已处理")
                    
            except Exception as e:
                logger.error(f"评估指标 '{metric_name}' 执行失败: {e}")
                results[metric_name] = None
        
        return results
    
    def _process_evaluator_result(self, result, metric_name: str, metric_config: Dict[str, Any]):
        """
        处理评估器返回值
        
        Args:
            result: 评估器返回值（可以是数值、字符串或Figure对象）
            metric_name: 指标名称
            metric_config: 指标配置
            
        Returns:
            处理后的结果（数值或字符串）
        """
        import matplotlib.figure
        
        # 数值型：直接返回
        if isinstance(result, (int, float, np.number)):
            return float(result)
        
        # 字符串型：直接返回
        if isinstance(result, str):
            return result
        
        # 图像型：保存图像并返回样本数或其他数值
        if isinstance(result, matplotlib.figure.Figure):
            return self._save_figure(result, metric_name, metric_config)
        
        # 元组型：(figure, metric_value)
        if isinstance(result, tuple) and len(result) == 2:
            fig, value = result
            if isinstance(fig, matplotlib.figure.Figure):
                self._save_figure(fig, metric_name, metric_config)
                return float(value) if isinstance(value, (int, float, np.number)) else value
        
        # 其他类型：尝试转换为float
        try:
            return float(result)
        except:
            logger.warning(f"无法处理评估器 '{metric_name}' 的返回值类型: {type(result)}")
            return None
    
    def _save_figure(self, fig, metric_name: str, metric_config: Dict[str, Any]) -> float:
        """
        保存matplotlib图像
        
        Args:
            fig: matplotlib Figure对象
            metric_name: 指标名称
            metric_config: 指标配置
            
        Returns:
            保存成功返回1.0，失败返回0.0
        """
        if self.plots_dir is None:
            if self.logging_level in ['normal', 'verbose']:
                logger.warning(f"未设置plots_dir，无法保存图像: {metric_name}")
            import matplotlib.pyplot as plt
            plt.close(fig)
            return 0.0
        
        try:
            import os
            
            # 确定保存路径
            if self.epoch_info is not None:
                # 训练过程中：保存到epochinfo子目录
                epoch_num = self.epoch_info.get('epoch', 0)
                epochinfo_dir = self.plots_dir / 'epochinfo'
                epochinfo_dir.mkdir(parents=True, exist_ok=True)
                
                # 文件名：metric_name_epoch_XXX.png
                filename = f"{metric_name}_epoch_{epoch_num+1:03d}.png"
                save_path = epochinfo_dir / filename
            else:
                # 最终评估：保存到主plots目录
                filename = f"{metric_name}.png"
                save_path = self.plots_dir / filename
            
            # 保存图像
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # 根据日志等级决定是否显示保存信息
            if self.logging_level in ['normal', 'verbose']:
                logger.info(f"图像已保存: {save_path}")
            
            # 关闭图形释放内存
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            return 1.0  # 返回1.0表示保存成功
            
        except Exception as e:
            logger.error(f"保存图像失败 '{metric_name}': {e}")
            import matplotlib.pyplot as plt
            plt.close(fig)
            return 0.0
    
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
