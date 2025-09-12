"""
时序异常检测基准测试框架
========================

一个全面、模块化的时序异常检测算法基准测试框架。

主要组件：
- ConfigManager: 配置管理器
- DataPipeline: 数据加载和预处理管道
- ModelFactory: 模型工厂和注册中心
- Trainer: 模型训练器
- TimeSeriesEvaluator: 时序专用评估器

"""

__version__ = "1.0.0"
__author__ = "Chen"
__email__ = "chen.zhenling@qq.com"

# 导入主要组件
from .config_manager import ConfigManager, load_config, save_config, create_default_config
from .dataloaders import DataPipeline, BaseDataLoader, DataMetadata
from .models.base_model import BaseModel, ModelFactory
from .trainer import Trainer, TrainingConfig
from .metrics import TimeSeriesEvaluator, MetricsCalculator, evaluate_model, print_evaluation_report

__all__ = [
    # 配置管理
    'ConfigManager',
    'load_config', 
    'save_config',
    'create_default_config',
    
    # 数据处理
    'DataPipeline',
    'BaseDataLoader',
    'DataMetadata',
    
    # 模型相关
    'BaseModel',
    'ModelFactory',
    
    # 训练相关
    'Trainer',
    'TrainingConfig',
    'BatchTrainer',
    
    # 评估相关
    'TimeSeriesEvaluator',
    'MetricsCalculator',
    'evaluate_model',
    'print_evaluation_report'
]

# 版本信息
def get_version():
    """获取框架版本"""
    return __version__

def get_info():
    """获取框架信息"""
    return {
        'name': 'Time Series Anomaly Detection Benchmark Framework',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': '时序异常检测算法基准测试框架'
    }
