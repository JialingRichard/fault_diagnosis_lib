"""
模型加载器 (ModelLoader)
======================

负责根据配置动态加载和实例化模型
采用简化设计：直接将配置参数映射到模型构造函数

"""

import importlib
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Type
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """模型加载异常"""
    pass


class ModelLoader:
    """
    简化的模型加载器
    
    直接将配置参数映射到模型构造函数，无需复杂的工厂模式
    """
    
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """初始化模型加载器"""
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "models"
        
        self.models_dir = Path(models_dir)
        self.model_registry = {}
        
        logger.debug(f"模型加载器初始化完成，模型目录: {self.models_dir}")
    
    def load_model_from_config(self, 
                              model_name: str,
                              config: Dict[str, Any],
                              input_dim: int) -> nn.Module:
        """根据配置加载模型 - 直接使用完整配置"""
        # 基本检查
        if 'models' not in config or model_name not in config['models']:
            available = list(config.get('models', {}).keys())
            raise ModelLoadError(f"配置中未找到模型 '{model_name}'，可用: {available}")
        
        model_config = config['models'][model_name]
        model_key = model_name.lower()
        
        # 动态导入模型（如果未注册）
        if model_key not in self.model_registry:
            # 确定类名：配置中的class字段 或 默认使用model_name
            class_name = model_config.get('class', model_name)
            
            # 确定模块名：配置中的module字段 或 默认使用model_name
            module_name = model_config.get('module', model_name)
            
            module = importlib.import_module(f"models.{module_name}")
            model_class = getattr(module, class_name)
            self.model_registry[model_key] = {'class': model_class}
        
        # 创建模型 - 过滤掉非模型参数
        model_class = self.model_registry[model_key]['class']
        
        # 过滤配置参数：排除ModelLoader专用的元数据字段
        non_model_params = {'class'}  # 可以扩展：'module', 'pretrained_path' 等
        filtered_config = {k: v for k, v in model_config.items() 
                          if k not in non_model_params}
        
        model = model_class(input_dim=input_dim, **filtered_config)
        
        # 设置设备 - 严格按配置执行
        device = config.get('device')
        model = model.to(device)
        logger.info(f"模型 '{model_name}' 加载到设备: {device}")
        return model

