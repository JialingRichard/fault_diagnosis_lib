"""
配置管理器 (ConfigManager)
==========================

负责实验配置的加载

"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from copy import deepcopy
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """配置验证异常"""
    pass


class ConfigLoader:
    """
    配置加载器
    
    提供统一的配置加载接口，支持：
    - YAML配置文件加载和保存
    - 配置验证
    - 配置合并和继承
    - 环境变量替换
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为当前目录的configs子目录
        """
        self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            print(f"配置文件不存在")
        
        # 支持的配置文件扩展名
        self.supported_extensions = {'.yaml', '.json'}
        
        logger.info(f"配置加载器初始化完成，配置目录: {self.config_dir}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            ConfigValidationError: 配置格式错误
        """
        config_path = Path(self.config_dir)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        if config_path.suffix not in self.supported_extensions:
            raise ConfigValidationError(
                f"不支持的配置文件格式: {config_path.suffix}, "
                f"支持的格式: {', '.join(self.supported_extensions)}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix == '.json':
                    config = json.load(f)
                else:  # YAML
                    config = yaml.safe_load(f)
            
            logger.info(f"配置文件加载成功: {config_path}")
            
            # 环境变量替换
            # config = self._replace_env_variables(config)
            
            # 验证配置
            self._validate_config(config)
            
            return config
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigValidationError(f"配置文件格式错误: {e}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], 
                   config_path: Union[str, Path],
                   format: str = 'yaml') -> None:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            config_path: 保存路径
            format: 文件格式，'yaml' 或 'json'
        """
        config_path = Path(config_path)
        
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:  # YAML
                    yaml.dump(config, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            
            logger.info(f"配置文件保存成功: {config_path}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置文件的基本结构和必需字段
        
        Args:
            config: 配置字典
            
        Raises:
            ConfigValidationError: 配置验证失败
        """
        # 新配置结构的必需部分
        required_sections = ['datasets', 'models', 'experiments']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(f"缺少必需配置节: {section}")
        
        # 验证实验配置
        if not config['experiments']:
            raise ConfigValidationError("experiments 列表不能为空")
        
        # 验证每个实验的基本结构
        for exp in config['experiments']:
            required_exp_fields = ['name', 'model', 'dataset']
            for field in required_exp_fields:
                if field not in exp:
                    raise ConfigValidationError(f"实验配置缺少 '{field}' 字段")
        
        # 基本验证：确保非空
        if not config['models']:
            raise ConfigValidationError("models 不能为空")
        if not config['datasets']:
            raise ConfigValidationError("datasets 不能为空")
        
        logger.info("配置验证通过")
    
