"""
配置管理器 (ConfigManager)
==========================

负责实验配置的加载、验证、保存和管理，支持：
- YAML配置文件的读写
- 配置模板和继承
- 参数验证和类型检查
- 配置版本管理

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


class ConfigManager:
    """
    配置管理器
    
    提供统一的配置管理接口，支持：
    - YAML配置文件加载和保存
    - 配置验证
    - 配置合并和继承
    - 环境变量替换
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为当前目录的configs子目录
        """
        self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            print(f"配置文件不存在")
        
        # 支持的配置文件扩展名
        self.supported_extensions = {'.yaml', '.json'}
        
        logger.info(f"配置管理器初始化完成，配置目录: {self.config_dir}")
    
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
        required_sections = ['experiment', 'data', 'models', 'evaluation', 'output']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(f"缺少必需配置节: {section}")
        
        # 验证实验配置
        if 'name' not in config['experiment']:
            raise ConfigValidationError("实验配置缺少 'name' 字段")
        
        # 验证数据配置
        if 'dataset' not in config['data']:
            raise ConfigValidationError("数据配置缺少 'dataset' 字段")
        
        # 验证模型配置
        if not config['models']:
            raise ConfigValidationError("模型配置不能为空")
        
        # 验证评估配置
        if 'metrics' not in config['evaluation']:
            raise ConfigValidationError("评估配置缺少 'metrics' 字段")
        
        logger.info("配置验证通过")
    
    def get_config_info(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取配置文件信息
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置信息字典
        """
        config = self.load_config(config_path)
        
        info = {
            'file_path': str(config_path),
            'experiment_name': config.get('experiment', {}).get('name', 'Unknown'),
            'description': config.get('experiment', {}).get('description', ''),
            'dataset': config.get('data', {}).get('dataset', 'Unknown'),
            'models': list(config.get('models', {}).keys()),
            'metrics': config.get('evaluation', {}).get('metrics', []),
            'file_size': Path(config_path).stat().st_size,
            'modified_time': datetime.fromtimestamp(
                Path(config_path).stat().st_mtime
            ).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return info


    # # 便捷函数
    # def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    #     """
    #     便捷的配置加载函数
        
    #     Args:
    #         config_path: 配置文件路径
            
    #     Returns:
    #         配置字典
    #     """
    #     manager = ConfigManager()
    #     return manager.load_config(config_path)


    # def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    #     """
    #     便捷的配置保存函数
        
    #     Args:
    #         config: 配置字典
    #         config_path: 保存路径
    #     """
    #     manager = ConfigManager()
    #     manager.save_config(config, config_path)

    # def create_config_template(self, template_name: str = "default") -> Dict[str, Any]:
    #     """
    #     创建配置模板
        
    #     Args:
    #         template_name: 模板名称
            
    #     Returns:
    #         配置模板字典
    #     """
        
    #     if template_name == "default":
    #         return self._create_default_template()
    #     elif template_name == "minimal":
    #         return self._create_minimal_template()
    #     elif template_name == "full":
    #         return self._create_full_template()
    #     else:
    #         raise ValueError(f"不支持的模板类型: {template_name}")
    
    # def _create_default_template(self) -> Dict[str, Any]:
    #     """创建默认配置模板"""
    #     return {
    #         'experiment': {
    #             'name': 'default_experiment',
    #             'description': '默认实验配置',
    #             'version': '1.0',
    #             'author': 'Benchmark Team',
    #             'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #         },
    #         'data': {
    #             'dataset': 'swat',
    #             'train_file': '../data/raw/train.csv',
    #             'test_file': '../data/raw/test.csv',
    #             'preprocessing': {
    #                 'normalize': True,
    #                 'fill_missing': True,
    #                 'remove_outliers': False
    #             },
    #             'window_size': 10,
    #             'stride': 1
    #         },
    #         'models': {
    #             'iforest': {
    #                 'n_estimators': 100,
    #                 'contamination': 0.1,
    #                 'random_state': 42
    #             }
    #         },
    #         'evaluation': {
    #             'metrics': ['f1', 'precision', 'recall', 'auc'],
    #             'tolerance': 0,
    #             'auto_threshold': True
    #         },
    #         'output': {
    #             'save_results': True,
    #             'save_models': False,
    #             'generate_plots': True,
    #             'results_dir': '../results'
    #         }
    #     }
    
    # def _create_minimal_template(self) -> Dict[str, Any]:
    #     """创建最小配置模板"""
    #     return {
    #         'experiment': {
    #             'name': 'minimal_experiment'
    #         },
    #         'data': {
    #             'dataset': 'synthetic'
    #         },
    #         'models': {
    #             'iforest': {}
    #         },
    #         'evaluation': {
    #             'metrics': ['f1']
    #         },
    #         'output': {
    #             'save_results': False
    #         }
    #     }
    
    # def _create_full_template(self) -> Dict[str, Any]:
    #     """创建完整配置模板"""
    #     template = self._create_default_template()
        
    #     # 扩展模型配置
    #     template['models'].update({
    #         'lstm_ae': {
    #             'hidden_dim': 64,
    #             'num_layers': 2,
    #             'dropout': 0.2,
    #             'lr': 0.001,
    #             'batch_size': 32,
    #             'epochs': 50,
    #             'patience': 10
    #         },
    #         'lof': {
    #             'n_neighbors': 20,
    #             'contamination': 0.1
    #         }
    #     })
        
    #     # 扩展评估配置
    #     template['evaluation'].update({
    #         'cross_validation': {
    #             'enabled': False,
    #             'folds': 5,
    #             'method': 'time_series'
    #         }
    #     })
        
    #     # 扩展输出配置
    #     template['output'].update({
    #         'save_predictions': False,
    #         'generate_report': True,
    #         'log_level': 'INFO'
    #     })
        
    #     return template
    

    
    # def _replace_env_variables(self, config: Any) -> Any:
    #     """
    #     递归替换配置中的环境变量
    #     格式: ${ENV_VAR_NAME} 或 ${ENV_VAR_NAME:default_value}
        
    #     Args:
    #         config: 配置值（可能是字典、列表、字符串等）
            
    #     Returns:
    #         替换环境变量后的配置值
    #     """
    #     if isinstance(config, dict):
    #         return {k: self._replace_env_variables(v) for k, v in config.items()}
    #     elif isinstance(config, list):
    #         return [self._replace_env_variables(item) for item in config]
    #     elif isinstance(config, str):
    #         return self._replace_string_env_variables(config)
    #     else:
    #         return config
    
    # def _replace_string_env_variables(self, text: str) -> str:
    #     """
    #     替换字符串中的环境变量
        
    #     Args:
    #         text: 包含环境变量的字符串
            
    #     Returns:
    #         替换后的字符串
    #     """
    #     import re
        
    #     # 匹配 ${VAR_NAME} 或 ${VAR_NAME:default_value} 格式
    #     pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
    #     def replace_match(match):
    #         var_name = match.group(1)
    #         default_value = match.group(2) if match.group(2) is not None else ''
    #         return os.getenv(var_name, default_value)
        
    #     return re.sub(pattern, replace_match, text)
    
    # def list_configs(self, pattern: str = "*.yaml") -> List[Path]:
    #     """
    #     列出配置目录中的所有配置文件
        
    #     Args:
    #         pattern: 文件名模式
            
    #     Returns:
    #         配置文件路径列表
    #     """
    #     config_files = list(self.config_dir.glob(pattern))
    #     config_files.extend(self.config_dir.glob("*.yml"))
    #     config_files.extend(self.config_dir.glob("*.json"))
        
    #     return sorted(set(config_files))
    
    # def validate_config_file(self, config_path: Union[str, Path]) -> bool:
    #     """
    #     验证配置文件是否有效
        
    #     Args:
    #         config_path: 配置文件路径
            
    #     Returns:
    #         是否有效
    #     """
    #     try:
    #         self.load_config(config_path)
    #         return True
    #     except Exception as e:
    #         logger.error(f"配置文件验证失败: {e}")
    #         return False
    



# def create_default_config() -> Dict[str, Any]:
#     """
#     创建默认配置的便捷函数
    
#     Returns:
#         默认配置字典
#     """
#     manager = ConfigManager()
#     return manager.create_config_template('default')


# # 配置工具类
# class ConfigUtils:
#     """配置工具类，提供额外的配置操作功能"""
    
#     @staticmethod
#     def flatten_config(config: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
#         """
#         将嵌套配置展平为单层字典
        
#         Args:
#             config: 嵌套配置字典
#             separator: 键分隔符
            
#         Returns:
#             展平后的字典
#         """
#         def _flatten(obj, prefix=''):
#             if isinstance(obj, dict):
#                 for key, value in obj.items():
#                     new_key = f"{prefix}{separator}{key}" if prefix else key
#                     yield from _flatten(value, new_key)
#             else:
#                 yield prefix, obj
        
#         return dict(_flatten(config))
    
    # @staticmethod
    # def unflatten_config(flat_config: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    #     """
    #     将展平的配置还原为嵌套字典
        
    #     Args:
    #         flat_config: 展平的配置字典
    #         separator: 键分隔符
            
    #     Returns:
    #         嵌套配置字典
    #     """
    #     result = {}
        
    #     for key, value in flat_config.items():
    #         keys = key.split(separator)
    #         target = result
            
    #         for k in keys[:-1]:
    #             if k not in target:
    #                 target[k] = {}
    #             target = target[k]
            
    #         target[keys[-1]] = value
        
    #     return result
    
    # @staticmethod
    # def diff_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     比较两个配置的差异
        
    #     Args:
    #         config1: 配置1
    #         config2: 配置2
            
    #     Returns:
    #         差异字典
    #     """
    #     flat1 = ConfigUtils.flatten_config(config1)
    #     flat2 = ConfigUtils.flatten_config(config2)
        
    #     all_keys = set(flat1.keys()) | set(flat2.keys())
        
    #     diff = {}
    #     for key in all_keys:
    #         val1 = flat1.get(key, '<MISSING>')
    #         val2 = flat2.get(key, '<MISSING>')
            
    #         if val1 != val2:
    #             diff[key] = {
    #                 'config1': val1,
    #                 'config2': val2
    #             }
        
    #     return diff
