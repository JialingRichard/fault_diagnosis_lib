"""
Config Manager
==============

Responsible for loading and validating experiment configuration files.
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
    """Configuration validation error"""
    pass


class ConfigLoader:
    """
    Configuration loader.
    
    Provides a unified interface for:
    - Loading/saving YAML/JSON configs
    - Validation
    - (Optional) merging/overrides
    - (Optional) environment variable substitution
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the loader.
        
        Args:
            config_dir: path to config file
        """
        self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            print("Config file does not exist")
        
        # 支持的配置文件扩展名
        self.supported_extensions = {'.yaml', '.json'}
        
        logger.debug(f"ConfigLoader initialized, path: {self.config_dir}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration file.
        
        Returns:
            Dict config
        
        Raises:
            FileNotFoundError
            ConfigValidationError
        """
        config_path = Path(self.config_dir)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if config_path.suffix not in self.supported_extensions:
            raise ConfigValidationError(
                f"Unsupported config type: {config_path.suffix}, "
                f"supported: {', '.join(self.supported_extensions)}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix == '.json':
                    config = json.load(f)
                else:  # YAML
                    config = yaml.safe_load(f)
            
            logger.info(f"Config loaded: {config_path}")

            # Replace environment variables if needed
            # config = self._replace_env_variables(config)

            # Validate config
            self._validate_config(config)
            
            return config
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigValidationError(f"Invalid config format: {e}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], 
                   config_path: Union[str, Path],
                   format: str = 'yaml') -> None:
        """
        Save configuration to file.
        """
        config_path = Path(config_path)
        
        # Ensure parent exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                else:  # YAML
                    yaml.dump(config, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            
            logger.info(f"Config saved: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate basic structure and required fields.
        """
        # Required sections
        required_sections = ['datasets', 'models', 'experiments']
        
        for section in required_sections:
            if section not in config:
                raise ConfigValidationError(f"Missing required section: {section}")
        
        # Validate experiments
        if not config['experiments']:
            raise ConfigValidationError("experiments list cannot be empty")
        
        # Validate each experiment basic structure
        for exp in config['experiments']:
            required_exp_fields = ['name', 'model']
            for field in required_exp_fields:
                if field not in exp:
                    raise ConfigValidationError(f"Experiment missing field '{field}'")
            
            # Check dataset or dataset_collection
            if 'dataset' not in exp and 'dataset_collection' not in exp:
                raise ConfigValidationError(f"Experiment must contain 'dataset' or 'dataset_collection'")
        
        # Basic non-empty validation
        if not config['models']:
            raise ConfigValidationError("models cannot be empty")
        if not config['datasets']:
            raise ConfigValidationError("datasets cannot be empty")
        
        logger.info("Config validated")
    
