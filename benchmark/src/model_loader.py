"""
Model Loader
======================
Responsible for dynamically loading and instantiating models based on configuration.
Simplified design: directly maps config parameters to model constructors.
"""

import importlib
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Type
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """model loading error"""
    pass


class ModelLoader:
    """
    Simplified model loader

    Directly maps config parameters to model constructor, no complex factory pattern needed
    """
    
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """Initialize model loader"""
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "models"
        
        self.models_dir = Path(models_dir)
        self.model_registry = {}
        
        logger.debug(f"ModelLoader initialized, models dir: {self.models_dir}")
    
    def load_model_from_config(self, 
                              model_name: str, 
                              config: dict, 
                              input_dim: int,
                              output_dim: int = None,
                              time_steps: int = None) -> nn.Module:
        """Load model from config - directly use full config"""
        # Basic checks
        if 'models' not in config or model_name not in config['models']:
            available = list(config.get('models', {}).keys())
            raise ModelLoadError(f"Model '{model_name}' not found in config. Available: {available}")
        
        model_config = config['models'][model_name]
        model_key = model_name.lower()

        # Dynamically import model (if not registered)
        if model_key not in self.model_registry:
            # Determine class name: class field in config or default to model_name
            class_name = model_config.get('class', model_name)

            # Determine module name: module field in config or default to model_name
            module_name = model_config.get('module', model_name)

            # Support relative paths: if module_name contains path separators, use it directly; otherwise, add models. prefix
            if '/' in module_name or '.' in module_name:
                # Use relative path directly, replace / with .
                import_path = module_name.replace('/', '.')
            else:
                # Traditional approach: add models. prefix
                import_path = f"models.{module_name}"
            
            module = importlib.import_module(import_path)
            model_class = getattr(module, class_name)
            self.model_registry[model_key] = {'class': model_class}

        # Create model - filter out non-model parameters
        model_class = self.model_registry[model_key]['class']

        # Filter config parameters: exclude ModelLoader-specific metadata fields
        non_model_params = {'class', 'module'}  # Extend: filter module field
        filtered_config = {k: v for k, v in model_config.items()
                          if k not in non_model_params}

        # If output_dim is provided, add to config
        if output_dim is not None:
            filtered_config['output_dim'] = output_dim
        if time_steps is not None:
            filtered_config['time_steps'] = time_steps
        
        model = model_class(input_dim=input_dim, **filtered_config)
        
        # Set device strictly from global config
        device = (config.get('global') or {}).get('device', config.get('device', 'cpu'))
        model = model.to(device)
        logger.info(f"Model '{model_name}' moved to device: {device}")
        return model
