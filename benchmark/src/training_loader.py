"""
Training Loader (TrainingLoader)
===============================

Responsible for dynamically loading training strategies according to the config.

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
    """Training loading error"""
    pass


class TrainingLoader:
    """
    Dynamically choose training strategy and create trainer based on config.
    """
    
    def __init__(self, trainers_dir: Path = None):
        """Initialize training loader"""
        if trainers_dir is None:
            trainers_dir = Path(__file__).parent.parent / "trainers"
        
        self.trainers_dir = Path(trainers_dir)
        self.trainer_registry = {}
        
        logger.debug(f"TrainingLoader initialized, trainers dir: {self.trainers_dir}")
    
    def create_trainer(self, config: Dict[str, Any], training_template_name: str, 
                      model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray):
        """
        Create trainer instance.
        
        Args:
            config: Full config dict
            training_template_name: Training template name
            model: Model to train
            X_train, y_train: Training data
            X_test, y_test: Test data (for validation)
        """
        # get training config
        if 'training_templates' not in config:
            raise TrainingLoadError("Missing 'training_templates' in config")
        
        if training_template_name not in config['training_templates']:
            available = list(config['training_templates'].keys())
            raise TrainingLoadError(f"Training template '{training_template_name}' not found. Available: {available}")
        
        training_config = config['training_templates'][training_template_name]
        training_type = training_config.get('type', 'supervised')
        
        # Load trainer class by training type
        trainer_class = self._load_trainer_class(training_type, training_config)
        
        # Create trainer instance
        # get device from global config
        device = (config.get('global') or {}).get('device', config.get('device', 'cpu'))
        trainer = trainer_class(
            model=model,
            config=training_config,
            device=device,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            full_config=config  # pass full config
        )
        
        logger.debug(f"Trainer created, type: {training_type}")
        return trainer
    
    def _load_trainer_class(self, training_type: str, training_config: Dict[str, Any]):
        """Dynamically load trainer class"""
        trainer_key = training_type.lower()
        
        if trainer_key not in self.trainer_registry:
            try:
                # Resolve module name: config 'trainer_file' or default based on type
                module_name = training_config.get('trainer_file', f"{training_type}_trainer")
                
                # Resolve class name: config 'trainer_class' or default '<Type>Trainer'
                class_name = training_config.get('trainer_class', f"{training_type.title()}Trainer")
                
                # Dynamic import
                module = importlib.import_module(f"trainers.{module_name}")
                trainer_class = getattr(module, class_name)
                self.trainer_registry[trainer_key] = trainer_class
                logger.debug(f"Trainer '{training_type}' loaded (module: {module_name}, class: {class_name})")
            except ImportError:
                raise TrainingLoadError(f"Cannot import trainer module: trainers.{module_name}")
            except AttributeError:
                raise TrainingLoadError(f"Trainer module trainers.{module_name} has no class '{class_name}'")
        
        return self.trainer_registry[trainer_key]
