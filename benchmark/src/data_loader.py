"""
Data Loaders for Time Series Anomaly Detection Benchmark
========================================================

This module provides a unified and configuration-driven data loading
and preprocessing pipeline.

"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, Tuple
from pathlib import Path
import importlib


logger = logging.getLogger(__name__)


# Metadata

class DataMetadata:
    """
    A flexible container for dataset metadata.
    Behaves like an open attribute bag.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        items = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"DataMetadata({items})"

    def __repr__(self):
        return self.__str__()

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

class DataLoader:
    """
    Unified data loader responsible for coordinating data loading and preprocessing.
    """
    def __init__(self):
        """Initialize data loader."""
        self.preprocessor_registry = {}  # Cache for loaded preprocessing functions

    def prepare_data(self, config: Dict[str, Any], dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataMetadata]:
        """
        Main entry point for data preparation.

        Args:
            config: Full configuration dictionary
            dataset_name: Dataset name

        Returns:
            (X_train, X_test, y_train, y_test, metadata)
        """
        logger.info("Starting data loading and preprocessing...")

        dataset_config = config['datasets'][dataset_name]
        preprocessing_config = dataset_config.get('preprocessing', {})

        X_train, y_train, X_test, y_test, metadata = self._load_data(dataset_name, dataset_config)
        logger.info("Data loader has provided pre-split data.")

        # Use modular preprocessor
        try:
            X_train, X_test = self._apply_preprocessing(X_train, X_test, preprocessing_config)
            logger.info("Modular preprocessing completed.")
        except Exception as e:
            logger.warning(f"Failed to apply modular preprocessing, falling back to traditional preprocessing: {e}")
            # Fall back to traditional preprocessing method
            scaler = self._get_scaler(preprocessing_config)
            if scaler:
                original_shape_train = X_train.shape
                original_shape_test = X_test.shape
                X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
                X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
                X_train = X_train.reshape(original_shape_train)
                X_test = X_test.reshape(original_shape_test)
                
                if isinstance(scaler, MinMaxScaler):
                    scaler_type = "MinMax"
                elif isinstance(scaler, StandardScaler):
                    scaler_type = "Standard"
                else:
                    scaler_type = "Unknown"

                logger.info(f"Applied '{scaler_type}' normalization.")
            else:
                print("No normalization applied.")

        logger.info("Data preparation completed.")
        return X_train, X_test, y_train, y_test, metadata

    def _apply_preprocessing(self, X_train: np.ndarray, X_test: np.ndarray, 
                           preprocessing_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply modular preprocessing steps
        
        Args:
            X_train, X_test
            preprocessing_config: Preprocessing configuration

        Returns:
            (X_train_processed, X_test_processed)
        """
        if not preprocessing_config:
            logger.info("No preprocessing steps configured, returning original data")
            return X_train, X_test

        # Record original data shapes
        logger.info(f"Starting preprocessing | Original data: train{X_train.shape}, test{X_test.shape}")

        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

        # Apply each preprocessing step in order
        for step_name, step_config in preprocessing_config.items():
            if step_name == 'steps':
                # New step list format
                for step in step_config:
                    step_name = step['name']
                    step_params = step.get('params', {})
                    file_name = step.get('file', None)
                    function_name = step.get('function', step_name)

                    # Record number of features before processing
                    features_before = X_train_processed.shape[-1]
                    
                    preprocessor_func = self._load_preprocessor(function_name, file_name)
                    if preprocessor_func:
                        X_train_processed, X_test_processed = preprocessor_func(
                            X_train_processed, X_test_processed, **step_params)

                        # Record number of features after processing
                        features_after = X_train_processed.shape[-1]

                        # Construct detailed log information
                        param_str = ", ".join([f"{k}={v}" for k, v in step_params.items()]) if step_params else "No parameters"
                        feature_change = f"{features_before}→{features_after}" if features_before != features_after else f"{features_before}"

                        logger.info(f"✓ {step_name}: {file_name}.{function_name}({param_str}) | Feature count: {feature_change}")
            else:
                # Simplified format support
                if self._is_simple_config(step_name, step_config):
                    # Record number of features before processing
                    features_before = X_train_processed.shape[-1]
                    
                    preprocessor_func = self._load_preprocessor_simple(step_name, step_config)
                    if preprocessor_func:
                        X_train_processed, X_test_processed = preprocessor_func(
                            X_train_processed, X_test_processed)

                        # Record number of features after processing
                        features_after = X_train_processed.shape[-1]
                        feature_change = f"{features_before}→{features_after}" if features_before != features_after else f"{features_before}"

                        logger.info(f"✓ {step_name}: {step_config} | Feature count: {feature_change}")

        # Record final data shapes
        logger.info(f"Preprocessing completed | Final data: train{X_train_processed.shape}, test{X_test_processed.shape}")

        return X_train_processed, X_test_processed
    
    def _is_simple_config(self, step_name: str, step_config: Any) -> bool:
        """Check if the configuration is in simplified format"""
        simple_configs = {
            'normalize': [True, False, 'minmax', 'standard', 'robust'],
            'add_noise': [True, False],
            'remove_outliers': [True, False],
            'smooth': [True, False]
        }
        return step_name in simple_configs and step_config in simple_configs[step_name]
    
    def _load_preprocessor_simple(self, step_name: str, step_config: Any):
        """Load preprocessor for simplified configuration"""
        if step_name == 'normalize':
            if step_config is True or step_config == 'standard':
                return self._load_preprocessor('standard_normalize', 'normalizers')
            elif step_config == 'minmax':
                return self._load_preprocessor('minmax_normalize', 'normalizers')
            elif step_config == 'robust':
                return self._load_preprocessor('robust_normalize', 'normalizers')
            elif step_config is False:
                return self._load_preprocessor('no_normalize', 'normalizers')
        elif step_name == 'add_noise' and step_config is True:
            return self._load_preprocessor('add_gaussian_noise', 'noise_processors')
        elif step_name == 'remove_outliers' and step_config is True:
            return self._load_preprocessor('remove_outliers', 'noise_processors')
        elif step_name == 'smooth' and step_config is True:
            return self._load_preprocessor('smooth_data', 'noise_processors')
        
        return None
    
    def _load_preprocessor(self, function_name: str, module_name: str = None):
        """Dynamically load preprocessing function"""
        if module_name is None:
            # Try to find in common modules
            common_modules = ['normalizers', 'noise_processors', 'feature_engineering']
            for mod in common_modules:
                try:
                    func = self._load_from_module(mod, function_name)
                    if func:
                        return func
                except:
                    continue
            logger.error(f"Cannot find preprocessing function: {function_name}")
            return None
        
        return self._load_from_module(module_name, function_name)
    
    def _load_from_module(self, module_name: str, function_name: str):
        """Load function from specified module"""
        cache_key = f"{module_name}.{function_name}"

        # Check cache
        if cache_key in self.preprocessor_registry:
            return self.preprocessor_registry[cache_key]
        
        try:
            # Dynamically import module
            module = importlib.import_module(f"preprocessors.{module_name}")

            # Get function
            if hasattr(module, function_name):
                preprocessor_func = getattr(module, function_name)
                self.preprocessor_registry[cache_key] = preprocessor_func
                logger.debug(f"Preprocessing function '{function_name}' loaded successfully (module: {module_name})")
                return preprocessor_func
            else:
                logger.error(f"Cannot find function '{function_name}' in module '{module_name}'")
                return None
                
        except ImportError:
            logger.error(f"Cannot import preprocessing module: preprocessors.{module_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to load preprocessing function: {module_name}.{function_name}, error: {str(e)}")
            return None

    def _get_scaler(self, preprocessing_config: Dict[str, Any]):
        """Get a scaler instance based on the configuration."""
        # Support two configuration formats:
        # 1. normalize: true/false (simplified format)
        # 2. normalization: 'minmax'/'standard'/'none' (detailed format)

        # Check simplified format first
        if 'normalize' in preprocessing_config:
            normalize = preprocessing_config.get('normalize', False)
            if normalize is True:
                # Default to standard normalization
                return StandardScaler()
            elif normalize is False:
                return None
            elif isinstance(normalize, str):
                # If normalize is a string, treat it as normalization type
                normalization = normalize.lower()
            else:
                return None
        else:
            # Check detailed format
            normalization = preprocessing_config.get('normalization', 'none').lower()

        # Handle normalization types
        if normalization == 'minmax':
            return MinMaxScaler()
        elif normalization in ['standard', 'z-score']:
            return StandardScaler()
        elif normalization in ['none', 'false']:
            return None
        else:
            logger.warning(f"Unknown normalization type: {normalization}, skipping normalization")
            return None
    
    def _load_file(self, filename: str) -> np.ndarray:
        project_root = Path(__file__).parent.parent
        file_path = project_root / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file does not exist: {file_path}")
        
        if file_path.suffix == '.npy':
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_data(self, dataset_name: str, dataset_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataMetadata]:
        """Load pre-split data files."""
        logger.info("Loading pre-split dataset...")

        # Read file names from config
        train_X = self._load_file(dataset_config['train_data'])
        train_y = self._load_file(dataset_config['train_label'])
        test_X = self._load_file(dataset_config['test_data'])
        test_y = self._load_file(dataset_config['test_label'])

        # Create metadata
        metadata = DataMetadata(
            dataset_name=dataset_name,
            fault_type='binary' if len(np.unique(train_y)) <= 2 else 'multi-class',
            feature_dim=train_X.shape[2],
            num_classes=len(np.unique(np.concatenate((train_y.flatten(), test_y.flatten())))),
            timesteps=train_X.shape[1],
            number_train_samples=train_X.shape[0],
            number_test_samples=test_X.shape[0],
        )
        
        return train_X, train_y, test_X, test_y, metadata