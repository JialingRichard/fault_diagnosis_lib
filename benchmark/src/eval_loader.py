"""
Evaluation Loader (EvalLoader)
==============================

Responsible for dynamically loading and executing evaluator functions.

"""

import importlib
import numpy as np
from typing import Dict, Any, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EvalLoadError(Exception):
    """Evaluation loading error"""
    pass


class EvalLoader:
    """
    Evaluation loader that dynamically loads and runs evaluators.
    """
    
    def __init__(self, evaluators_dir: Path = None):
        """Initialize the evaluation loader"""
        if evaluators_dir is None:
            evaluators_dir = Path(__file__).parent.parent / "evaluators"
        
        self.evaluators_dir = Path(evaluators_dir)
        self.evaluator_registry = {}
        
        # Evaluation context
        self.plots_dir = None
        self.epoch_info = None  # {'epoch': 10} means during training; None means final evaluation
        
        logger.debug(f"EvalLoader initialized, evaluators dir: {self.evaluators_dir}")
    
    def set_context(self, plots_dir=None, epoch_info=None):
        """Set evaluation context.
        
        Args:
            plots_dir: Directory to save figures
            epoch_info: Epoch info dict like {'epoch': 10}; None for final evaluation
        """
        self.plots_dir = Path(plots_dir) if plots_dir else None
        self.epoch_info = epoch_info
    
    def evaluate(self, config: Dict[str, Any], eval_template_name: str,
                X_train: np.ndarray, y_train: np.ndarray, y_train_pred: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray, y_test_pred: np.ndarray) -> Dict[str, float]:
        """Run evaluation for a template.
        
        Args:
            config: Full configuration
            eval_template_name: Evaluation template name
            X_train, y_train, y_train_pred: Training data/labels/predictions
            X_test, y_test, y_test_pred: Test(or selected split) data/labels/predictions
        
        Returns:
            Dict of results {'metric_name': score}
        """
        # get evaluation config
        if 'evaluation_templates' not in config:
            raise EvalLoadError("Missing 'evaluation_templates' in config")
        
        if eval_template_name not in config['evaluation_templates']:
            available = list(config['evaluation_templates'].keys())
            raise EvalLoadError(f"Evaluation template '{eval_template_name}' not found. Available: {available}")
        
        eval_config = config['evaluation_templates'][eval_template_name]
        # Flattened template support: if no 'metrics', treat non '_' keys as metrics config
        if isinstance(eval_config, dict) and 'metrics' not in eval_config:
            metrics = {k: v for k, v in eval_config.items() if not str(k).startswith('_')}
        else:
            metrics = eval_config.get('metrics', {})
        
        if not metrics:
            raise EvalLoadError(f"No metrics defined in evaluation template '{eval_template_name}'")

        # Execute all evaluation metrics
        results = {}
        for metric_name, metric_config in metrics.items():
            evaluator_func = self._load_evaluator(metric_name, metric_config)
            try:
                result = evaluator_func(X_train, y_train, y_train_pred, 
                                      X_test, y_test, y_test_pred)
                
                # Post-process different return types
                processed_result = self._process_evaluator_result(
                    result, metric_name, metric_config
                )
                results[metric_name] = processed_result
                
                # Log by result type
                if isinstance(processed_result, (int, float)):
                    logger.info(f"Metric '{metric_name}': {processed_result:.4f}")
                elif isinstance(processed_result, str):
                    logger.info(f"Metric '{metric_name}': {processed_result}")
                else:
                    logger.info(f"Metric '{metric_name}': processed")
                
            except Exception as e:
                logger.error(f"Metric '{metric_name}' failed: {e}")
                results[metric_name] = None
        
        return results
    
    def _process_evaluator_result(self, result, metric_name: str, metric_config: Dict[str, Any]):
        """Process evaluator return value into a numeric/string form."""
        import matplotlib.figure
        
        # Numeric
        if isinstance(result, (int, float, np.number)):
            return float(result)
        
        # String
        if isinstance(result, str):
            return result
        
        # Figure: save and return numeric (e.g., count)
        if isinstance(result, matplotlib.figure.Figure):
            return self._save_figure(result, metric_name, metric_config)
        
        # Tuple: (figure, metric_value)
        if isinstance(result, tuple) and len(result) == 2:
            fig, value = result
            if isinstance(fig, matplotlib.figure.Figure):
                self._save_figure(fig, metric_name, metric_config)
                return float(value) if isinstance(value, (int, float, np.number)) else value
        
        # Others: try casting to float
        try:
            return float(result)
        except:
            logger.warning(f"Cannot process return type from evaluator '{metric_name}': {type(result)}")
            return None
    
    def _save_figure(self, fig, metric_name: str, metric_config: Dict[str, Any]) -> float:
        """Save a matplotlib figure to the proper folder and return 1.0 on success."""
        if self.plots_dir is None:
            logger.warning(f"plots_dir not set; cannot save figure for metric: {metric_name}")
            import matplotlib.pyplot as plt
            plt.close(fig)
            return 0.0
        
        try:
            import os

            # Determine save path
            if self.epoch_info is not None:
                # During training: save to 'epochinfo' subdir
                epoch_num = self.epoch_info.get('epoch', 0)
                epochinfo_dir = self.plots_dir / 'epochinfo'
                epochinfo_dir.mkdir(parents=True, exist_ok=True)
                
                # filename: metric_name_epoch_XXX.png
                filename = f"{metric_name}_epoch_{epoch_num+1:03d}.png"
                save_path = epochinfo_dir / filename
            else:
                # Final evaluation: save to main plots dir
                filename = f"{metric_name}.png"
                save_path = self.plots_dir / filename

            # Save figure
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Log save info
            logger.info(f"Figure saved: {save_path}")

            # Close figure to release memory
            import matplotlib.pyplot as plt
            plt.close(fig)

            return 1.0  # Return 1.0 to indicate success

        except Exception as e:
            logger.error(f"Failed to save figure for '{metric_name}': {e}")
            import matplotlib.pyplot as plt
            plt.close(fig)
            return 0.0
    
    def _load_evaluator(self, metric_name: str, metric_config: Dict[str, Any]) -> Callable:
        """Dynamically load evaluator function by name."""
        evaluator_key = metric_name.lower()
        
        if evaluator_key not in self.evaluator_registry:
            try:
                # Resolve module name: config 'file' or fall back to metric_name
                module_name = metric_config.get('file', metric_name) if metric_config else metric_name
                
                # Resolve function: config 'function' or default 'evaluate'
                function_name = metric_config.get('function', 'evaluate') if metric_config else 'evaluate'
                
                # Dynamic import
                module = importlib.import_module(f"evaluators.{module_name}")
                evaluator_func = getattr(module, function_name)
                self.evaluator_registry[evaluator_key] = evaluator_func
                logger.info(f"Evaluator '{metric_name}' loaded (module: {module_name}, function: {function_name})")
            except ImportError:
                raise EvalLoadError(f"Cannot import evaluator module: evaluators.{module_name}")
            except AttributeError:
                raise EvalLoadError(f"Evaluator module evaluators.{module_name} has no function '{function_name}'")
        
        return self.evaluator_registry[evaluator_key]
