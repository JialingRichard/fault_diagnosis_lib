"""
Epoch Info Loader

Dynamically loads and executes epoch info print functions.
"""

import importlib.util
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Callable
import inspect

logger = logging.getLogger(__name__)


class EpochInfoLoader:
    """
    epoch info loader that dynamically loads and executes epoch info print functions.
    """
    
    def __init__(self, epochinfo_dir: Path = None, eval_loader=None):
        """Initialize epoch info loader"""
        if epochinfo_dir is None:
            epochinfo_dir = Path(__file__).parent.parent / "epochinfos"
        
        self.epochinfo_dir = Path(epochinfo_dir)
        self.epochinfo_registry = {}
        self.eval_loader = eval_loader

        logger.debug(f"Epoch info loader initialized, directory: {self.epochinfo_dir}")

    def print_epoch_info(self, config: Dict[str, Any], epochinfo_template_name: str,
                        epoch_data: Dict[str, Any]) -> None:
        """
        Print epoch info
        
        Args:
            config: Full configuration
            epochinfo_template_name: Epoch info template name
            epoch_data: Epoch data, containing all necessary information
        """
        # New logic: epochinfo directly references the template name from evaluation_templates
        eval_templates = config.get('evaluation_templates', {})
        if epochinfo_template_name and epochinfo_template_name in eval_templates:
            # Basic info + evaluation metrics
            self._print_with_evaluation(epoch_data, epochinfo_template_name, config)
        else:
            # Not specified or template does not exist, print basic info
            if epochinfo_template_name:
                logger.warning(f"Epoch info template '{epochinfo_template_name}' not found, using default print")
            self._default_print(epoch_data)
    
    def _load_print_function(self, format_config: Dict[str, str]) -> Callable:
        """
        Load print function
        
        Args:
            format_config: Format configuration, containing file and function fields

        Returns:
            Print function or None
        """
        if isinstance(format_config, str):
            # Simple string, use default function name
            file_name = format_config
            function_name = 'print_epoch_info'
        else:
            # Dictionary format, specify file and function
            file_name = format_config.get('file', 'default')
            function_name = format_config.get('function', 'print_epoch_info')

        # Construct cache key
        cache_key = f"{file_name}.{function_name}"

        # Check cache
        if cache_key in self.epochinfo_registry:
            return self.epochinfo_registry[cache_key]
        
        try:
            # Construct file path
            file_path = self.epochinfo_dir / f"{file_name}.py"
            
            if not file_path.exists():
                logger.error(f"Epoch info file does not exist: {file_path}")
                return None

            # Dynamically load module
            spec = importlib.util.spec_from_file_location(file_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get function
            if hasattr(module, function_name):
                print_func = getattr(module, function_name)

                # Validate function signature
                sig = inspect.signature(print_func)
                expected_params = ['epoch_data', 'config']
                actual_params = list(sig.parameters.keys())

                if len(actual_params) >= 1:  # At least need epoch_data parameter
                    self.epochinfo_registry[cache_key] = print_func
                    logger.debug(f"Epoch info function '{function_name}' loaded successfully (module: {file_name})")
                    return print_func
                else:
                    logger.error(f"Function '{function_name}' parameter does not meet requirements, at least 1 parameter needed: epoch_data")
                    return None
            else:
                logger.error(f"Module '{file_name}' does not contain function '{function_name}'")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load epoch info function: {file_name}.{function_name}, error: {str(e)}")
            return None
    
    def _default_print(self, epoch_data: Dict[str, Any]) -> None:
        """
        Default epoch info print
        
        Args:
            epoch_data: epoch data
        """
        epoch = epoch_data.get('epoch', 0)
        total_epochs = epoch_data.get('total_epochs', 0)
        train_loss = epoch_data.get('train_loss', 0.0)
        val_loss = epoch_data.get('val_loss', 0.0)
        improvement = epoch_data.get('improvement', None)
        patience_counter = epoch_data.get('patience_counter', 0)
        patience = epoch_data.get('patience', 0)
        
        if improvement is not None and improvement > 0:
            print(f"   E{epoch+1:3d}/{total_epochs}: {train_loss:.4f}→{val_loss:.4f} ↓{improvement:.4f}")
        else:
            print(f"   E{epoch+1:3d}/{total_epochs}: {train_loss:.4f}->{val_loss:.4f} x{patience_counter}/{patience}")
    
    def _print_with_evaluation(self, epoch_data: Dict[str, Any], eval_template_name: str, config: Dict[str, Any]) -> None:
        """
        Print basic epoch info + evaluation metrics
        
        Args:
            epoch_data: epoch data
            eval_template_name: epochinfo template config
            config: complete config
        """
        # Get basic info string
        epoch = epoch_data.get('epoch', 0)
        total_epochs = epoch_data.get('total_epochs', 0)
        train_loss = epoch_data.get('train_loss', 0.0)
        val_loss = epoch_data.get('val_loss', 0.0)
        improvement = epoch_data.get('improvement', None)
        patience_counter = epoch_data.get('patience_counter', 0)
        patience = epoch_data.get('patience', 0)
        
        if improvement is not None and improvement > 0:
            basic_info = f"   E{epoch+1:3d}/{total_epochs}: {train_loss:.4f}→{val_loss:.4f} ↓{improvement:.4f}"
        else:
            basic_info = f"   E{epoch+1:3d}/{total_epochs}: {train_loss:.4f}->{val_loss:.4f} x{patience_counter}/{patience}"

        # If eval_loader exists and evaluation is configured, calculate evaluation metrics
        metrics_str = ""
        if self.eval_loader and eval_template_name:
            trainer = epoch_data.get('trainer')
            if trainer:
                try:
                    # Split: avoid information leakage by using validation set during training
                    epochinfo_split = str(getattr(trainer, 'config', {}).get('epochinfo_split', 'val')).lower()
                    if epochinfo_split not in {'val', 'test'}:
                        epochinfo_split = 'val'

                    # Get/cache required predictions
                    # Decide whether any metric requires train predictions
                    need_train_pred = self._template_needs_train_pred(eval_template_name, config)
                    train_pred = epoch_data.get('train_pred')
                    if need_train_pred and train_pred is None:
                        # compute silently to avoid extra progress noise
                        train_pred = trainer.predict(trainer.X_train, show_progress=False)
                    elif not need_train_pred:
                        # provide placeholder to evaluators that ignore train channel
                        train_pred = np.empty((0,))

                    if epochinfo_split == 'val':
                        # Select validation set as evaluation "test channel"
                        sel_X = trainer.X_val
                        sel_y = trainer.y_val
                        sel_pred = getattr(trainer, '_cached_val_pred', None)
                        if sel_pred is None:
                            sel_pred = trainer.predict(trainer.X_val)
                    else:  # 'test'
                        sel_X = trainer.X_test
                        sel_y = trainer.y_test
                        sel_pred = epoch_data.get('test_pred') or getattr(trainer, '_cached_test_pred', None)
                        if sel_pred is None:
                            sel_pred = trainer.predict(trainer.X_test)

                    # Prepare numpy labels
                    X_train = trainer.X_train
                    y_train = trainer.y_train
                    if hasattr(y_train, 'cpu'):
                        y_train = y_train.cpu().numpy()
                    if hasattr(sel_y, 'cpu'):
                        y_sel_np = sel_y.cpu().numpy()
                    else:
                        y_sel_np = sel_y

                    # Set context for evaluator (plots directory, epoch info)
                    if hasattr(trainer, 'result_manager') and trainer.result_manager:
                        plots_dir = trainer.result_manager.get_experiment_plot_dir(trainer.experiment_name)
                        # Set epoch context, images will be saved to epochinfo subdirectory
                        self.eval_loader.set_context(
                            plots_dir=plots_dir,
                            epoch_info=epoch_data  # Pass epoch info, images will be saved to epochinfo/ subdirectory
                        )

                    # Temporarily disable logging
                    original_level = logging.getLogger('src.eval_loader').level
                    logging.getLogger('src.eval_loader').setLevel(logging.WARNING)

                    # Use selected split as evaluator's "test channel" input
                    eval_results = self.eval_loader.evaluate(
                        config, eval_template_name,
                        X_train, y_train, train_pred,
                        sel_X, y_sel_np, sel_pred
                    )

                    # Restore logging level
                    logging.getLogger('src.eval_loader').setLevel(original_level)

                    # Build evaluation results string with split tag
                    split_tag = f"split:{epochinfo_split}"
                    metrics_parts = [f"{k}={v:.3f}" for k, v in eval_results.items() if v is not None]
                    metrics_parts.append(split_tag)
                    metrics_str = " | " + " | ".join(metrics_parts)
                    
                except Exception as e:
                    logger.warning(f"Failed to compute evaluation metrics: {e}")

        # Print in one line
        print(f"{basic_info}{metrics_str}")

    def _template_needs_train_pred(self, eval_template_name: str, config: Dict[str, Any]) -> bool:
        """Heuristically decide whether any metric in the template needs train predictions.

        Rules:
        - If any metric config sets needs_train_pred: true, return True.
        - Known metrics requiring train channel: metric name 'train_test_gap' or
          module evaluators.sklearn_metrics with function 'train_test_gap_evaluate'.
        """
        try:
            eval_templates = config.get('evaluation_templates', {})
            tpl = eval_templates.get(eval_template_name, {})
            # Flattened template support
            if isinstance(tpl, dict) and 'metrics' not in tpl:
                metrics = {k: v for k, v in tpl.items() if not str(k).startswith('_')}
            else:
                metrics = tpl.get('metrics', {}) or {}

            for metric_name, metric_cfg in metrics.items():
                # explicit flag
                if isinstance(metric_cfg, dict) and bool(metric_cfg.get('needs_train_pred', False)):
                    return True
                # heuristic based on known evaluator names
                name_low = str(metric_name).lower()
                if 'train_test_gap' in name_low:
                    return True
                if isinstance(metric_cfg, dict):
                    module_name = str(metric_cfg.get('file', ''))
                    func_name = str(metric_cfg.get('function', ''))
                    if module_name == 'sklearn_metrics' and func_name == 'train_test_gap_evaluate':
                        return True
            return False
        except Exception:
            return False
