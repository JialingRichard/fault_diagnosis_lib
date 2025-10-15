#!/usr/bin/env python3
"""
Fault Diagnosis Library - Entry Point
Configuration-driven benchmarking framework for multi-model, multi-dataset fault diagnosis.

Usage:
    python main.py [config_file]
    
Default config: configs/default_experiment.yaml
"""
import os
import sys
import argparse
import logging
import time

def _set_global_seed(cfg):
    """Set global random seeds (Python/NumPy/PyTorch) and optional determinism.

    Reads `global.seed` and `global.deterministic` from config. Uses 42 by default.
    """
    try:
        import random
        import numpy as np
        import torch
    except Exception:
        # Keep main flow running even if dependencies not present
        return None

    gcfg = (cfg or {}).get('global', {}) if isinstance(cfg, dict) else {}
    seed = gcfg.get('seed', 42)
    deterministic = bool(gcfg.get('deterministic', False))

    # Python/NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU/CUDA
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism (optional, may reduce performance)
        torch.backends.cudnn.deterministic = deterministic
        # Disable benchmark when deterministic for reproducibility
        if deterministic:
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    print(f"Global seed: {seed} | Deterministic: {deterministic}")
    return seed

# Ensure running from the benchmark directory and set the path
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(benchmark_dir)
sys.path.append(str(benchmark_dir))

from src.config_loader import ConfigLoader
from src.data_loader import DataLoader
from src.model_loader import ModelLoader
from src.training_loader import TrainingLoader
from src.eval_loader import EvalLoader
from src.result_manager import ResultManager

def setup_logging():
    """Configure basic logging (unused now, kept for reference)."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def expand_dataset_collections(config):
    """Expand dataset collections into individual experiments."""
    from pathlib import Path
    
    expanded_experiments = []
    
    for experiment in config.get('experiments', []):
        if 'dataset_collection' in experiment:
            # Handle dataset collection
            collection_name = experiment['dataset_collection']
            
            if collection_name not in config['datasets']:
                print(f" Dataset collection '{collection_name}' not found")
                continue
            
            collection_config = config['datasets'][collection_name]
            collection_path = Path(collection_config['collection_path'])
            
            if not collection_path.exists():
                print(f"Dataset collection path does not exist: {collection_path}")
                continue
            
            # Discover sub-datasets
            subdatasets = []
            for sub_dir in collection_path.iterdir():
                if sub_dir.is_dir():
                    # Require mandatory .npy files
                    required_files = ['train_X.npy', 'train_y.npy', 'test_X.npy', 'test_y.npy']
                    if all((sub_dir / f).exists() for f in required_files):
                        subdatasets.append(sub_dir.name)
            
            if not subdatasets:
                print(f" Dataset collection '{collection_name}' has no valid sub-datasets")
                continue
            
            print(f"Found dataset collection '{collection_name}': {len(subdatasets)} sub-datasets")
            print(f"   {', '.join(subdatasets)}")
            
            # Create dataset configuration for each sub-dataset
            for subdataset in subdatasets:
                # Create dataset configuration
                dataset_key = f"{collection_name}_{subdataset}"
                config['datasets'][dataset_key] = {
                    'train_data': f"{collection_config['collection_path']}/{subdataset}/train_X.npy",
                    'train_label': f"{collection_config['collection_path']}/{subdataset}/train_y.npy", 
                    'test_data': f"{collection_config['collection_path']}/{subdataset}/test_X.npy",
                    'test_label': f"{collection_config['collection_path']}/{subdataset}/test_y.npy",
                    'preprocessing': collection_config.get('preprocessing', {})
                }
                
                # Create experiment configuration
                new_experiment = experiment.copy()
                del new_experiment['dataset_collection']  # Remove the collection field
                new_experiment['dataset'] = dataset_key
                new_experiment['name'] = f"{experiment['name']}_{subdataset}"
                
                expanded_experiments.append(new_experiment)
        else:
            # Regular experiment, add directly
            expanded_experiments.append(experiment)

    # update experiments list
    config['experiments'] = expanded_experiments
    return config

def expand_grid_search(config):
    """
    Expand grid search configuration into multiple independent experiments
    supports grid search syntax in config {val1, val2, val3} to define parameter grids
    
    Args:
        config: original configuration
    
    Returns:
        expanded configuration
    """
    import itertools
    import re
    import ast
    import copy
    
    def parse_grid_values(value):
        """Parse set syntax and return a list of values"""
        if isinstance(value, str) and value.strip().startswith('{') and value.strip().endswith('}'):
            # Remove outer braces
            content = value.strip()[1:-1]
            try:
                # Try to parse using ast.literal_eval
                parsed = ast.literal_eval(f"[{content}]")
                return parsed
            except:
                # If it fails, try simple comma splitting
                items = [item.strip() for item in content.split(',')]
                parsed_items = []
                for item in items:
                    try:
                        parsed_items.append(ast.literal_eval(item))
                    except:
                        parsed_items.append(item)
                return parsed_items
        return [value]
    
    def find_grid_params(obj, path=""):
        """Recursively find all grid parameters"""
        grid_params = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str) and value.strip().startswith('{') and value.strip().endswith('}'):
                    grid_params[current_path] = parse_grid_values(value)
                elif isinstance(value, dict):
                    grid_params.update(find_grid_params(value, current_path))
        return grid_params
    
    def set_nested_value(obj, path, value):
        """Set values in a nested dictionary"""
        keys = path.split('.')
        current = obj
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def is_param_relevant_to_experiment(param_path, experiment):
        """Determine if a parameter is relevant to an experiment"""
        # Model parameters: only affect experiments using that model
        if param_path.startswith('models.'):
            model_name = param_path.split('.')[1]
            return experiment.get('model') == model_name

        # Training template parameters: only affect experiments using that training template
        if param_path.startswith('training_templates.'):
            template_name = param_path.split('.')[1]
            return experiment.get('training') == template_name

        # Dataset parameters: only affect experiments using that dataset
        if param_path.startswith('datasets.'):
            dataset_name = param_path.split('.')[1]
            return experiment.get('dataset') == dataset_name

        # Other global parameters: affect all experiments
        return True

    # Find all parameters that need grid search
    grid_params = find_grid_params(config)
    
    if not grid_params:
        return config
    
    print(f"Found grid search parameters: {len(grid_params)}")
    for param, values in grid_params.items():
        print(f"   {param}: {len(values)} values {values}")

    # Process each experiment separately
    final_experiments = []
    
    for experiment in config.get('experiments', []):
        # Find grid parameters relevant to the current experiment
        relevant_params = {}
        for param_path, values in grid_params.items():
            if is_param_relevant_to_experiment(param_path, experiment):
                relevant_params[param_path] = values
        
        if not relevant_params:
            # If there are no relevant grid parameters, keep the original experiment unchanged
            final_experiments.append(experiment)
        else:
            # Generate all parameter combinations for the experiment
            param_names = list(relevant_params.keys())
            param_values = list(relevant_params.values())
            combinations = list(itertools.product(*param_values))

            # Create a new experiment for each combination
            for i, combination in enumerate(combinations, 1):
                new_experiment = copy.deepcopy(experiment)
                
                # Create descriptive experiment names
                param_desc = []
                for param_name, param_value in zip(param_names, combination):
                    # Simplify parameter names (remove path prefix)
                    simple_name = param_name.split('.')[-1]
                    # Format parameter values
                    if isinstance(param_value, list):
                        # Uniformly process lists: convert to string and remove spaces and brackets
                        value_str = str(param_value).replace(" ", "").replace("[", "").replace("]", "").replace(",", "_")
                    elif isinstance(param_value, float):
                        value_str = f"{param_value:.0e}" if param_value < 0.01 else str(param_value)
                    else:
                        value_str = str(param_value)
                    param_desc.append(f"{simple_name}_{value_str}")
                
                # Generate descriptive experiment names
                param_suffix = "_".join(param_desc)
                new_experiment['name'] = f"{experiment['name']}_grid_{param_suffix}"

                # Record the parameter values for the current combination (for later use)
                new_experiment['_grid_params'] = dict(zip(param_names, combination))
                
                final_experiments.append(new_experiment)
    
    print(f"A total of {len(final_experiments)} experiment configurations will be generated")

    # Create final configuration
    final_config = copy.deepcopy(config)
    final_config['experiments'] = final_experiments

    # Clean up grid search syntax, keep original values
    for param_path in grid_params.keys():
        original_values = grid_params[param_path]
        set_nested_value(final_config, param_path, original_values[0])
    
    return final_config

    # Collect all expanded experiments
    # all_experiments = []
    # for i, expanded_config in enumerate(expanded_configs, 1):
    #     for experiment in expanded_config.get('experiments', []):
    #         # Add grid search parameter information for each experiment (for debugging)
    #         experiment['_grid_params'] = dict(zip(param_names, combinations[i-1]))
    #         all_experiments.append(experiment)
    
    # final_config['experiments'] = all_experiments
    
    # return final_config

def run_experiments(config_file: str = 'configs/default_experiment.yaml'):
    """
    Run the complete experimental process
    
    Args:
        config_file: 配置文件路径
    """
    # initialize result manager early so all subsequent outputs go to run/debug logs
    result_manager = ResultManager(config_file)
    
    print("=" * 80)
    print("Fault Diagnosis Benchmark Framework - 故障诊断基准测试")
    print("=" * 80)
    print(f"Config File: {config_file}")
    print(f"Work Root Dir: {os.getcwd()}")
    print()
    
    # 1. load config
    print("Step 1: Load experiment configuration")
    config_loader = ConfigLoader(config_file)
    config = config_loader.load_config()
    # 1.1 Set global random seed (can be read from config global.seed)
    _set_global_seed(config)

    # Print global information
    gcfg = (config.get('global') or {})
    print("Global Configuration:")
    print(f"  device: {gcfg.get('device', 'cpu')} | seed: {gcfg.get('seed', 'N/A')} | deterministic: {gcfg.get('deterministic', False)} | ckpt_policy: {gcfg.get('checkpoint_policy', 'best')}")
    if any(k in gcfg for k in ('author','version','date')):
        print(f"  author: {gcfg.get('author','-')} | version: {gcfg.get('version','-')} | date: {gcfg.get('date','-')}")
    print()

    # 1.5 Expand dataset collections
    config = expand_dataset_collections(config)

    # 1.6 Expand grid search
    config = expand_grid_search(config)
    
    experiments = config.get('experiments', [])
    print(f"   Found {len(experiments)} experiment configurations")
    for i, exp in enumerate(experiments, 1):
        dataset_info = exp.get('dataset', 'N/A')
        print(f"   {i}. {exp['name']} ({exp['model']} on {dataset_info})")
    print()

    # Pre-check (optional): Quickly verify monitor evaluator availability and return values before large-scale training
    gcfg = (config.get('global') or {})
    if gcfg.get('pre_test', False):
        print("Pre-check phase: Verify monitor configuration and evaluator availability for each experiment…")
        pretest_failures = []
        eval_loader = EvalLoader()
        from src.model_loader import ModelLoader as _ML
        _ml = _ML()
        from src.data_loader import DataLoader as _DL
        _dl = _DL()
        device = gcfg.get('device', 'cpu')
        for exp in experiments:
            logging.getLogger(__name__).debug(f"Pre-check started: {exp['name']}")
            try:
                # Prepare data and model
                X_train, X_test, y_train, y_test, metadata = _dl.prepare_data(config, exp['dataset'])
                model = _ml.load_model_from_config(
                    exp['model'], config,
                    input_dim=metadata.feature_dim,
                    output_dim=metadata.num_classes,
                    time_steps=X_train.shape[1] if len(X_train.shape) > 1 else None
                )
                # Take a minimal subset of the training set for a forward pass, reuse for all checks
                import torch
                model.eval()
                n = min(2, len(X_train))
                X_sub = torch.FloatTensor(X_train[:n]).to(device)
                with torch.no_grad():
                    out = model(X_sub)
                    if len(out.shape) == 3:
                        out = out[:, -1, :]
                    y_pred_sub = torch.argmax(out, dim=1).cpu().numpy()
                y_true_sub = y_train[:n].flatten()

                # Read epochinfo/monitor configuration
                tcfg = config['training_templates'][exp['training']]
                ep_t = tcfg.get('epochinfo')
                mon = tcfg.get('monitor', {})
                # More user-friendly missing parameter hints to avoid KeyError('split') type errors
                required_keys = ('metric', 'mode', 'split')
                missing = [k for k in required_keys if k not in mon]
                if missing:
                    raise KeyError(f"monitor is missing required fields: {missing}")
                metric = mon['metric']
                mode = mon['mode']
                split = mon['split']
                # get corresponding template and metric
                tpl = config['evaluation_templates'][ep_t]
                metric_map = {k: v for k, v in tpl.items() if not str(k).startswith('_')} if 'metrics' not in tpl else tpl['metrics']
                # Load and run all metrics under the epochinfo template one by one; only enforce numeric values for monitor metrics
                for m_name, m_cfg in metric_map.items():
                    evaluator = eval_loader._load_evaluator(m_name, m_cfg)
                    try:
                        # use training subset for both train/test channels, just for runtime check
                        val = evaluator(
                            X_train[:n], y_true_sub, y_pred_sub,
                            X_train[:n], y_true_sub, y_pred_sub
                        )
                        if m_name == metric:
                            float(val)
                    except Exception as e:
                        raise RuntimeError(f"Metric '{m_name}' pre-check failed: {e}")

                # Final evaluation template: only check without error reporting, no numeric enforcement
                final_tpl_name = exp.get('evaluation')
                metric_map_final = {}
                if final_tpl_name:
                    tpl_final = config['evaluation_templates'][final_tpl_name]
                    metric_map_final = {k: v for k, v in tpl_final.items() if not str(k).startswith('_')} if 'metrics' not in tpl_final else tpl_final['metrics']
                    for m_name, m_cfg in metric_map_final.items():
                        evaluator_f = eval_loader._load_evaluator(m_name, m_cfg)
                        try:
                            _ = evaluator_f(
                                X_train[:n], y_true_sub, y_pred_sub,
                                X_train[:n], y_true_sub, y_pred_sub
                            )
                        except Exception as e:
                            raise RuntimeError(f"Final evaluation metric '{m_name}' pre-check failed: {e}")

                print(f"   Pre-check passed: {exp['name']} | Monitor:{metric}-{mode}-{split} | Template '{ep_t}' has {len(metric_map)} items | Final template '{final_tpl_name}' has {len(metric_map_final) if final_tpl_name else 0} items")
            except Exception as e:
                # Log to console and log (including full stack)
                pretest_failures.append((exp['name'], str(e)))
                print(f"   Pre-check failed: {exp['name']} | Error: {e}")
                logging.getLogger(__name__).exception(f"Pre-check failed: {exp['name']} | Details")
                # Write to error.log, including context and stack
                try:
                    import traceback
                    exp_ctx = (
                        f"Experiment: {exp['name']}\n"
                        f"Model:{exp['model']} | Data:{exp.get('dataset','N/A')} | Train:{exp.get('training','N/A')} | Eval:{exp.get('evaluation','N/A')}\n"
                    )
                    result_manager.log_experiment_error(exp_ctx, traceback.format_exc())
                except Exception:
                    pass
        if pretest_failures:
            raise RuntimeError(f"Pre-check failed for {len(pretest_failures)} experiments: {[n for n,_ in pretest_failures]}")
        print("Pre-check completed: All experiments' monitor evaluators are available\n")

    # 2. Initialize system components (including result manager)
    print("Step 2: Initialize system components")
    data_loader = DataLoader()
    model_loader = ModelLoader()
    training_loader = TrainingLoader()
    eval_loader = EvalLoader()
    print("All components are ready")
    print()

    # 3. Run experiments
    results_summary = []
    
    for exp_idx, experiment in enumerate(experiments, 1):
        print(f"\n{'='*20} Experiment {exp_idx}/{len(experiments)}: {experiment['name']} {'='*20}")

        # Get epochinfo and monitor information from training config
        training_config = config['training_templates'][experiment['training']]
        epochinfo_name = training_config.get('epochinfo', 'default')
        monitor_cfg = training_config.get('monitor', {}) or {}
        monitor_str = ""
        try:
            m_metric = str(monitor_cfg.get('metric', '')).strip()
            m_mode = str(monitor_cfg.get('mode', '')).strip()
            m_split = str(monitor_cfg.get('split', '')).strip()
            if m_metric and m_mode and m_split:
                monitor_str = f" | Monitor:{m_metric}-{m_mode}-{m_split}"
        except Exception:
            pass

        print(f"Model:{experiment['model']} | Data:{experiment['dataset']} | Train:{experiment['training']} | Epoch:{epochinfo_name} {monitor_str} | Eval:{experiment['evaluation']}")
        print("-" * 80)
        
        try:
            import time
            exp_start_ts = time.time()
            print(f"Experiment start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp_start_ts))}")
            # Apply grid search parameters (if any)
            if '_grid_params' in experiment:
                import copy
                def set_nested_value(obj, path, value):
                    """Set value in nested dictionary"""
                    keys = path.split('.')
                    current = obj
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[keys[-1]] = value
                
                temp_config = copy.deepcopy(config)
                for param_path, param_value in experiment['_grid_params'].items():
                    set_nested_value(temp_config, param_path, param_value)
                config_for_experiment = temp_config
            else:
                config_for_experiment = config

            # 3.1 Data Loading
            X_train, X_test, y_train, y_test, metadata = data_loader.prepare_data(
                config_for_experiment, experiment['dataset']
            )
            print(f"Data: {X_train.shape[0]:,} train + {X_test.shape[0]:,} test | Features: {metadata.feature_dim} | Classes: {metadata.num_classes}")

            # Model Loading - Fix: Pass the correct output_dim (number of classes)
            model = model_loader.load_model_from_config(
                experiment['model'], config_for_experiment, 
                input_dim=metadata.feature_dim,
                output_dim=metadata.num_classes,  # Add output dimension (number of classes)
                time_steps=X_train.shape[1] if len(X_train.shape) > 1 else None
            )
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Model: {param_count:,} parameters")

            # Start Training
            print(f"Starting training...")
            trainer = training_loader.create_trainer(
                config_for_experiment, experiment['training'], model,
                X_train, y_train, X_test, y_test
            )
            # Set eval_loader and result_manager for the trainer
            trainer.eval_loader = eval_loader
            trainer.result_manager = result_manager
            trainer.experiment_name = experiment['name']
            training_results = trainer.train()

            # 3.4 Evaluation (default: test set)
            print(f"Starting evaluation...")
            # If recorded, print the best checkpoint used for this evaluation
            sel_ckpt = training_results.get('selected_checkpoint')
            if sel_ckpt:
                print(f"Using best checkpoint for final evaluation: {sel_ckpt}")

            # Evaluate using the actual data used by the trainer (which may be a subset)
            actual_X_train = training_results.get('actual_X_train', X_train)
            actual_y_train = training_results.get('actual_y_train', y_train)
            actual_X_test = training_results.get('actual_X_test', X_test)
            actual_y_test = training_results.get('actual_y_test', y_test)

            # Set plots directory for the evaluator
            plots_dir = result_manager.get_experiment_plot_dir(experiment['name'])

            # Set evaluator context (for image saving, etc.)
            eval_loader.set_context(
                plots_dir=plots_dir,
                epoch_info=None  # None means final evaluation, will be saved to main plots directory
            )
            
            eval_results = eval_loader.evaluate(
                config, experiment['evaluation'],
                actual_X_train, actual_y_train, training_results['train_predictions'],
                actual_X_test, actual_y_test, training_results['test_predictions']
            )

            # Prepare experiment-level result summary: If the experiment configures summary and requires val split, then compute a validation set evaluation (using the experiment's evaluation template)
            exp_summary_cfg = experiment.get('summary', {}) if isinstance(experiment.get('summary', {}), dict) else {}
            summary_eval_results = None
            if exp_summary_cfg.get('keep_only_best', False) and str(exp_summary_cfg.get('split', 'test')).lower() == 'val':
                try:
                    y_val = trainer.y_val
                    y_val_np = y_val.cpu().numpy() if hasattr(y_val, 'cpu') else y_val
                    val_pred = getattr(trainer, '_cached_val_pred', None)
                    if val_pred is None:
                        val_pred = trainer.predict(trainer.X_val)
                    eval_loader.set_context(plots_dir=plots_dir, epoch_info=None)
                    summary_eval_results = eval_loader.evaluate(
                        config, experiment['evaluation'],
                        actual_X_train, actual_y_train, training_results['train_predictions'],
                        trainer.X_val, y_val_np, val_pred
                    )
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Summary reduction: Validation set evaluation failed, falling back to test set: {e}")

            print(f"Training complete: {training_results['total_epochs']} epochs | Validation loss: {training_results['final_val_loss']:.4f}")
            print(f"Evaluation results: ", end="")
            for metric, score in eval_results.items():
                if score is not None:
                    print(f"{metric}={score:.4f} ", end="")
            print()

            # Save results
            results_summary.append({
                'name': experiment['name'],
                'model': experiment['model'],
                'dataset': experiment['dataset'],
                'training': experiment['training'],
                'parameters': param_count,
                'epochs': training_results['total_epochs'],
                'val_loss': training_results['final_val_loss'],
                'eval_results': eval_results,
                'summary_cfg': exp_summary_cfg,
                'summary_eval_results': summary_eval_results,
                # Add dataset details
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'feature_dim': metadata.feature_dim,
                'num_classes': metadata.num_classes,
                'sequence_length': X_train.shape[1] if len(X_train.shape) > 1 else None
            })

            # Record experiment end time, duration, and resource usage
            exp_end_ts = time.time()
            extra_timing = {}
            try:
                import torch
                if torch.cuda.is_available():
                    extra_timing['cuda_max_mem_MiB'] = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
            except Exception:
                pass
            try:
                # Sample size and other information
                extra_timing.update({
                    'train_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0]
                })
            except Exception:
                pass
            result_manager.log_experiment_timing(experiment['name'], exp_start_ts, exp_end_ts, extra=extra_timing)
            print(f"Experiment end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp_end_ts))} | Duration: {exp_end_ts - exp_start_ts:.1f}s")
            print(f"Experiment complete\n")

        except Exception as e:
            # Build error log information, including the full context of the experiment
            error_info = f"{'='*20} Experiment {exp_idx}/{len(experiments)}: {experiment['name']} {'='*20}\n"
            error_info += f"Model:{experiment['model']} | Data:{experiment['dataset']} | Train:{experiment['training']} | Epoch:{epochinfo_name} | Eval:{experiment['evaluation']}\n"
            error_info += "-" * 80 + "\n"

            # If there is data information, include it
            try:
                if 'X_train' in locals() and 'X_test' in locals() and 'metadata' in locals():
                    error_info += f"Data: {X_train.shape[0]:,} train + {X_test.shape[0]:,} test | Features: {metadata.feature_dim} | Classes: {metadata.num_classes}\n"
            except:
                pass

            print(f"Experiment failed: {str(e)}")
            logging.error(f"Experiment failed: {experiment['name']}", exc_info=True)

            # Write error information to error.log
            result_manager.log_experiment_error(error_info, str(e))
            try:
                import time
                exp_end_ts = time.time()
                result_manager.log_experiment_timing(experiment['name'], exp_start_ts, exp_end_ts, extra={'failed': True})
                print(f"Experiment end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp_end_ts))} | Duration: {exp_end_ts - exp_start_ts:.1f}s")
            except Exception:
                pass
            
        print()

    # Optional: Simplify to the best result by (model + dataset) (grid_only_best)
    # Experiment-level simplification: Group by (model + dataset), if any experiment in the group declares keep_only_best,
    # use the summary configuration declared by the first experiment in the group as the criterion to simplify the group
    if results_summary:
        from collections import defaultdict
        group_map = defaultdict(list)
        for res in results_summary:
            group_map[(res.get('model'), res.get('dataset'))].append(res)
        new_results = []
        for key, items in group_map.items():
            # Find the summary configuration declared within the group
            cfg_items = [it for it in items if isinstance(it.get('summary_cfg'), dict) and it['summary_cfg'].get('keep_only_best', False)]
            if not cfg_items:
                new_results.extend(items)
                continue
            base_cfg = cfg_items[0]['summary_cfg']
            metric = str(base_cfg.get('metric', '')).strip()
            mode = str(base_cfg.get('mode', 'max')).lower()
            split = str(base_cfg.get('split', 'test')).lower()
            if not metric:
                raise RuntimeError(f"No metrics found {key}")
            def _numeric(v):
                return isinstance(v, (int, float))
            def score(it):
                src = (it.get('eval_results') or {}) if split == 'test' else (it.get('summary_eval_results') or {})
                if not _numeric(src.get(metric)):
                    raise RuntimeError(f"summary : Experiment {it.get('name')} lacks or non-numeric metric={metric}, split={split}")
                return float(src.get(metric))
            def better(a, b):
                return a < b if mode == 'min' else a > b
            best = None
            best_s = None
            for it in items:
                s = score(it)
                if best is None or better(s, best_s):
                    best, best_s = it, s
            new_results.append(best)
        if len(new_results) != len(results_summary):
            # print(f" summary 精简：每个(模型+数据集)已按实验级仅保留最优，共{len(new_results)}项（原{len(results_summary)}）")
            print(f" Each grid search has been simplified, total {len(new_results)} items (original {len(results_summary)})")
        results_summary = new_results

    # 4. Generate summary report
    print("Step 4: Generate experiment summary report")
    print("=" * 80)
    print("Experiment comparison summary")
    print("=" * 80)
    
    if results_summary:
        # Group by dataset
        dataset_groups = {}
        for result in results_summary:
            dataset = result['dataset']
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(result)

        # Prepare export data
        export_data = {}

        # Generate separate comparison tables for each dataset
        for dataset, results in dataset_groups.items():
            print(f"\nDataset: {dataset}")
            print("-" * 60)

            # Dynamically get all available evaluation metrics
            all_metrics = set()
            for result in results:
                if 'eval_results' in result and result['eval_results']:
                    all_metrics.update(result['eval_results'].keys())

            # Sort metric names to ensure consistent display order
            sorted_metrics = sorted(all_metrics)

            # Dynamically generate header
            header = f"{'Experiment Name':<25} {'Model':<8} {'Parameters':<10} {'Epochs':<6} {'Val Loss':<10}"
            for metric in sorted_metrics:
                header += f" {metric:<10}"
            print(header)
            print("-" * len(header))
            
            for result in results:
                params = result.get('parameters', 0) or 0
                epochs = result.get('epochs', 0) or 0  
                val_loss = result.get('val_loss', 0) or 0
                
                row = f"{result['name']:<25} {result['model']:<8} {params:<10,} " \
                      f"{epochs:<6} {val_loss:<10.4f}"

                # Dynamically add all evaluation metrics
                for metric in sorted_metrics:
                    value = result['eval_results'].get(metric, 0) if result.get('eval_results') else 0
                    if isinstance(value, (int, float)):
                        row += f" {value:<10.4f}"
                    else:
                        row += f" {'N/A':<10}"
                
                print(row)

            # Best results summary
            if len(results) > 1:
                print(f"\n{dataset} Dataset Best Results:")

                # Find the best result for each metric (excluding non-performance metrics like test_samples)
                performance_metrics = [m for m in sorted_metrics
                                       if m not in ['test_samples', 'train_test_gap']]

                for metric in performance_metrics:
                    # Find the best result for this metric
                    best_result = max(results,
                                    key=lambda x: x['eval_results'].get(metric, 0)
                                    if x.get('eval_results') else 0)
                    best_value = best_result['eval_results'].get(metric, 0) if best_result.get('eval_results') else 0
                    
                    if isinstance(best_value, (int, float)) and best_value > 0:
                        print(f"   Best {metric}: {best_result['name']} ({metric}: {best_value:.4f})")

            # Prepare export data for this dataset
            export_rows = []
            for result in results:
                row = {
                    'Experiment Name': result['name'],
                    'Model': result['model'],
                    'Train Samples': result.get('train_samples', 0) or 0,
                    'Test Samples': result.get('test_samples', 0) or 0,
                    'Feature Dim': result.get('feature_dim', 0) or 0,
                    'Num Classes': result.get('num_classes', 0) or 0,
                    'Sequence Length': result.get('sequence_length', 0) or 0,
                    'Parameters': result.get('parameters', 0) or 0,
                    'Epochs': result.get('epochs', 0) or 0,
                    'Val Loss': result.get('val_loss', 0) or 0
                }
                # Add evaluation metrics
                for metric in sorted_metrics:
                    value = result['eval_results'].get(metric, 0) if result.get('eval_results') else 0
                    row[metric] = value if isinstance(value, (int, float)) else 0
                export_rows.append(row)
            
            export_data[dataset] = export_rows

        # Export to Excel files
        try:
            import pandas as pd
            from datetime import datetime

            # Use the experiment run directory (instead of the results root directory)
            results_dir = result_manager.run_dir
            os.makedirs(results_dir, exist_ok=True)

            # Generate filenames with timestamps
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. Generate Excel file categorized by dataset
            dataset_excel_filename = f"results_by_dataset_{timestamp}.xlsx"
            dataset_excel_path = os.path.join(results_dir, dataset_excel_filename)
            
            with pd.ExcelWriter(dataset_excel_path, engine='openpyxl') as writer:
                for dataset, rows in export_data.items():
                    if rows:
                        df = pd.DataFrame(rows)
                        # clean dataset name for sheet name
                        clean_dataset_name = dataset.replace('/', '_').replace('\\', '_')
                        sheet_name = clean_dataset_name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 2. Generate Excel file categorized by model
            model_excel_filename = f"results_by_model_{timestamp}.xlsx"
            model_excel_path = os.path.join(results_dir, model_excel_filename)

            # Prepare data grouped by model
            model_groups = {}
            for result in results_summary:
                model = result['model']
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append(result)
            
            with pd.ExcelWriter(model_excel_path, engine='openpyxl') as writer:
                for model, results in model_groups.items():
                    if results:
                        # Prepare model view data
                        model_rows = []
                        for result in results:
                            dataset_name = result['dataset']
                            
                            row = {
                                'Dataset': dataset_name,
                                'Experiment Name': result['name'],
                                'Train Samples': result.get('train_samples', 0) or 0,
                                'Test Samples': result.get('test_samples', 0) or 0,
                                'Feature Dim': result.get('feature_dim', 0) or 0,
                                'Num Classes': result.get('num_classes', 0) or 0,
                                'Sequence Length': result.get('sequence_length', 0) or 0,
                                'Parameters': result.get('parameters', 0) or 0,
                                'Epochs': result.get('epochs', 0) or 0,
                                'Val Loss': result.get('val_loss', 0) or 0
                            }
                            
                            # Add evaluation metrics
                            if result.get('eval_results'):
                                for metric, value in result['eval_results'].items():
                                    if isinstance(value, (int, float)):
                                        row[metric] = value
                                    else:
                                        row[metric] = 0
                            
                            model_rows.append(row)

                        # Create sheet for model
                        df_model = pd.DataFrame(model_rows)
                        sheet_name = model[:31]  # Excel sheet name limit
                        df_model.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"\n Results to Excel:")
            print(f"  By Dataset: {dataset_excel_path}")
            print(f"  By Model: {model_excel_path}")

        except ImportError:
            print("\nWarning: pandas or openpyxl is not installed, skipping Excel export.")
            print("Please install pandas: pip install pandas openpyxl")
        except Exception as e:
            print(f"\nWarning: Failed to export Excel file: {str(e)}")
    else:
        print("No successful experiments to summarize.")
    
    print()
    print("All experiments completed")
    print("=" * 80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Fault Diagnosis Benchmark Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python main.py                          # Use default config
  python main.py configs/my_config.yaml   # Use custom config
        """
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default='configs/default_experiment.yaml',
        help='Configuration file path (default: configs/default_experiment.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check configuration file
    if not os.path.exists(args.config):
        print(f"Error: Configuration file does not exist: {args.config}")
        return 1
    
    try:
        run_experiments(args.config)
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error("Error", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
