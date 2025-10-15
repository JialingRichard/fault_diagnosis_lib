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

# 确保从benchmark目录运行并设置路径
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
            
            # 为每个子数据集创建数据集配置
            for subdataset in subdatasets:
                # 创建数据集配置
                dataset_key = f"{collection_name}_{subdataset}"
                config['datasets'][dataset_key] = {
                    'train_data': f"{collection_config['collection_path']}/{subdataset}/train_X.npy",
                    'train_label': f"{collection_config['collection_path']}/{subdataset}/train_y.npy", 
                    'test_data': f"{collection_config['collection_path']}/{subdataset}/test_X.npy",
                    'test_label': f"{collection_config['collection_path']}/{subdataset}/test_y.npy",
                    'preprocessing': collection_config.get('preprocessing', {})
                }
                
                # 创建实验配置
                new_experiment = experiment.copy()
                del new_experiment['dataset_collection']  # 移除collection字段
                new_experiment['dataset'] = dataset_key
                new_experiment['name'] = f"{experiment['name']}_{subdataset}"
                
                expanded_experiments.append(new_experiment)
        else:
            # 普通实验，直接添加
            expanded_experiments.append(experiment)
    
    # 更新实验列表
    config['experiments'] = expanded_experiments
    return config

def expand_grid_search(config):
    """
    展开网格搜索配置为多个独立实验
    
    支持在配置中使用集合语法 {val1, val2, val3} 来定义参数网格
    只对相关实验应用相关参数组合
    
    Args:
        config: 原始配置
    
    Returns:
        展开后的配置
    """
    import itertools
    import re
    import ast
    import copy
    
    def parse_grid_values(value):
        """解析集合语法，返回值列表"""
        if isinstance(value, str) and value.strip().startswith('{') and value.strip().endswith('}'):
            # 移除外层的大括号
            content = value.strip()[1:-1]
            try:
                # 尝试使用ast.literal_eval解析
                parsed = ast.literal_eval(f"[{content}]")
                return parsed
            except:
                # 如果失败，尝试简单的逗号分割
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
        """递归查找所有网格参数"""
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
        """设置嵌套字典中的值"""
        keys = path.split('.')
        current = obj
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def is_param_relevant_to_experiment(param_path, experiment):
        """判断参数是否与实验相关"""
        # 模型参数：只影响使用该模型的实验
        if param_path.startswith('models.'):
            model_name = param_path.split('.')[1]
            return experiment.get('model') == model_name
        
        # 训练模板参数：只影响使用该训练模板的实验
        if param_path.startswith('training_templates.'):
            template_name = param_path.split('.')[1]
            return experiment.get('training') == template_name
        
        # 数据集参数：只影响使用该数据集的实验
        if param_path.startswith('datasets.'):
            dataset_name = param_path.split('.')[1]
            return experiment.get('dataset') == dataset_name
        
        # 其他全局参数：影响所有实验
        return True
    
    # 查找所有需要网格搜索的参数
    grid_params = find_grid_params(config)
    
    if not grid_params:
        return config
    
    print(f"发现网格搜索参数: {len(grid_params)} 个")
    for param, values in grid_params.items():
        print(f"   {param}: {len(values)} 个值 {values}")
    
    # 分别处理每个实验
    final_experiments = []
    
    for experiment in config.get('experiments', []):
        # 找到与当前实验相关的网格参数
        relevant_params = {}
        for param_path, values in grid_params.items():
            if is_param_relevant_to_experiment(param_path, experiment):
                relevant_params[param_path] = values
        
        if not relevant_params:
            # 如果没有相关的网格参数，保持原实验不变
            final_experiments.append(experiment)
        else:
            # 生成该实验的所有参数组合
            param_names = list(relevant_params.keys())
            param_values = list(relevant_params.values())
            combinations = list(itertools.product(*param_values))
            
            # 为每个组合创建新的实验
            for i, combination in enumerate(combinations, 1):
                new_experiment = copy.deepcopy(experiment)
                
                # 生成包含参数值的描述性名称
                param_desc = []
                for param_name, param_value in zip(param_names, combination):
                    # 简化参数名（去掉路径前缀）
                    simple_name = param_name.split('.')[-1]
                    # 格式化参数值
                    if isinstance(param_value, list):
                        # 统一处理列表：转为字符串并去掉空格和括号
                        value_str = str(param_value).replace(" ", "").replace("[", "").replace("]", "").replace(",", "_")
                    elif isinstance(param_value, float):
                        value_str = f"{param_value:.0e}" if param_value < 0.01 else str(param_value)
                    else:
                        value_str = str(param_value)
                    param_desc.append(f"{simple_name}_{value_str}")
                
                # 创建描述性的实验名称
                param_suffix = "_".join(param_desc)
                new_experiment['name'] = f"{experiment['name']}_grid_{param_suffix}"
                
                # 记录当前组合的参数值（用于后续应用）
                new_experiment['_grid_params'] = dict(zip(param_names, combination))
                
                final_experiments.append(new_experiment)
    
    print(f"总共将生成 {len(final_experiments)} 个实验配置")
    
    # 创建最终配置
    final_config = copy.deepcopy(config)
    final_config['experiments'] = final_experiments
    
    # 清理网格搜索语法，保持原始值
    for param_path in grid_params.keys():
        original_values = grid_params[param_path]
        set_nested_value(final_config, param_path, original_values[0])
    
    return final_config
    
    # 收集所有展开的实验
    all_experiments = []
    for i, expanded_config in enumerate(expanded_configs, 1):
        for experiment in expanded_config.get('experiments', []):
            # 为每个实验添加网格搜索参数信息（用于调试）
            experiment['_grid_params'] = dict(zip(param_names, combinations[i-1]))
            all_experiments.append(experiment)
    
    final_config['experiments'] = all_experiments
    
    return final_config

def run_experiments(config_file: str = 'configs/default_experiment.yaml'):
    """
    运行完整的实验流程
    
    Args:
        config_file: 配置文件路径
    """
    # 提前初始化结果管理器，使所有后续输出进入 run/debug 日志
    result_manager = ResultManager(config_file)
    
    print("=" * 80)
    print("Fault Diagnosis Library - 故障诊断基准测试")
    print("=" * 80)
    print(f"配置文件: {config_file}")
    print(f"工作目录: {os.getcwd()}")
    print()
    
    # 1. 加载配置
    print("步骤 1: 加载实验配置")
    config_loader = ConfigLoader(config_file)
    config = config_loader.load_config()
    # 1.1 设置全局随机种子（可从配置 global.seed 读取）
    _set_global_seed(config)
    
    # 打印全局信息
    gcfg = (config.get('global') or {})
    print("全局配置:")
    print(f"  device: {gcfg.get('device', 'cpu')} | seed: {gcfg.get('seed', 'N/A')} | deterministic: {gcfg.get('deterministic', False)} | ckpt_policy: {gcfg.get('checkpoint_policy', 'best')}")
    if any(k in gcfg for k in ('author','version','date')):
        print(f"  author: {gcfg.get('author','-')} | version: {gcfg.get('version','-')} | date: {gcfg.get('date','-')}")
    print()

    # 1.5 展开数据集合集
    config = expand_dataset_collections(config)
    
    # 1.6 展开网格搜索
    config = expand_grid_search(config)
    
    experiments = config.get('experiments', [])
    print(f"   发现 {len(experiments)} 个实验配置")
    for i, exp in enumerate(experiments, 1):
        dataset_info = exp.get('dataset', 'N/A')
        print(f"   {i}. {exp['name']} ({exp['model']} on {dataset_info})")
    print()

    # 预检（可选）：在大规模训练前，快速验证 monitor evaluator 是否可用且返回数值
    gcfg = (config.get('global') or {})
    if gcfg.get('pre_test', False):
        print("预检阶段: 验证各实验的 monitor 配置与 evaluator 可用性…")
        pretest_failures = []
        eval_loader = EvalLoader()
        from src.model_loader import ModelLoader as _ML
        _ml = _ML()
        from src.data_loader import DataLoader as _DL
        _dl = _DL()
        device = gcfg.get('device', 'cpu')
        for exp in experiments:
            logging.getLogger(__name__).debug(f"预检开始: {exp['name']}")
            try:
                # 数据与模型
                X_train, X_test, y_train, y_test, metadata = _dl.prepare_data(config, exp['dataset'])
                model = _ml.load_model_from_config(
                    exp['model'], config,
                    input_dim=metadata.feature_dim,
                    output_dim=metadata.num_classes,
                    time_steps=X_train.shape[1] if len(X_train.shape) > 1 else None
                )
                # 取训练集的极小子集进行一次前向，复用到所有校验
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

                # 读取 epochinfo/monitor 配置
                tcfg = config['training_templates'][exp['training']]
                ep_t = tcfg.get('epochinfo')
                mon = tcfg.get('monitor', {})
                # 更友好的缺参提示，避免 KeyError('split') 这类含糊错误
                required_keys = ('metric', 'mode', 'split')
                missing = [k for k in required_keys if k not in mon]
                if missing:
                    raise KeyError(f"monitor 缺少必填字段: {missing}")
                metric = mon['metric']
                mode = mon['mode']
                split = mon['split']
                # 取对应模板与指标
                tpl = config['evaluation_templates'][ep_t]
                metric_map = {k: v for k, v in tpl.items() if not str(k).startswith('_')} if 'metrics' not in tpl else tpl['metrics']
                # 逐一加载并运行 epochinfo 模板下的所有指标；仅对 monitor 指标强制为数值
                for m_name, m_cfg in metric_map.items():
                    evaluator = eval_loader._load_evaluator(m_name, m_cfg)
                    try:
                        # 用训练子集同时填入 train/test 通道，仅做可运行性校验
                        val = evaluator(
                            X_train[:n], y_true_sub, y_pred_sub,
                            X_train[:n], y_true_sub, y_pred_sub
                        )
                        if m_name == metric:
                            float(val)
                    except Exception as e:
                        raise RuntimeError(f"指标 '{m_name}' 预检失败: {e}")

                # 最终评估模板：仅校验不报错，不强制数值
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
                            raise RuntimeError(f"最终评估指标 '{m_name}' 预检失败: {e}")

                print(f"   预检通过: {exp['name']} | Monitor:{metric}-{mode}-{split} | 模板 '{ep_t}' 共{len(metric_map)}项 | 最终模板 '{final_tpl_name}' 共{len(metric_map_final) if final_tpl_name else 0}项")
            except Exception as e:
                # 记录到控制台与日志（包含完整栈）
                pretest_failures.append((exp['name'], str(e)))
                print(f"   预检失败: {exp['name']} | 错误: {e}")
                logging.getLogger(__name__).exception(f"预检失败: {exp['name']} | 详情")
                # 写入 error.log，包含上下文与堆栈
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
            raise RuntimeError(f"预检失败 {len(pretest_failures)} 个实验: {[n for n,_ in pretest_failures]}")
        print("预检完成: 所有实验的 monitor evaluator 均可用\n")
    
    # 2. 初始化系统组件（包括结果管理器）
    print("步骤 2: 初始化系统组件")
    data_loader = DataLoader()
    model_loader = ModelLoader()
    training_loader = TrainingLoader()
    eval_loader = EvalLoader()
    print("所有组件就绪")
    print()
    
    # 3. 运行实验
    results_summary = []
    
    for exp_idx, experiment in enumerate(experiments, 1):
        print(f"\n{'='*20} 实验 {exp_idx}/{len(experiments)}: {experiment['name']} {'='*20}")
        
        # 获取训练配置中的epochinfo与monitor信息
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
            print(f"实验开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp_start_ts))}")
            # 应用网格搜索参数（如果有的话）
            if '_grid_params' in experiment:
                import copy
                def set_nested_value(obj, path, value):
                    """设置嵌套字典中的值"""
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
            
            # 3.1 数据加载
            # 数据加载
            X_train, X_test, y_train, y_test, metadata = data_loader.prepare_data(
                config_for_experiment, experiment['dataset']
            )
            print(f"数据: {X_train.shape[0]:,}训练+{X_test.shape[0]:,}测试 | 特征:{metadata.feature_dim} | 类别:{metadata.num_classes}")
            
            # 模型加载 - 修复：传递正确的output_dim（类别数量）
            model = model_loader.load_model_from_config(
                experiment['model'], config_for_experiment, 
                input_dim=metadata.feature_dim,
                output_dim=metadata.num_classes,  # 添加输出维度（类别数量）
                time_steps=X_train.shape[1] if len(X_train.shape) > 1 else None
            )
            param_count = sum(p.numel() for p in model.parameters())
            print(f"模型: {param_count:,}参数")
            
            # 开始训练
            print(f"开始训练...")
            trainer = training_loader.create_trainer(
                config_for_experiment, experiment['training'], model,
                X_train, y_train, X_test, y_test
            )
            # 设置eval_loader和result_manager给训练器
            trainer.eval_loader = eval_loader
            trainer.result_manager = result_manager
            trainer.experiment_name = experiment['name']
            training_results = trainer.train()
            
            # 3.4 评估（默认：测试集）
            print(f"开始评估...")
            # 如有记录，打印本次评估所用的最佳checkpoint
            sel_ckpt = training_results.get('selected_checkpoint')
            if sel_ckpt:
                print(f"使用最佳checkpoint进行最终评估: {sel_ckpt}")
            
            # 使用训练器实际使用的数据进行评估（可能是子集）
            actual_X_train = training_results.get('actual_X_train', X_train)
            actual_y_train = training_results.get('actual_y_train', y_train)
            actual_X_test = training_results.get('actual_X_test', X_test)
            actual_y_test = training_results.get('actual_y_test', y_test)
            
            # 为绘图evaluator设置plots目录
            plots_dir = result_manager.get_experiment_plot_dir(experiment['name'])
            
            # 设置评估器上下文（用于图像保存等）
            eval_loader.set_context(
                plots_dir=plots_dir,
                epoch_info=None  # None表示最终评估，会保存到主plots目录
            )
            
            eval_results = eval_loader.evaluate(
                config, experiment['evaluation'],
                actual_X_train, actual_y_train, training_results['train_predictions'],
                actual_X_test, actual_y_test, training_results['test_predictions']
            )

            # 实验级结果精简的准备：若该实验配置了 summary 且需要 val split，则计算一次验证集评估（沿用实验的 evaluation 模板）
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
                    logging.getLogger(__name__).warning(f"summary 精简: 计算验证集评估失败，将回退到测试集: {e}")
            
            print(f"训练完成: {training_results['total_epochs']}轮 | 验证损失: {training_results['final_val_loss']:.4f}")
            print(f"评估结果: ", end="")
            for metric, score in eval_results.items():
                if score is not None:
                    print(f"{metric}={score:.4f} ", end="")
            print()
            
            # 保存结果
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
                # 添加数据集详细信息
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'feature_dim': metadata.feature_dim,
                'num_classes': metadata.num_classes,
                'sequence_length': X_train.shape[1] if len(X_train.shape) > 1 else None
            })
            
            # 记录实验结束时间与耗时、资源占用
            exp_end_ts = time.time()
            extra_timing = {}
            try:
                import torch
                if torch.cuda.is_available():
                    extra_timing['cuda_max_mem_MiB'] = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
            except Exception:
                pass
            try:
                # 样本量等信息
                extra_timing.update({
                    'train_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0]
                })
            except Exception:
                pass
            result_manager.log_experiment_timing(experiment['name'], exp_start_ts, exp_end_ts, extra=extra_timing)
            print(f"实验结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp_end_ts))} | 耗时: {exp_end_ts - exp_start_ts:.1f}s")
            print(f"实验完成\n")
            
        except Exception as e:
            # 构建错误日志信息，包含实验的完整上下文
            error_info = f"{'='*20} 实验 {exp_idx}/{len(experiments)}: {experiment['name']} {'='*20}\n"
            error_info += f"Model:{experiment['model']} | Data:{experiment['dataset']} | Train:{experiment['training']} | Epoch:{epochinfo_name} | Eval:{experiment['evaluation']}\n"
            error_info += "-" * 80 + "\n"
            
            # 如果有数据信息，也包含进去
            try:
                if 'X_train' in locals() and 'X_test' in locals() and 'metadata' in locals():
                    error_info += f"数据: {X_train.shape[0]:,}训练+{X_test.shape[0]:,}测试 | 特征:{metadata.feature_dim} | 类别:{metadata.num_classes}\n"
            except:
                pass
            
            print(f"实验失败: {str(e)}")
            logging.error(f"实验失败: {experiment['name']}", exc_info=True)
            
            # 将错误信息写入error.log
            result_manager.log_experiment_error(error_info, str(e))
            try:
                import time
                exp_end_ts = time.time()
                result_manager.log_experiment_timing(experiment['name'], exp_start_ts, exp_end_ts, extra={'failed': True})
                print(f"实验结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp_end_ts))} | 耗时: {exp_end_ts - exp_start_ts:.1f}s")
            except Exception:
                pass
            
        print()
    
    # 可选：按(模型+数据集)精简为最优结果（grid_only_best）
    # 实验级精简：按(模型+数据集)分组，若组内任一实验声明 keep_only_best，则使用该组第一个声明的 summary 配置作为准则精简该组
    if results_summary:
        from collections import defaultdict
        group_map = defaultdict(list)
        for res in results_summary:
            group_map[(res.get('model'), res.get('dataset'))].append(res)
        new_results = []
        for key, items in group_map.items():
            # 查找组内声明的 summary 配置
            cfg_items = [it for it in items if isinstance(it.get('summary_cfg'), dict) and it['summary_cfg'].get('keep_only_best', False)]
            if not cfg_items:
                new_results.extend(items)
                continue
            base_cfg = cfg_items[0]['summary_cfg']
            metric = str(base_cfg.get('metric', '')).strip()
            mode = str(base_cfg.get('mode', 'max')).lower()
            split = str(base_cfg.get('split', 'test')).lower()
            if not metric:
                raise RuntimeError(f"summary 精简: 未提供 metric（组 {key}）")
            def _numeric(v):
                return isinstance(v, (int, float))
            def score(it):
                src = (it.get('eval_results') or {}) if split == 'test' else (it.get('summary_eval_results') or {})
                if not _numeric(src.get(metric)):
                    raise RuntimeError(f"summary 精简: 实验 {it.get('name')} 缺少指定指标或非数值 metric={metric}, split={split}")
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
            print(f"已按实验级 summary 精简：每个(模型+数据集)仅保留最优，共{len(new_results)}项（原{len(results_summary)}）")
        results_summary = new_results

    # 4. 生成总结报告
    print("步骤 4: 生成实验总结报告")
    print("=" * 80)
    print("实验对比总结")
    print("=" * 80)
    
    if results_summary:
        # 按数据集分组
        dataset_groups = {}
        for result in results_summary:
            dataset = result['dataset']
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(result)
        
        # 准备导出数据
        export_data = {}
        
        # 为每个数据集生成单独的对比表
        for dataset, results in dataset_groups.items():
            print(f"\n数据集: {dataset}")
            print("-" * 60)
            
            # 动态获取所有可用的评估指标
            all_metrics = set()
            for result in results:
                if 'eval_results' in result and result['eval_results']:
                    all_metrics.update(result['eval_results'].keys())
            
            # 排序指标名称，保证一致的显示顺序
            sorted_metrics = sorted(all_metrics)
            
            # 动态生成表头
            header = f"{'实验名称':<25} {'模型':<8} {'参数量':<10} {'轮数':<6} {'val_loss':<10}"
            for metric in sorted_metrics:
                header += f" {metric:<10}"
            print(header)
            print("-" * len(header))
            
            # 结果行
            for result in results:
                params = result.get('parameters', 0) or 0
                epochs = result.get('epochs', 0) or 0  
                val_loss = result.get('val_loss', 0) or 0
                
                row = f"{result['name']:<25} {result['model']:<8} {params:<10,} " \
                      f"{epochs:<6} {val_loss:<10.4f}"
                
                # 动态添加所有评估指标
                for metric in sorted_metrics:
                    value = result['eval_results'].get(metric, 0) if result.get('eval_results') else 0
                    if isinstance(value, (int, float)):
                        row += f" {value:<10.4f}"
                    else:
                        row += f" {'N/A':<10}"
                
                print(row)
            
            # 该数据集的最佳结果
            if len(results) > 1:
                print(f"\n{dataset}数据集最佳结果:")
                
                # 为每个指标找出最佳结果（排除test_samples等非性能指标）
                performance_metrics = [m for m in sorted_metrics 
                                     if m not in ['test_samples', 'train_test_gap']]
                
                for metric in performance_metrics:
                    # 找到该指标的最高值
                    best_result = max(results, 
                                    key=lambda x: x['eval_results'].get(metric, 0) 
                                    if x.get('eval_results') else 0)
                    best_value = best_result['eval_results'].get(metric, 0) if best_result.get('eval_results') else 0
                    
                    if isinstance(best_value, (int, float)) and best_value > 0:
                        print(f"   最高 {metric}: {best_result['name']} ({metric}: {best_value:.4f})")
            
            # 准备该数据集的导出数据
            export_rows = []
            for result in results:
                row = {
                    '实验名称': result['name'],
                    '模型': result['model'],
                    '训练样本数': result.get('train_samples', 0) or 0,
                    '测试样本数': result.get('test_samples', 0) or 0,
                    '特征维度': result.get('feature_dim', 0) or 0,
                    '类别数量': result.get('num_classes', 0) or 0,
                    '序列长度': result.get('sequence_length', 0) or 0,
                    '参数量': result.get('parameters', 0) or 0,
                    '轮数': result.get('epochs', 0) or 0,
                    'val_loss': result.get('val_loss', 0) or 0
                }
                # 添加评估指标
                for metric in sorted_metrics:
                    value = result['eval_results'].get(metric, 0) if result.get('eval_results') else 0
                    row[metric] = value if isinstance(value, (int, float)) else 0
                export_rows.append(row)
            
            export_data[dataset] = export_rows
        
        # 导出到Excel文件
        try:
            import pandas as pd
            from datetime import datetime
            
            # 使用实验运行目录（而不是results根目录）
            results_dir = result_manager.run_dir
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 按数据集分类的Excel文件
            dataset_excel_filename = f"results_by_dataset_{timestamp}.xlsx"
            dataset_excel_path = os.path.join(results_dir, dataset_excel_filename)
            
            with pd.ExcelWriter(dataset_excel_path, engine='openpyxl') as writer:
                for dataset, rows in export_data.items():
                    if rows:
                        df = pd.DataFrame(rows)
                        # 清理sheet名称（Excel sheet名称限制）
                        clean_dataset_name = dataset.replace('/', '_').replace('\\', '_')
                        sheet_name = clean_dataset_name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 2. 按模型分类的Excel文件
            model_excel_filename = f"results_by_model_{timestamp}.xlsx"
            model_excel_path = os.path.join(results_dir, model_excel_filename)
            
            # 准备按模型分类的数据
            model_groups = {}
            for result in results_summary:
                model = result['model']
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append(result)
            
            with pd.ExcelWriter(model_excel_path, engine='openpyxl') as writer:
                for model, results in model_groups.items():
                    if results:
                        # 准备模型视图数据
                        model_rows = []
                        for result in results:
                            dataset_name = result['dataset']
                            
                            row = {
                                '数据集': dataset_name,
                                '实验名称': result['name'],
                                '训练样本数': result.get('train_samples', 0) or 0,
                                '测试样本数': result.get('test_samples', 0) or 0,
                                '特征维度': result.get('feature_dim', 0) or 0,
                                '类别数量': result.get('num_classes', 0) or 0,
                                '序列长度': result.get('sequence_length', 0) or 0,
                                '参数量': result.get('parameters', 0) or 0,
                                '训练轮数': result.get('epochs', 0) or 0,
                                'val_loss': result.get('val_loss', 0) or 0
                            }
                            
                            # 添加评估指标
                            if result.get('eval_results'):
                                for metric, value in result['eval_results'].items():
                                    if isinstance(value, (int, float)):
                                        row[metric] = value
                                    else:
                                        row[metric] = 0
                            
                            model_rows.append(row)
                        
                        # 创建模型的sheet
                        df_model = pd.DataFrame(model_rows)
                        sheet_name = model[:31]  # Excel sheet名称限制
                        df_model.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"\n实验结果已导出到两个Excel文件:")
            print(f"  按数据集分类: {dataset_excel_path}")
            print(f"  按模型分类: {model_excel_path}")
            
        except ImportError:
            print("\n警告: 无法导入pandas，跳过Excel导出功能")
            print("请安装pandas: pip install pandas openpyxl")
        except Exception as e:
            print(f"\n警告: 导出Excel文件时出错: {str(e)}")
    else:
        print("没有成功完成的实验")
    
    print()
    print("所有实验完成")
    print("=" * 80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="故障诊断基准测试框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                          # 使用默认配置
  python main.py configs/my_config.yaml   # 使用自定义配置
        """
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default='configs/default_experiment.yaml',
        help='配置文件路径 (默认: configs/default_experiment.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return 1
    
    try:
        run_experiments(args.config)
        return 0
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        logging.error("程序执行失败", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
