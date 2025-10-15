"""
Epoch Info Loader

动态加载epoch信息打印函数的加载器
"""

import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, Callable
import inspect

logger = logging.getLogger(__name__)


class EpochInfoLoader:
    """
    Epoch信息加载器
    
    动态加载epoch信息打印函数并执行
    """
    
    def __init__(self, epochinfo_dir: Path = None, eval_loader=None):
        """初始化epoch信息加载器"""
        if epochinfo_dir is None:
            epochinfo_dir = Path(__file__).parent.parent / "epochinfos"
        
        self.epochinfo_dir = Path(epochinfo_dir)
        self.epochinfo_registry = {}
        self.eval_loader = eval_loader
        
        logger.debug(f"Epoch信息加载器初始化完成，目录: {self.epochinfo_dir}")
    
    def print_epoch_info(self, config: Dict[str, Any], epochinfo_template_name: str,
                        epoch_data: Dict[str, Any]) -> None:
        """
        打印epoch信息
        
        Args:
            config: 完整配置
            epochinfo_template_name: epoch信息模板名称
            epoch_data: epoch数据，包含所有需要的信息
        """
        # 新逻辑：epochinfo 直接引用 evaluation_templates 的模板名
        eval_templates = config.get('evaluation_templates', {})
        if epochinfo_template_name and epochinfo_template_name in eval_templates:
            # 基本信息 + 评估指标
            self._print_with_evaluation(epoch_data, epochinfo_template_name, config)
        else:
            # 未指定或模板不存在，打印基本信息
            if epochinfo_template_name:
                logger.warning(f"Epoch信息模板 '{epochinfo_template_name}' 未找到，使用默认打印")
            self._default_print(epoch_data)
    
    def _load_print_function(self, format_config: Dict[str, str]) -> Callable:
        """
        加载打印函数
        
        Args:
            format_config: 格式配置，包含file和function字段
            
        Returns:
            打印函数或None
        """
        if isinstance(format_config, str):
            # 简单字符串，使用默认函数名
            file_name = format_config
            function_name = 'print_epoch_info'
        else:
            # 字典格式，指定文件和函数
            file_name = format_config.get('file', 'default')
            function_name = format_config.get('function', 'print_epoch_info')
        
        # 构建缓存键
        cache_key = f"{file_name}.{function_name}"
        
        # 检查缓存
        if cache_key in self.epochinfo_registry:
            return self.epochinfo_registry[cache_key]
        
        try:
            # 构建文件路径
            file_path = self.epochinfo_dir / f"{file_name}.py"
            
            if not file_path.exists():
                logger.error(f"Epoch信息文件不存在: {file_path}")
                return None
            
            # 动态加载模块
            spec = importlib.util.spec_from_file_location(file_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 获取函数
            if hasattr(module, function_name):
                print_func = getattr(module, function_name)
                
                # 验证函数签名
                sig = inspect.signature(print_func)
                expected_params = ['epoch_data', 'config']
                actual_params = list(sig.parameters.keys())
                
                if len(actual_params) >= 1:  # 至少需要epoch_data参数
                    self.epochinfo_registry[cache_key] = print_func
                    logger.debug(f"Epoch信息函数 '{function_name}' 加载成功 (模块: {file_name})")
                    return print_func
                else:
                    logger.error(f"函数 '{function_name}' 参数不符合要求，需要至少1个参数: epoch_data")
                    return None
            else:
                logger.error(f"模块 '{file_name}' 中未找到函数 '{function_name}'")
                return None
                
        except Exception as e:
            logger.error(f"加载epoch信息函数失败: {file_name}.{function_name}, 错误: {str(e)}")
            return None
    
    def _default_print(self, epoch_data: Dict[str, Any]) -> None:
        """
        默认的epoch信息打印
        
        Args:
            epoch_data: epoch数据
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
            print(f"   E{epoch+1:3d}/{total_epochs}: {train_loss:.4f}→{val_loss:.4f} ×{patience_counter}/{patience}")
    
    def _print_with_evaluation(self, epoch_data: Dict[str, Any], eval_template_name: str, config: Dict[str, Any]) -> None:
        """
        打印基本epoch信息 + 评估指标
        
        Args:
            epoch_data: epoch数据
            epochinfo_template: epochinfo模板配置
            config: 完整配置
        """
        # 先获取基本信息字符串
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
            basic_info = f"   E{epoch+1:3d}/{total_epochs}: {train_loss:.4f}→{val_loss:.4f} ×{patience_counter}/{patience}"
        
        # 如果有eval_loader且配置了evaluation，计算评估指标
        metrics_str = ""
        if self.eval_loader and eval_template_name:
            trainer = epoch_data.get('trainer')
            if trainer:
                try:
                    # split: 训练期评估默认使用验证集，避免信息泄露
                    epochinfo_split = str(getattr(trainer, 'config', {}).get('epochinfo_split', 'val')).lower()
                    if epochinfo_split not in {'val', 'test'}:
                        epochinfo_split = 'val'

                    # 获取/缓存需要的预测
                    train_pred = epoch_data.get('train_pred')
                    if train_pred is None:
                        train_pred = trainer.predict(trainer.X_train)

                    if epochinfo_split == 'val':
                        # 选择验证集作为评估“test通道”
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

                    # 准备 numpy 标签
                    X_train = trainer.X_train
                    y_train = trainer.y_train
                    if hasattr(y_train, 'cpu'):
                        y_train = y_train.cpu().numpy()
                    if hasattr(sel_y, 'cpu'):
                        y_sel_np = sel_y.cpu().numpy()
                    else:
                        y_sel_np = sel_y

                    # 为评估器设置上下文（plots目录、epoch信息）
                    if hasattr(trainer, 'result_manager') and trainer.result_manager:
                        plots_dir = trainer.result_manager.get_experiment_plot_dir(trainer.experiment_name)
                        # 设置epoch上下文，图像会保存到epochinfo子目录
                        self.eval_loader.set_context(
                            plots_dir=plots_dir,
                            epoch_info=epoch_data  # 传递epoch信息，图像会保存到epochinfo/子目录
                        )
                    
                    # 临时禁用日志
                    original_level = logging.getLogger('src.eval_loader').level
                    logging.getLogger('src.eval_loader').setLevel(logging.WARNING)
                    
                    # 将选定 split 作为 evaluator 的“test通道”输入
                    eval_results = self.eval_loader.evaluate(
                        config, eval_template_name,
                        X_train, y_train, train_pred,
                        sel_X, y_sel_np, sel_pred
                    )
                    
                    # 恢复日志级别
                    logging.getLogger('src.eval_loader').setLevel(original_level)
                    
                    # 构建评估结果字符串，附带 split 标签
                    split_tag = f"split:{epochinfo_split}"
                    metrics_parts = [f"{k}={v:.3f}" for k, v in eval_results.items() if v is not None]
                    metrics_parts.append(split_tag)
                    metrics_str = " | " + " | ".join(metrics_parts)
                    
                except Exception as e:
                    logger.warning(f"计算评估指标失败: {e}")
        
        # 打印在一行
        print(f"{basic_info}{metrics_str}")
