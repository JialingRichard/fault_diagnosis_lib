"""
结果管理器 (ResultManager)
==========================

负责目录管理、版本号生成、实时日志和checkpoint保存

Linus准则：简洁高效，如无必要勿增实体
"""

import os
import sys
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import torch


class ResultManager:
    """结果管理器 - 处理所有输出保存"""
    
    def __init__(self, config_file: str, results_base_dir: str = None):
        """
        初始化结果管理器
        
        Args:
            config_file: 配置文件路径
            results_base_dir: 结果基础目录
        """
        if results_base_dir is None:
            results_base_dir = Path(__file__).parent.parent / "results"
        
        self.results_base_dir = Path(results_base_dir)
        self.config_name = Path(config_file).stem  # 去掉.yaml后缀
        
        # 创建版本目录
        self.run_dir = self._create_run_directory()
        
        # 设置日志文件
        self.log_file = self.run_dir / "run.log"
        self.error_log_file = self.run_dir / "error.log"
        self._setup_logging()
        
        # 保存配置快照
        self._save_config_snapshot(config_file)
        
        print(f"结果将保存到: {self.run_dir}")
    
    def _create_run_directory(self) -> Path:
        """创建运行目录，自动生成版本号"""
        config_dir = self.results_base_dir / self.config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找已有版本号
        existing_versions = []
        for dir_path in config_dir.iterdir():
            if dir_path.is_dir() and dir_path.name.startswith('v'):
                try:
                    version_num = int(dir_path.name.split('_')[0][1:])  # 提取v后面的数字
                    existing_versions.append(version_num)
                except (ValueError, IndexError):
                    continue
        
        # 生成新版本号
        next_version = max(existing_versions, default=-1) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"v{next_version}_{timestamp}"
        
        run_dir = config_dir / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        return run_dir
    
    def _setup_logging(self):
        """设置实时日志记录"""
        # 创建自定义处理器，同时输出到控制台和文件
        class TeeHandler(logging.Handler):
            def __init__(self, log_file):
                super().__init__()
                self.console = logging.StreamHandler(sys.stdout)
                self.file = logging.FileHandler(log_file, mode='a', encoding='utf-8')
                
            def emit(self, record):
                self.console.emit(record)
                self.file.emit(record)
                self.file.flush()  # 实时写入
        
        # 重定向print输出
        class TeeStream:
            def __init__(self, log_file):
                self.terminal = sys.stdout
                self.log_file = open(log_file, 'a', encoding='utf-8')
                
            def write(self, message):
                self.terminal.write(message)
                self.log_file.write(message)
                self.log_file.flush()  # 实时写入
                
            def flush(self):
                self.terminal.flush()
                self.log_file.flush()
        
        # 替换stdout以捕获所有print输出
        sys.stdout = TeeStream(self.log_file)
    
    def _save_config_snapshot(self, config_file: str):
        """保存配置文件快照"""
        config_snapshot = self.run_dir / "config.yaml"
        shutil.copy2(config_file, config_snapshot)
    
    def create_experiment_dir(self, experiment_name: str) -> Path:
        """为单个实验创建目录"""
        exp_dir = self.run_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        
        return exp_dir
    
    def save_checkpoint(self, experiment_name: str, model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer, epoch: int, 
                       val_loss: float, metrics: Dict[str, float] = None,
                       logging_level: str = 'normal'):
        """保存checkpoint"""
        exp_dir = self.create_experiment_dir(experiment_name)
        checkpoint_dir = exp_dir / "checkpoints"
        
        # 文件名包含epoch和loss信息
        checkpoint_name = f"epoch_{epoch}_loss_{val_loss:.4f}.pth"
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        # 保存checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # 根据日志等级决定是否显示
        if logging_level in ['normal', 'verbose']:
            print(f"Checkpoint保存: {checkpoint_name}")
        
        return checkpoint_path
    
    def get_experiment_plot_dir(self, experiment_name: str) -> Path:
        """获取实验的plots目录"""
        exp_dir = self.create_experiment_dir(experiment_name)
        return exp_dir / "plots"
    
    def log_experiment_error(self, experiment_info: str, error_message: str):
        """记录实验错误到error.log文件"""
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(experiment_info)
                f.write(f"实验失败: {error_message}\n\n")
                f.flush()  # 实时写入
        except Exception as e:
            print(f"写入错误日志失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        if hasattr(sys.stdout, 'log_file'):
            sys.stdout.log_file.close()
            sys.stdout = sys.stdout.terminal  # 恢复原始stdout