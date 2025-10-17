"""
Result Manager (ResultManager)
==============================

Manages run directories, versioning, live logs, and checkpoints.
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
    """Result manager for all outputs"""
    
    def __init__(self, config_file: str, results_base_dir: str = None):
        """Initialize result manager.
        
        Args:
            config_file: config file path
            results_base_dir: base results directory
        """
        if results_base_dir is None:
            results_base_dir = Path(__file__).parent.parent / "results"
        
        self.results_base_dir = Path(results_base_dir)
        self.config_name = Path(config_file).stem  # 去掉.yaml后缀
        
        # Create versioned run directory
        self.run_dir = self._create_run_directory()
        
        # Log files
        self.log_file = self.run_dir / "run.log"
        self.error_log_file = self.run_dir / "error.log"
        self._setup_logging()
        
        # Save config snapshot
        self._save_config_snapshot(config_file)
        
        print(f"Results will be saved to: {self.run_dir}")
    
    def _create_run_directory(self) -> Path:
        """Create run directory with auto-incremented version"""
        config_dir = self.results_base_dir / self.config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover existing versions
        existing_versions = []
        for dir_path in config_dir.iterdir():
            if dir_path.is_dir() and dir_path.name.startswith('v'):
                try:
                    version_num = int(dir_path.name.split('_')[0][1:])  # get number after 'v'
                    existing_versions.append(version_num)
                except (ValueError, IndexError):
                    continue
        
        # Generate next version
        next_version = max(existing_versions, default=-1) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"v{next_version}_{timestamp}"
        
        run_dir = config_dir / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        return run_dir
    
    def _setup_logging(self):
        """Setup live logging with multiple handlers

        Goals:
        - run.log: capture everything printed to stdout (print and console logging)
        - debug.log: superset of run.log (also includes DEBUG-level logs); to avoid duplicate INFO+ lines,
          the debug handler only writes records below INFO (i.e., DEBUG). INFO+ will enter debug.log via TeeStream.
        - error.log: capture ERROR+ records
        """
        # 1) Capture warnings to logging
        logging.captureWarnings(True)

        # Paths used by handlers and tee
        debug_log_path = self.run_dir / 'debug.log'

        # 2) Redirect print to run.log and debug.log (tee)
        class TeeStream:
            def __init__(self, log_files):
                self.terminal = sys.stdout
                # Open all log files in append mode
                self._files = [open(str(p), 'a', encoding='utf-8') for p in log_files]

            def write(self, message):
                self.terminal.write(message)
                for f in self._files:
                    f.write(message)
                    f.flush()

            def flush(self):
                self.terminal.flush()
                for f in self._files:
                    f.flush()

            def close(self):
                for f in self._files:
                    try:
                        f.close()
                    except Exception:
                        pass

        # Replace stdout with tee stream (run.log + debug.log)
        sys.stdout = TeeStream([self.log_file, debug_log_path])

        # Additionally, tee stderr to debug.log only (and keep terminal stderr)
        class TeeErr:
            def __init__(self, log_file_path):
                self.terminal = sys.__stderr__ if hasattr(sys, '__stderr__') else sys.stderr
                self.log_file = open(str(log_file_path), 'a', encoding='utf-8')
            def write(self, message):
                try:
                    self.terminal.write(message)
                except Exception:
                    pass
                try:
                    self.log_file.write(message)
                    self.log_file.flush()
                except Exception:
                    pass
            def flush(self):
                try:
                    self.terminal.flush()
                except Exception:
                    pass
                try:
                    self.log_file.flush()
                except Exception:
                    pass
            def close(self):
                try:
                    self.log_file.close()
                except Exception:
                    pass

        sys.stderr = TeeErr(debug_log_path)

        # 3) Configure root logger (DEBUG), handlers control output levels
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Avoid adding handler multiple times (during re-initialization)
        existing = {type(h).__name__ + getattr(h, 'baseFilename', '') for h in root_logger.handlers}

        # Console output (INFO+) to current stdout (tee'd to run.log)
        has_stdout_handler = any(
            isinstance(h, logging.StreamHandler) and getattr(h, 'stream', None) is sys.stdout
            for h in root_logger.handlers
        )
        if not has_stdout_handler:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            root_logger.addHandler(console_handler)

        # Debug details: write to debug.log, but avoid duplicating INFO+ (already tee'd to debug.log)
        debug_handler = logging.FileHandler(debug_log_path, mode='a', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        # filter matplotlib.font_manager findfont noise and filter out INFO+
        class _MatplotlibFindfontFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                try:
                    if str(record.name).startswith('matplotlib.font_manager') and 'findfont' in str(record.getMessage()):
                        return False
                except Exception:
                    pass
                return True
        class _BelowInfoFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                # Only keep records with level below INFO (i.e., DEBUG)
                try:
                    return int(record.levelno) < int(logging.INFO)
                except Exception:
                    return True
        debug_handler.addFilter(_MatplotlibFindfontFilter())
        debug_handler.addFilter(_BelowInfoFilter())
        key_debug = type(debug_handler).__name__ + str(debug_log_path)
        if key_debug not in existing:
            root_logger.addHandler(debug_handler)

        # Errors: write to error.log (ERROR+)
        error_log_path = self.run_dir / 'error.log'
        error_handler = logging.FileHandler(error_log_path, mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        key_error = type(error_handler).__name__ + str(error_log_path)
        if key_error not in existing:
            root_logger.addHandler(error_handler)

        # Reduce matplotlib.font_manager noise
        try:
            logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        except Exception:
            pass
    
    def _save_config_snapshot(self, config_file: str):
        """Save a copy of the config file"""
        config_snapshot = self.run_dir / "config.yaml"
        shutil.copy2(config_file, config_snapshot)
    
    def create_experiment_dir(self, experiment_name: str) -> Path:
        """Create per-experiment directories"""
        exp_dir = self.run_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)

        # create subdirectories
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        
        return exp_dir
    
    def save_checkpoint(self, experiment_name: str, model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer, epoch: int, 
                       val_loss: float, metrics: Dict[str, float] = None):
        """Save checkpoint"""
        exp_dir = self.create_experiment_dir(experiment_name)
        checkpoint_dir = exp_dir / "checkpoints"
        
        # Filename contains epoch and loss info
        checkpoint_name = f"epoch_{epoch}_loss_{val_loss:.4f}.pth"
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics or {}
        }
        
        torch.save(checkpoint, checkpoint_path)

        # Do not insert extra lines in console; detailed info written to debug.log (controlled by global logging)
        logging.getLogger(__name__).debug(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path
    
    def get_experiment_plot_dir(self, experiment_name: str) -> Path:
        """Get plots dir for experiment"""
        exp_dir = self.create_experiment_dir(experiment_name)
        return exp_dir / "plots"
    
    def log_experiment_error(self, experiment_info: str, error_message: str):
        """Record experiment error into error.log"""
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(experiment_info)
                f.write(f"Experiment failed: {error_message}\n\n")
                f.flush()  # flush log file immediately
        except Exception as e:
            print(f"Failed to write error.log: {e}")

    def log_experiment_timing(self, name: str, start_ts: float, end_ts: float, extra: Dict[str, Any] | None = None):
        """Record experiment timing into CSV (timings.csv)"""
        import time, csv
        timings_csv = self.run_dir / 'timings.csv'
        exists = timings_csv.exists()
        duration = max(0.0, end_ts - start_ts)
        row = {
            'experiment': name,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_ts)),
            'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_ts)),
            'duration_sec': f"{duration:.3f}"
        }
        if extra:
            # Flatten extra fields (e.g., memory, samples)
            for k, v in extra.items():
                row[str(k)] = v
        try:
            with open(timings_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to write timings.csv: {e}")
    
    def cleanup(self):
        """清理资源"""
        # Close tee files if present and restore stdout/stderr
        if hasattr(sys.stdout, 'close') and hasattr(sys.stdout, 'terminal'):
            try:
                sys.stdout.close()
            except Exception:
                pass
            sys.stdout = sys.stdout.terminal  # restore original stdout
        if hasattr(sys, '__stderr__'):
            try:
                if hasattr(sys.stderr, 'close'):
                    sys.stderr.close()
            except Exception:
                pass
            sys.stderr = sys.__stderr__
