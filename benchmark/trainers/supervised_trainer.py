"""
Supervised trainer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple
import numpy as np
import logging
import sys
import shutil
from pathlib import Path

# Add src to import path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from epochinfo_loader import EpochInfoLoader

logger = logging.getLogger(__name__)


class SupervisedTrainer:
    """
    Supervised trainer
    
    Standard supervised training flow.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str,
                 X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, full_config: Dict[str, Any] = None):
        """
        Initialize supervised trainer
        
        Args:
            model: model to train
            config: training configuration
            device: training device
            X_train, y_train: training data
            X_test, y_test: test data
            full_config: full config (for access to templates, etc.)
        """
        self.model = model
        self.config = config
        self.full_config = full_config or config
        self.device = device
        
        # Apply data subsampling if configured
        data_fraction = config.get('data_fraction', 1.0)
        if data_fraction < 1.0:
            train_size = int(len(X_train) * data_fraction)
            test_size = int(len(X_test) * data_fraction)
            
            # Random subsample (keep class distribution approximately)
            train_indices = np.random.choice(len(X_train), train_size, replace=False)
            test_indices = np.random.choice(len(X_test), test_size, replace=False)
            
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
            X_test = X_test[test_indices]
            y_test = y_test[test_indices]
            
            print(f"   Subset: {data_fraction:.1%} ({train_size:,} train + {test_size:,} test)")
        
        # Validation split (no implicit default; must be explicit)
        validation_split = config.get('validation_split', None)
        use_test_as_val = bool(config.get('use_test_as_val', False))
        if isinstance(validation_split, (int, float)) and validation_split > 0.0 and validation_split < 1.0:
            from sklearn.model_selection import train_test_split
            
            # Check if stratified split is possible (min 2 samples per class)
            y_flat = y_train.flatten()
            unique_classes, class_counts = np.unique(y_flat, return_counts=True)
            min_samples_per_class = class_counts.min()
            
            # Use stratified split if possible; otherwise random split
            if min_samples_per_class > 1:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train,
                    test_size=validation_split,
                    stratify=y_flat,
                    random_state=42
                )
                print(f"   Using stratified split for validation set")
            else:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train,
                    test_size=validation_split,
                    random_state=42
                )
                print(f"   Warning: Too few samples in at least one class (min={min_samples_per_class}); using random split for validation set")
            
            train_samples = len(X_train_split)
            val_samples = len(X_val)
            print(f"   Split: {train_samples:,} train + {val_samples:,} val + {len(X_test):,} test")
            
            X_train = X_train_split
            y_train = y_train_split
            
            # Set validation tensors (CPU-resident)
            self.X_val = torch.FloatTensor(X_val)
            self.y_val = torch.LongTensor(y_val.flatten())
        else:
            # No valid validation_split configured
            if use_test_as_val:
                print(f"   Warning: Using test set as validation (use_test_as_val=True, may cause information leakage)")
                self.X_val = torch.FloatTensor(X_test)
                self.y_val = torch.LongTensor(y_test.flatten())
            else:
                raise ValueError(
                    "No valid validation_split and not allowed to use test as validation."
                    " Set validation_split in (0,1) or set use_test_as_val=true at your own risk."
                )
        
        # Keep data on CPU; move per batch to device during training/validation/predict
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train.flatten())
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.LongTensor(y_test.flatten())
        
        
        self.batch_size = config.get('batch_size', 32)
        
        # Create DataLoader (enable pin_memory to speed up H2D copies)
        pin_mem = True if str(self.device).startswith('cuda') else False
        num_workers = int(self.config.get('num_workers', 0))
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=pin_mem,
            num_workers=num_workers
        )
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {'train_loss': [], 'val_loss': []}
        # Progress bar controls (bar100 only)
        self.use_progress_bar = bool(self.config.get('progress_bar', False))
        self._epoch_index = 0
        self._total_epochs = 0
        
        # Initialize epoch info loader
        self.epochinfo_loader = EpochInfoLoader()
        self.eval_loader = None  # set by external
        self.result_manager = None  # set by external
        self.experiment_name = None  # set by external
        self.best_checkpoint_path = None  # record best checkpoint
        # Cache predictions for this round to avoid recomputation
        self._cached_train_pred = None
        self._cached_val_pred = None
        self._cached_test_pred = None
        # Control per-epoch prediction caching via config
        self.compute_train_pred_each_epoch = bool(self.config.get('compute_train_pred_each_epoch', False))
        # If monitoring by metrics (not val_loss), val predictions are often needed; default True then
        default_val_pred = False if str(self.config.get('monitor', {}).get('metric', 'val_loss')).lower() == 'val_loss' else True
        self.compute_val_pred_each_epoch = bool(self.config.get('compute_val_pred_each_epoch', default_val_pred))
        self.compute_test_pred_each_epoch = bool(self.config.get('compute_test_pred_each_epoch', False))
        
        # Read global and monitor configuration (strict)
        gcfg = (self.full_config.get('global') or {}) if isinstance(self.full_config, dict) else {}
        self.checkpoint_policy = gcfg.get('checkpoint_policy', 'best')
        self.epochinfo_template = self.config.get('epochinfo', None)
        monitor_cfg = self.config.get('monitor', None)
        if not self.epochinfo_template or not monitor_cfg:
            raise ValueError("lack of epochinfo and monitor")
        required_monitor = {'metric', 'mode', 'split'}
        if not required_monitor.issubset(monitor_cfg.keys()):
            raise ValueError(f"monitor lacks required fields: {required_monitor}")
        self.monitor_metric = str(monitor_cfg['metric'])
        self.monitor_mode = str(monitor_cfg['mode']).lower()
        if self.monitor_mode not in {'min','max'}:
            raise ValueError("monitor.mode must be 'min' or 'max'")
        self.monitor_split = str(monitor_cfg['split']).lower()
        if self.monitor_split not in {'val','test'}:
            raise ValueError("monitor.split must be 'val' or 'test'")
        if self.monitor_split == 'test':
            print("   Warning: monitor.split='test' will choose best by test metrics (risk of leakage)")
        # show training split (for logging purpose; actual choice done in epochinfo_loader)
        self.epochinfo_split = str(self.config.get('epochinfo_split', 'val')).lower()
        if self.epochinfo_split == 'test':
            print("   Warning: epochinfo_split='test' will evaluate on test during training (risk of leakage)")
        # Load metric map from epochinfo template
        eval_templates = (self.full_config.get('evaluation_templates') or {})
        if self.epochinfo_template not in eval_templates:
            raise ValueError(f"epochinfo mapped template not found: {self.epochinfo_template}")
        tpl = eval_templates[self.epochinfo_template]
        if isinstance(tpl, dict) and 'metrics' not in tpl:
            metric_map = {k: v for k, v in tpl.items() if not str(k).startswith('_')}
        else:
            metric_map = tpl.get('metrics', {})
        if self.monitor_metric not in metric_map:
            raise ValueError(f"monitor.metric '{self.monitor_metric}' not found in template '{self.epochinfo_template}'")
        self.best_monitor_value = float('inf') if self.monitor_mode == 'min' else -float('inf')
        # Early stop configuration: default by val_loss; optional by monitor metric
        self.early_stop_use_monitor = bool(self.config.get('early_stop_use_monitor', False))
        

    
    def _create_optimizer(self):
        """Create optimizer"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('lr', 0.001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop
        
        Returns:
            Dict of training results
        """
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        
        print(f"   {epochs} epochs (patience:{patience})")
        
        for epoch in range(epochs):
            # expose epoch indices for progress bar descriptions
            self._epoch_index = epoch
            self._total_epochs = epochs
            # Train
            train_loss = self._train_epoch()
            
            # Validate
            val_loss = self._validate()
            
            # Generate predictions for this epoch (cache for epochinfo/monitor)
            try:
                self._cached_train_pred = self.predict(self.X_train, show_progress=False) if self.compute_train_pred_each_epoch else None
            except Exception:
                self._cached_train_pred = None
            try:
                # Validate and test are computed to avoid double calls
                self._cached_val_pred = self.predict(self.X_val, show_progress=False) if self.compute_val_pred_each_epoch else None
            except Exception:
                self._cached_val_pred = None
            try:
                self._cached_test_pred = self.predict(self.X_test, show_progress=False) if self.compute_test_pred_each_epoch else None
            except Exception:
                self._cached_test_pred = None

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Print epoch info by interval
            print_interval = self.config.get('print_interval', 10)
            
            # Compute val_loss improvement (for display; not necessarily for early stop)
            improvement = None
            val_loss_improved = False
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                val_loss_improved = True

            # Compute monitor value and save checkpoint by policy
            monitor_value = self._compute_monitor_value(val_loss)
            # Determine if monitor improved
            better = (monitor_value < self.best_monitor_value) if self.monitor_mode == 'min' else (monitor_value > self.best_monitor_value)

            # Early stop: adjust patience by monitor or val_loss depending on config
            if self.early_stop_use_monitor:
                if better:
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            else:
                if val_loss_improved:
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            # Save policy
            save_now = (self.checkpoint_policy == 'all') or (self.checkpoint_policy == 'best' and better)
            
            if save_now and self.result_manager and self.experiment_name:
                ckpt_path = self.result_manager.save_checkpoint(
                    self.experiment_name, self.model, self.optimizer, 
                    epoch + 1, val_loss  # filename still contains val_loss
                )
                try:
                    # if this epoch is better, update best reference
                    if better:
                        if self.checkpoint_policy == 'best' or self.checkpoint_policy == 'all':
                            best_path = ckpt_path.parent / 'best.pth'
                            shutil.copy2(ckpt_path, best_path)
                            self.best_checkpoint_path = best_path
                except Exception as e:
                    logging.getLogger(__name__).debug(f"Failed to maintain best.pth: {e}")
            
            # Print epoch info (modular system)
            if epoch % print_interval == 0 or epoch == epochs - 1:
                epoch_data = {
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'improvement': improvement,
                    'patience_counter': self.patience_counter,
                    'patience': patience,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    # Reuse cached predictions
                    'train_pred': self._cached_train_pred,
                    'test_pred': self._cached_test_pred,
                    # Provide method to compute accuracy, but do not compute actively
                    'trainer': self
                }
                
                # Get epoch info template
                epochinfo_template = self.config.get('epochinfo', 'default')
                # Set eval_loader for epochinfo if needed
                if self.eval_loader and not self.epochinfo_loader.eval_loader:
                    self.epochinfo_loader.eval_loader = self.eval_loader
                self.epochinfo_loader.print_epoch_info(self.full_config, epochinfo_template, epoch_data)

                # This round print completed
            
            # Early stop
            if self.patience_counter >= patience:
                print(f"   Early stop: no improvement for {patience} epochs @E{epoch+1}")
                break

        # As configured: use best checkpoint for final evaluation
        load_best = self.config.get('load_best_checkpoint_for_eval', True)
        selected_checkpoint = None
        if load_best and self.best_checkpoint_path is not None:
            try:
                checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                selected_checkpoint = str(self.best_checkpoint_path)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to load best checkpoint, using current model: {e}")

        # Generate predictions (show progress at final evaluation)
        train_pred = self.predict(self.X_train, show_progress=True)
        test_pred = self.predict(self.X_test, show_progress=True)
        
        results = {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1,
            'training_history': self.training_history,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'selected_checkpoint': selected_checkpoint,
            # Return actual data used for evaluation
            'actual_X_train': self.X_train.cpu().numpy(),
            'actual_y_train': self.y_train.cpu().numpy(),
            'actual_X_test': self.X_test.cpu().numpy(),
            'actual_y_test': self.y_test.cpu().numpy()
        }
        

        return results

    def _compute_monitor_value(self, current_val_loss: float) -> float:
        """compute current monitor value based on config"""
        # support explicit 'val_loss' as monitor metric (special case)
        if self.monitor_metric == 'val_loss':
            return float(current_val_loss)

        # select split data
        if self.monitor_split == 'val':
            X_t = self.X_val
            y_t = self.y_val
            y_pred_cached = self._cached_val_pred
        else:
            X_t = self.X_test
            y_t = self.y_test
            y_pred_cached = self._cached_test_pred

        # predict (reuse cache if available)
        y_t_pred = y_pred_cached if y_pred_cached is not None else self.predict(X_t)
        y_t_np = y_t.cpu().numpy() if hasattr(y_t, 'cpu') else y_t

        # Load evaluator function from eval_loader
        try:
            eval_templates = (self.full_config.get('evaluation_templates') or {})
            tpl = eval_templates[self.epochinfo_template]
            if isinstance(tpl, dict) and 'metrics' not in tpl:
                metric_map = {k: v for k, v in tpl.items() if not str(k).startswith('_')}
            else:
                metric_map = tpl.get('metrics', {})
            metric_cfg = metric_map[self.monitor_metric]

            if not self.eval_loader:
                from src.eval_loader import EvalLoader
                self.eval_loader = EvalLoader()
            evaluator_func = self.eval_loader._load_evaluator(self.monitor_metric, metric_cfg)

            # only use test channel to pass monitor split; train channel pass empty placeholders
            X_train_np = np.empty((0,))
            y_train_np = np.empty((0,))
            y_train_pred_np = np.empty((0,))
            value = evaluator_func(
                X_train_np, y_train_np, y_train_pred_np,
                None, y_t_np, y_t_pred
            )
            return float(value)
        except Exception as e:
            raise ValueError(f"calculate monitor metric failed: metric={self.monitor_metric}, split={self.monitor_split}, error: {e}")
    
    def _train_epoch(self) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        iterator = self.train_loader
        total_batches = max(1, len(self.train_loader))
        printed_pct = 0
        if self.use_progress_bar:
            try:
                sys.stderr.write('[Train] ' + ('#' * 100) + "\n")
                sys.stderr.write('[Train] ')
                sys.stderr.flush()
            except Exception:
                pass

        for bidx, (batch_X, batch_y) in enumerate(iterator, start=1):
            self.optimizer.zero_grad()

            # forward pass
            batch_X = batch_X.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            outputs = self.model(batch_X)

            # need to reshape if output is sequence
            if len(outputs.shape) == 3:  # (batch, seq, features)
                outputs = outputs.reshape(-1, outputs.shape[-1])
                batch_y = batch_y.repeat_interleave(outputs.shape[0] // batch_y.shape[0])

            loss = self.criterion(outputs, batch_y)

            # backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if self.use_progress_bar:
                # print one '-' for each new percent crossed (stderr only)
                try:
                    pct = int((bidx * 100) / total_batches)
                    while printed_pct < pct:
                        sys.stderr.write('-')
                        sys.stderr.flush()
                        printed_pct += 1
                    if bidx == total_batches:
                        sys.stderr.write('\n')
                        sys.stderr.flush()
                except Exception:
                    pass

        return total_loss / len(self.train_loader)
    
    def _validate(self) -> float:
        # validate model - use real validation set to avoid leakage
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        val_dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.batch_size,  
            shuffle=False,
            pin_memory=True if str(self.device).startswith('cuda') else False,
            num_workers=int(self.config.get('num_workers', 0))
        )
        
        with torch.no_grad():
            iterator = val_loader
            total_batches = max(1, len(val_loader))
            printed_pct = 0
            if self.use_progress_bar:
                try:
                    sys.stderr.write('[Val  ] ' + ('#' * 100) + "\n")
                    sys.stderr.write('[Val  ] ')
                    sys.stderr.flush()
                except Exception:
                    pass

            for bidx, (batch_X, batch_y) in enumerate(iterator, start=1):
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                outputs = self.model(batch_X)

                # need to reshape if output is sequence
                if len(outputs.shape) == 3:
                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    batch_y = batch_y.repeat_interleave(outputs.shape[0] // batch_y.shape[0])

                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                batch_count += 1
                if self.use_progress_bar:
                    try:
                        pct = int((bidx * 100) / total_batches)
                        while printed_pct < pct:
                            sys.stderr.write('-')
                            sys.stderr.flush()
                            printed_pct += 1
                        if bidx == total_batches:
                            sys.stderr.write('\n')
                            sys.stderr.flush()
                    except Exception:
                        pass
        
        return total_loss / batch_count if batch_count > 0 else 0.0
    
    def predict(self, X: torch.Tensor, show_progress: bool = True) -> np.ndarray:
        """Generate predictions - use batching to avoid OOM.

        Args:
            X: tensor dataset to predict
            show_progress: whether to show bar100 on stderr
        """
        self.model.eval()
        all_predictions = []

        # Create data loader for batch prediction
        dataset = torch.utils.data.TensorDataset(X)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True if str(self.device).startswith('cuda') else False,
            num_workers=int(self.config.get('num_workers', 0))
        )
        
        # Determine progress label
        label = 'Predict'
        try:
            if X.data_ptr() == self.X_test.data_ptr():
                label = 'Test'
            elif hasattr(self, 'X_val') and X.data_ptr() == self.X_val.data_ptr():
                label = 'Val'
            elif X.data_ptr() == self.X_train.data_ptr():
                label = 'Train'
        except Exception:
            label = 'Predict'

        total_batches = max(1, len(data_loader))
        printed_pct = 0
        if self.use_progress_bar and show_progress:
            try:
                sys.stderr.write(f'[{label:<5}] ' + ('#' * 100) + "\n")
                sys.stderr.write(f'[{label:<5}] ')
                sys.stderr.flush()
            except Exception:
                pass

        with torch.no_grad():
            for bidx, (batch_X,) in enumerate(data_loader, start=1):
                batch_X = batch_X.to(self.device, non_blocking=True)
                outputs = self.model(batch_X)

                # for sequence output, take last time step or average
                if len(outputs.shape) == 3:  # (batch, seq, features)
                    outputs = outputs[:, -1, :]  # take last time step

                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu().numpy())
                if self.use_progress_bar and show_progress:
                    try:
                        pct = int((bidx * 100) / total_batches)
                        while printed_pct < pct:
                            sys.stderr.write('-')
                            sys.stderr.flush()
                            printed_pct += 1
                        if bidx == total_batches:
                            sys.stderr.write('\n')
                            sys.stderr.flush()
                    except Exception:
                        pass
        
        return np.concatenate(all_predictions)
    
    def _calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy on given data
        
        Args:
            X: input data
            y: true labels

        Returns:
            accuracy
        """
        predictions = self.predict(X)
        # make sure y is numpy array, not tensor
        if hasattr(y, 'cpu'):
            y = y.cpu().numpy()
        return float(np.mean(predictions == y))
