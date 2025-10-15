"""
Classification Trainer (ClassificationTrainer)
=============================================

Trainer specialized for sequence-to-one classification tasks.
Adapts model outputs from (batch, seq_len, num_classes) to (batch, num_classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import numpy as np
import logging

from .supervised_trainer import SupervisedTrainer

logger = logging.getLogger(__name__)


class ClassificationTrainer(SupervisedTrainer):
    """
    Classification trainer for sequence-to-one tasks.
    Automatically adapts outputs from (batch, seq_len, num_classes) to (batch, num_classes).
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str,
                 X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, full_config: Dict[str, Any] = None):
        """
        Initialize classification trainer.
        
        Args:
            model: model to train
            config: training config
            device: training device
            X_train, y_train: training data
            X_test, y_test: test data
            full_config: full config (for templates, etc.)
        """
        # 调用父类初始化
        super().__init__(model, config, device, X_train, y_train, X_test, y_test, full_config)
        
        logger.info("Classification trainer initialized")
    
    def _validate_model_output(self, output: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate model output format for classification.
        
        Args:
            output: model output
            labels: label tensor
        
        Returns:
            output: validated output
            labels: flattened labels if needed
        
        Raises:
            ValueError if output shape is invalid
        """
        if output.dim() != 2:
            raise ValueError(
                f"Classification requires output shape (batch_size, num_classes), "
                f"but got {output.shape}. "
                f"Use CNN2one/LSTM2one instead of CNN2seq/LSTM2seq."
            )
        
        # 确保标签格式正确
        if labels.dim() > 1:
            labels = labels.flatten()
            
        return output, labels
    
    def _compute_loss(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss with format validation.
        """
        # 验证输出格式
        validated_output, validated_labels = self._validate_model_output(output, labels)
        
        # 计算交叉熵损失
        loss = self.criterion(validated_output, validated_labels)
        
        return loss
    
    def _compute_accuracy(self, output: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute accuracy with format validation.
        """
        # 验证输出格式
        validated_output, validated_labels = self._validate_model_output(output, labels)
        
        # 计算预测类别
        predictions = torch.argmax(validated_output, dim=1)
        
        # 计算准确率
        correct = (predictions == validated_labels).float()
        accuracy = correct.mean().item()
        
        return accuracy
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch using adapted loss.
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Zero grads
            self.optimizer.zero_grad()
            
            # Forward
            output = self.model(data)
            
            # Compute loss (with output validation)
            loss = self._compute_loss(output, target)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Stats
            total_loss += loss.item()
            total_accuracy += self._compute_accuracy(output, target)
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        train_accuracy = total_accuracy / num_batches
        
        return avg_train_loss, train_accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate using adapted loss.
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        
        with torch.no_grad():
            # Iterate by batch
            batch_size = self.batch_size
            for i in range(0, len(self.X_val), batch_size):
                end_idx = min(i + batch_size, len(self.X_val))
                data = self.X_val[i:end_idx]
                target = self.y_val[i:end_idx]
                
                # Forward
                output = self.model(data)
                
                # Compute loss/accuracy with output validation
                loss = self._compute_loss(output, target)
                accuracy = self._compute_accuracy(output, target)
                
                # Stats
                batch_size_actual = end_idx - i
                total_loss += loss.item() * batch_size_actual
                total_accuracy += accuracy * batch_size_actual
                num_samples += batch_size_actual
        
        avg_val_loss = total_loss / num_samples
        val_accuracy = total_accuracy / num_samples
        
        return avg_val_loss, val_accuracy
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict with output validation.
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            batch_size = self.batch_size
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                batch_data = X[i:end_idx]
                
                # Forward
                output = self.model(batch_data)
                
                # Validate output
                validated_output, _ = self._validate_model_output(output, torch.zeros(output.shape[0]))
                
                # Argmax to classes
                batch_predictions = torch.argmax(validated_output, dim=1)
                predictions.append(batch_predictions.cpu().numpy())
        
        return np.concatenate(predictions)
