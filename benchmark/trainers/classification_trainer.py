"""
分类训练器 (ClassificationTrainer)
=============================

专门处理序列到单标签分类任务的训练器
适配模型输出格式从 (batch, seq_len, num_classes) 到 (batch, num_classes)
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
    分类训练器
    
    专门用于序列到单标签分类任务
    自动适配模型输出格式：将 (batch, seq_len, num_classes) 转换为 (batch, num_classes)
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str,
                 X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, full_config: Dict[str, Any] = None):
        """
        初始化分类训练器
        
        Args:
            model: 待训练的模型
            config: 训练配置
            device: 训练设备
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            full_config: 完整配置（用于访问epochinfo_templates等）
        """
        # 调用父类初始化
        super().__init__(model, config, device, X_train, y_train, X_test, y_test, full_config)
        
        logger.info(f"分类训练器初始化完成")
    
    def _validate_model_output(self, output: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        验证模型输出格式，确保是分类格式
        
        Args:
            output: 模型输出
            labels: 标签张量
            
        Returns:
            output: 验证后的输出
            labels: 处理后的标签
            
        Raises:
            ValueError: 如果输出格式不正确
        """
        if output.dim() != 2:
            raise ValueError(
                f"分类任务要求模型输出 (batch_size, num_classes) 格式，"
                f"但收到 {output.shape}。"
                f"请使用 CNN2one/LSTM2one 等分类模型，而不是 CNN2seq/LSTM2seq。"
            )
        
        # 确保标签格式正确
        if labels.dim() > 1:
            labels = labels.flatten()
            
        return output, labels
    
    def _compute_loss(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算损失，验证输出格式
        
        Args:
            output: 模型输出
            labels: 真实标签
            
        Returns:
            loss: 损失值
        """
        # 验证输出格式
        validated_output, validated_labels = self._validate_model_output(output, labels)
        
        # 计算交叉熵损失
        loss = self.criterion(validated_output, validated_labels)
        
        return loss
    
    def _compute_accuracy(self, output: torch.Tensor, labels: torch.Tensor) -> float:
        """
        计算准确率，验证输出格式
        
        Args:
            output: 模型输出
            labels: 真实标签
            
        Returns:
            accuracy: 准确率
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
        训练一个epoch，重写以使用适配的损失计算
        
        Returns:
            avg_train_loss: 平均训练损失
            train_accuracy: 训练准确率
        """
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            
            # 计算损失（自动适配输出格式）
            loss = self._compute_loss(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_accuracy += self._compute_accuracy(output, target)
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        train_accuracy = total_accuracy / num_batches
        
        return avg_train_loss, train_accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        验证模型，重写以使用适配的损失计算
        
        Returns:
            avg_val_loss: 平均验证损失
            val_accuracy: 验证准确率
        """
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        
        with torch.no_grad():
            # 按批次验证
            batch_size = self.batch_size
            for i in range(0, len(self.X_val), batch_size):
                end_idx = min(i + batch_size, len(self.X_val))
                data = self.X_val[i:end_idx]
                target = self.y_val[i:end_idx]
                
                # 前向传播
                output = self.model(data)
                
                # 计算损失和准确率（自动适配输出格式）
                loss = self._compute_loss(output, target)
                accuracy = self._compute_accuracy(output, target)
                
                # 统计
                batch_size_actual = end_idx - i
                total_loss += loss.item() * batch_size_actual
                total_accuracy += accuracy * batch_size_actual
                num_samples += batch_size_actual
        
        avg_val_loss = total_loss / num_samples
        val_accuracy = total_accuracy / num_samples
        
        return avg_val_loss, val_accuracy
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        进行预测，自动适配输出格式
        
        Args:
            X: 输入数据
            
        Returns:
            predictions: 预测结果 (n_samples,)
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            batch_size = self.batch_size
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                batch_data = X[i:end_idx]
                
                # 前向传播
                output = self.model(batch_data)
                
                # 验证输出格式
                validated_output, _ = self._validate_model_output(output, torch.zeros(output.shape[0]))
                
                # 获取预测类别
                batch_predictions = torch.argmax(validated_output, dim=1)
                predictions.append(batch_predictions.cpu().numpy())
        
        return np.concatenate(predictions)