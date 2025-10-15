"""
CNN models
==========

1D CNNs for time series tasks:
- CNN2seq: sequence-to-sequence output
- CNN2one: sequence-to-one classification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class CNN2seq(nn.Module):
    """
    1D CNN sequence-to-sequence model.
    Output: (batch_size, sequence_length, output_dim)
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_filters: int = 64,
                 filter_sizes: list = None,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 output_dim: Optional[int] = None):
        """
        Initialize CNN model.
        
        Args:
            input_dim: input feature dimension
            num_filters: number of filters
            filter_sizes: list of kernel sizes
            num_layers: number of conv layers
            dropout: dropout prob
            output_dim: output dimension (defaults to input_dim)
        """
        super(CNN2seq, self).__init__()
        
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes if filter_sizes else [3, 5, 7]
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # 第一层：多尺度卷积
        self.conv_layers = nn.ModuleList()
        for filter_size in self.filter_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(input_dim, num_filters, kernel_size=filter_size, padding=filter_size//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_block)
        
        # 中间卷积层
        conv_input_dim = num_filters * len(self.filter_sizes)
        self.middle_layers = nn.ModuleList()
        
        # 确保至少有一个中间层来处理维度转换
        actual_middle_layers = max(1, num_layers - 1)
        
        for i in range(actual_middle_layers):
            layer = nn.Sequential(
                nn.Conv1d(conv_input_dim, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.middle_layers.append(layer)
            conv_input_dim = num_filters
        
        # 输出层
        self.output_layer = nn.Conv1d(num_filters, self.output_dim, kernel_size=1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            (batch_size, sequence_length, output_dim)
        """
        # CNN期望输入格式为 (batch, channels, length)
        # 需要转换 (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)
            conv_outputs.append(conv_out)
        
        # 拼接多尺度特征
        x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes), seq_len)
        
        # 中间卷积层
        for layer in self.middle_layers:
            x = layer(x)
        
        # 输出层
        x = self.output_layer(x)
        
        # 转换回 (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        return x


class CNN2one(nn.Module):
    """
    1D CNN sequence-to-one classification model.
    Output: (batch_size, output_dim)
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_filters: int = 64,
                 filter_sizes: list = None,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 output_dim: Optional[int] = None,
                 pooling_method: str = 'last',
                 **kwargs):
        """
        Initialize CNN classifier.
        
        Args:
            input_dim: input feature dimension
            num_filters: number of filters
            filter_sizes: list of kernel sizes
            num_layers: number of conv layers
            dropout: dropout prob
            output_dim: number of classes
            pooling_method: 'last'|'mean'|'max'|'attention'
        """
        super(CNN2one, self).__init__()
        
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes if filter_sizes else [3, 5, 7]
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.pooling_method = pooling_method
        
        # 第一层：多尺度卷积
        self.conv_layers = nn.ModuleList()
        for filter_size in self.filter_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(input_dim, num_filters, kernel_size=filter_size, padding=filter_size//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_block)
        
        # 中间卷积层
        conv_input_dim = num_filters * len(self.filter_sizes)
        self.middle_layers = nn.ModuleList()
        
        # 确保至少有一个中间层来处理维度转换
        actual_middle_layers = max(1, num_layers - 1)
        
        for i in range(actual_middle_layers):
            layer = nn.Sequential(
                nn.Conv1d(conv_input_dim, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.middle_layers.append(layer)
            conv_input_dim = num_filters
        
        # 特征提取层（保持序列格式）
        self.feature_layer = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        
        # 分类输出层（全连接）
        self.classifier = nn.Linear(num_filters, self.output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, sequence_length, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        # CNN期望输入格式为 (batch, channels, length)
        # 需要转换 (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)
            conv_outputs.append(conv_out)
        
        # 拼接多尺度特征
        x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes), seq_len)
        
        # 中间卷积层
        for layer in self.middle_layers:
            x = layer(x)
        
        # 特征提取
        x = self.feature_layer(x)  # (batch, num_filters, seq_len)
        
        # 序列池化：将序列维度压缩为单个特征向量
        if self.pooling_method == 'last':
            # 使用序列最后一个时间步
            pooled = x[:, :, -1]  # (batch, num_filters)
        elif self.pooling_method == 'mean':
            # 使用序列所有时间步的平均
            pooled = x.mean(dim=2)  # (batch, num_filters)
        elif self.pooling_method == 'max':
            # 使用序列所有时间步的最大值
            pooled = x.max(dim=2)[0]  # (batch, num_filters)
        elif self.pooling_method == 'attention':
            # 简单的注意力机制加权平均
            # 计算注意力权重
            attention_weights = torch.softmax(x.mean(dim=1), dim=1)  # (batch, seq_len)
            # 加权平均
            pooled = torch.sum(x * attention_weights.unsqueeze(1), dim=2)  # (batch, num_filters)
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")
        
        # 分类层
        output = self.classifier(pooled)  # (batch, output_dim)
        
        return output


# 注意：请在配置中显式指定类名（如 CNN2one 或 CNN2seq），不再提供默认别名。
