"""
CNN模型实现
==========

用于时序异常检测的1D卷积神经网络
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class CNN(nn.Module):
    """
    1D CNN异常检测模型
    
    适用于时间序列数据的卷积神经网络
    """
    
    def __init__(self, 
                 input_dim: int,
                 num_filters: int = 64,
                 filter_sizes: list = None,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 output_dim: Optional[int] = None):
        """
        初始化CNN模型
        
        Args:
            input_dim: 输入特征维度
            num_filters: 卷积核数量
            filter_sizes: 卷积核大小列表
            num_layers: 卷积层数
            dropout: dropout概率
            output_dim: 输出维度，如果为None则与input_dim相同
        """
        super(CNN, self).__init__()
        
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
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_dim)
            
        Returns:
            输出张量，形状为 (batch_size, sequence_length, output_dim)
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
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        预测函数
        
        Args:
            x: 输入数据
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            output = self.forward(x)
            return output.cpu().numpy()



    