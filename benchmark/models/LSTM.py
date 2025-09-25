"""
LSTM模型实现
===========

纯净的LSTM模型定义，用于时序异常检测
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LSTM(nn.Module):
    """
    LSTM异常检测模型
    
    纯净的LSTM网络结构，专注于模型定义
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_dim: Optional[int] = None):
        """
        初始化LSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            dropout: dropout概率
            output_dim: 输出维度，如果为None则与input_dim相同
        """
        super(LSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_dim)
            
        Returns:
            输出张量，形状为 (batch_size, sequence_length, output_dim)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 应用dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # 输出层
        output = self.output_layer(lstm_out)
        
        return output
    
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


class LSTM11(nn.Module):
    """
    LSTM异常检测模型 - 替代版本
    
    纯净的LSTM网络结构，专注于模型定义
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_dim: Optional[int] = None):
        """
        初始化LSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            dropout: dropout概率
            output_dim: 输出维度，如果为None则与input_dim相同
        """
        super(LSTM11, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_dim)
            
        Returns:
            输出张量，形状为 (batch_size, sequence_length, output_dim)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 应用dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # 输出层
        output = self.output_layer(lstm_out)
        
        return output
    
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
