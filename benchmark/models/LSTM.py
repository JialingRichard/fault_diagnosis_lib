"""
LSTM模型实现
===========

纯净的LSTM模型定义，用于时序异常检测
包含两个版本：
- LSTM2seq: 序列到序列输出，适用于序列预测任务  
- LSTM2one: 序列到单值输出，适用于分类任务
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LSTM2seq(nn.Module):
    """
    LSTM序列到序列模型
    
    纯净的LSTM网络结构，专注于序列预测
    输出格式：(batch_size, sequence_length, output_dim)
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
        super(LSTM2seq, self).__init__()
        
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



class LSTM2one(nn.Module):
    """
    LSTM序列到单值分类模型
    
    适用于时间序列分类任务的LSTM网络
    输出格式：(batch_size, output_dim)
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_dim: Optional[int] = None,
                 pooling_method: str = "last"):
        """
        初始化LSTM分类模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            dropout: dropout概率
            output_dim: 输出维度（分类类别数）
            pooling_method: 池化方法 ("last", "mean", "max", "attention")
        """
        super(LSTM2one, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.pooling_method = pooling_method
        
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
        
        # 分类输出层
        self.classifier = nn.Linear(hidden_dim, self.output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
        
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_dim)
            
        Returns:
            输出张量，形状为 (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden_dim)
        
        # 应用dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # 序列池化：将序列维度压缩为单个特征向量
        if self.pooling_method == "last":
            # 使用序列最后一个时间步
            pooled = lstm_out[:, -1, :]  # (batch, hidden_dim)
        elif self.pooling_method == "mean":
            # 使用序列所有时间步的平均
            pooled = lstm_out.mean(dim=1)  # (batch, hidden_dim)
        elif self.pooling_method == "max":
            # 使用序列所有时间步的最大值
            pooled = lstm_out.max(dim=1)[0]  # (batch, hidden_dim)
        elif self.pooling_method == "attention":
            # 简单的注意力机制加权平均
            # 计算注意力权重
            attention_weights = torch.softmax(lstm_out.mean(dim=2), dim=1)  # (batch, seq_len)
            # 加权平均
            pooled = torch.sum(lstm_out * attention_weights.unsqueeze(2), dim=1)  # (batch, hidden_dim)
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")
        
        # 分类层
        output = self.classifier(pooled)  # (batch, output_dim)
        
        return output


# 向后兼容：LSTM = LSTM2seq  
LSTM = LSTM2seq
