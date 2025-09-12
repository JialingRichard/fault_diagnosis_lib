"""
LSTM AutoEncoder Model Implementation
=====================================

This module provides an LSTM-based autoencoder for time series anomaly detection.

Author: Fault Diagnosis Benchmark Team
Date: 2025-01-11
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Optional, Tuple
import logging
import os

from .base_model import BaseModel, DataMetadata

logger = logging.getLogger(__name__)


class LSTMAutoEncoder(nn.Module):
    """LSTM自编码器网络结构"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 dropout: float = 0.1, bidirectional: bool = False):
        super(LSTMAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # 编码器
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 解码器
        decoder_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.decoder = nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 输出层
        decoder_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_layer = nn.Linear(decoder_output_dim, input_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 编码
        encoded, (hidden, cell) = self.encoder(x)
        
        # 使用编码器的最后隐藏状态初始化解码器
        batch_size, seq_len, _ = x.size()
        
        # 创建解码器输入（可以是零向量或编码器输出）
        decoder_input = encoded
        
        # 解码
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        
        # 应用dropout
        decoded = self.dropout(decoded)
        
        # 输出重构
        reconstructed = self.output_layer(decoded)
        
        return reconstructed


class LSTMAutoEncoderModel(BaseModel):
    """
    LSTM AutoEncoder 异常检测模型
    
    使用LSTM自编码器进行时序异常检测，通过重构误差来判断异常。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LSTM AutoEncoder模型
        
        Args:
            config: 模型配置参数
        """
        super().__init__(config, "LSTM_AutoEncoder")
        
        # 默认配置
        self.default_config = {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'bidirectional': False,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'normalize': True,
            'sequence_length': 50,
            'device': 'auto'  # 'auto', 'cpu', 'cuda'
        }
        
        # 合并配置
        self.model_config = {**self.default_config, **config}
        
        # 设备配置
        if self.model_config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.model_config['device'])
            
        logger.info(f"使用设备: {self.device}")
        
        # 预处理器
        self.scaler = MinMaxScaler() if self.model_config['normalize'] else None
        
        # 训练相关
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.best_model_state = None
        
        logger.info(f"LSTM AutoEncoder配置: {self.model_config}")
    
    @property
    def requires_training_loop(self) -> bool:
        """LSTM AutoEncoder需要复杂训练循环"""
        return True
    
    @property
    def supports_online_learning(self) -> bool:
        """LSTM AutoEncoder支持在线学习（通过增量训练）"""
        return True
    
    def build_model(self, input_shape: tuple, metadata: Optional[DataMetadata] = None) -> None:
        """
        构建LSTM AutoEncoder模型
        
        Args:
            input_shape: 输入数据形状 (batch_size, sequence_length, feature_dim)
            metadata: 数据元数据
        """
        if len(input_shape) == 2:
            # 如果是2D数据，添加序列维度
            _, feature_dim = input_shape
            sequence_length = self.model_config['sequence_length']
        else:
            _, sequence_length, feature_dim = input_shape
            
        # 创建模型
        self.model = LSTMAutoEncoder(
            input_dim=feature_dim,
            hidden_dim=self.model_config['hidden_dim'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout'],
            bidirectional=self.model_config['bidirectional']
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model_config['learning_rate']
        )
        
        logger.info(f"LSTM AutoEncoder模型构建完成")
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        if metadata:
            logger.info(f"数据集信息: {metadata}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None, 
            metadata: Optional[DataMetadata] = None) -> None:
        """
        训练LSTM AutoEncoder模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签（用于选择正常样本）
            metadata: 数据元数据
        """
        if not self.model:
            self.build_model(X_train.shape, metadata)
        
        # 数据预处理
        X_processed = self._preprocess_data(X_train, fit_scaler=True)
        
        # 转换为序列数据
        X_sequences = self._create_sequences(X_processed)
        
        # 对于自编码器，通常只使用正常数据训练
        if y_train is not None:
            y_sequences = self._create_sequences(y_train.reshape(-1, 1))[:, 0, 0]
            normal_mask = (y_sequences == 0)
            if np.any(normal_mask):
                X_sequences = X_sequences[normal_mask]
                logger.info(f"使用 {np.sum(normal_mask)} 个正常序列训练模型")
        
        # 创建数据加载器
        train_loader = self._create_dataloader(X_sequences, X_sequences)
        
        # 训练模型
        self._train_model(train_loader)
        
        self.is_trained = True
        logger.info("LSTM AutoEncoder训练完成")
    
    def _train_model(self, train_loader: DataLoader) -> None:
        """训练模型的核心逻辑"""
        best_loss = float('inf')
        patience_counter = 0
        epochs = self.model_config['epochs']
        patience = self.model_config['patience']
        
        logger.info(f"开始训练，设备: {self.device}")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                reconstructed = self.model(batch_X)
                loss = self.criterion(reconstructed, batch_y)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # print specific results each epoch
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # 早停策略
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_model_state = {
                    'model_state_dict': self.model.state_dict().copy(),
                    'optimizer_state_dict': self.optimizer.state_dict().copy(),
                    'loss': best_loss,
                    'epoch': epoch
                }
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"早停在第 {epoch+1} 轮，最佳损失: {best_loss:.6f}")
                break
                
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # 加载最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            
        # 记录训练信息
        training_record = {
            'timestamp': np.datetime64('now').astype(str),
            'training_sequences': len(train_loader.dataset),
            'final_loss': best_loss,
            'epochs_trained': epoch + 1,
            'early_stopped': patience_counter >= patience
        }
        self._add_training_record(training_record)
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数
        
        Args:
            X: 输入数据
            
        Returns:
            异常分数数组，基于重构误差
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用fit()方法")
        
        # 数据预处理
        X_processed = self._preprocess_data(X, fit_scaler=False)
        
        # 转换为序列数据
        X_sequences = self._create_sequences(X_processed)
        
        # 创建数据加载器
        test_loader = self._create_dataloader(X_sequences, X_sequences, shuffle=False)
        
        self.model.eval()
        anomaly_scores = []
        
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                
                # 前向传播
                reconstructed = self.model(batch_X)
                
                # 计算重构误差
                mse = nn.MSELoss(reduction='none')
                errors = mse(reconstructed, batch_X)
                
                # 计算每个序列的平均误差
                batch_scores = torch.mean(errors, dim=(1, 2)).cpu().numpy()
                anomaly_scores.extend(batch_scores)
        
        return np.array(anomaly_scores)
    
    def _preprocess_data(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """数据预处理"""
        if self.scaler is not None:
            if fit_scaler:
                X_processed = self.scaler.fit_transform(X)
                logger.info("数据标准化器已拟合")
            else:
                X_processed = self.scaler.transform(X)
        else:
            X_processed = X.copy()
        
        return X_processed
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """创建序列数据"""
        sequence_length = self.model_config['sequence_length']
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return np.array(sequences)
    
    def _create_dataloader(self, X: np.ndarray, y: np.ndarray, 
                          shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        return DataLoader(
            dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=shuffle,
            drop_last=False
        )
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scaler': self.scaler,
            'config': self.config,
            'model_config': self.model_config,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_history': self._training_history,
            'best_model_state': self.best_model_state
        }
        
        torch.save(model_data, filepath)
        logger.info(f"LSTM AutoEncoder模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """加载模型"""
        model_data = torch.load(filepath, map_location=self.device)
        
        # 重建模型结构
        if not self.model:
            # 从配置重建模型
            self.model_config = model_data['model_config']
            # 这里需要知道输入维度，可能需要额外保存
            # 暂时先用一个占位符
            input_dim = model_data['model_state_dict']['output_layer.weight'].shape[1]
            self.build_model((1, self.model_config['sequence_length'], input_dim))
        
        # 加载状态
        self.model.load_state_dict(model_data['model_state_dict'])
        
        if self.optimizer and model_data['optimizer_state_dict']:
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.model_config = model_data['model_config']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self._training_history = model_data.get('training_history', [])
        self.best_model_state = model_data.get('best_model_state')
        
        logger.info(f"LSTM AutoEncoder模型已从 {filepath} 加载")


# 注册模型到工厂
from .base_model import ModelFactory
ModelFactory.register_model('lstm_autoencoder', LSTMAutoEncoderModel)
ModelFactory.register_model('lstm_ae', LSTMAutoEncoderModel)  # 别名
