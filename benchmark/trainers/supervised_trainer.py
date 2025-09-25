"""
监督学习训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SupervisedTrainer:
    """
    监督学习训练器
    
    标准的监督学习训练流程
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str,
                 X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray):
        """
        初始化监督学习训练器
        
        Args:
            model: 待训练的模型
            config: 训练配置
            device: 训练设备
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
        """
        self.model = model
        self.config = config
        self.device = device
        
        # 数据转换
        self.X_train = torch.FloatTensor(X_train).to(device)
        self.y_train = torch.LongTensor(y_train.flatten()).to(device)
        self.X_test = torch.FloatTensor(X_test).to(device)
        self.y_test = torch.LongTensor(y_test.flatten()).to(device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.get('batch_size', 32),
            shuffle=True
        )
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"监督学习训练器初始化完成")
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('lr', 0.001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def train(self) -> Dict[str, Any]:
        """
        执行训练
        
        Returns:
            训练结果字典
        """
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        
        logger.info(f"开始训练，epochs: {epochs}, patience: {patience}")
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = self._train_epoch()
            
            # 验证阶段
            val_loss = self._validate()
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # 根据配置的打印间隔输出训练信息
            print_interval = self.config.get('print_interval', 10)
            
            # 早停检查
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if epoch % print_interval == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (改善 {improvement:.4f})")
            else:
                self.patience_counter += 1
                if epoch % print_interval == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (无改善 {self.patience_counter}/{patience})")
            
            # 早停
            if self.patience_counter >= patience:
                logger.info(f"早停触发: 连续 {patience} 轮验证损失无改善，epoch: {epoch+1}")
                break
        
        # 生成预测结果
        train_pred = self.predict(self.X_train)
        test_pred = self.predict(self.X_test)
        
        results = {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'total_epochs': epoch + 1,
            'training_history': self.training_history,
            'train_predictions': train_pred,
            'test_predictions': test_pred
        }
        
        logger.info(f"训练完成，最终验证损失: {val_loss:.4f}")
        return results
    
    def _train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in self.train_loader:
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(batch_X)
            
            # 对于序列输出，需要reshape
            if len(outputs.shape) == 3:  # (batch, seq, features)
                outputs = outputs.reshape(-1, outputs.shape[-1])
                batch_y = batch_y.repeat_interleave(outputs.shape[0] // batch_y.shape[0])
            
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _validate(self) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            outputs = self.model(self.X_test)
            
            # 对于序列输出，需要reshape
            if len(outputs.shape) == 3:
                outputs = outputs.reshape(-1, outputs.shape[-1])
                y_test = self.y_test.repeat_interleave(outputs.shape[0] // self.y_test.shape[0])
            else:
                y_test = self.y_test
            
            loss = self.criterion(outputs, y_test)
            total_loss = loss.item()
        
        return total_loss
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """生成预测"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            
            # 对于序列输出，取最后一个时间步或平均
            if len(outputs.shape) == 3:  # (batch, seq, features)
                outputs = outputs[:, -1, :]  # 取最后一个时间步
            
            predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()