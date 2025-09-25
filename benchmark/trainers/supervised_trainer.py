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
        
        # 应用数据子集采样（如果配置了data_fraction）
        data_fraction = config.get('data_fraction', 1.0)
        if data_fraction < 1.0:
            train_size = int(len(X_train) * data_fraction)
            test_size = int(len(X_test) * data_fraction)
            
            # 随机采样训练数据（保持类别分布）
            train_indices = np.random.choice(len(X_train), train_size, replace=False)
            test_indices = np.random.choice(len(X_test), test_size, replace=False)
            
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
            X_test = X_test[test_indices]
            y_test = y_test[test_indices]
            
            print(f"   数据子集: {data_fraction:.1%} ({train_size:,}训练 + {test_size:,}测试)")
        
        # 数据转换
        self.X_train = torch.FloatTensor(X_train).to(device)
        self.y_train = torch.LongTensor(y_train.flatten()).to(device)
        self.X_test = torch.FloatTensor(X_test).to(device)
        self.y_test = torch.LongTensor(y_test.flatten()).to(device)
        
        # 存储训练参数
        self.batch_size = config.get('batch_size', 32)
        
        # 创建数据加载器
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
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
        
        print(f"   {epochs}轮训练 (耐心度:{patience})")
        
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
                    print(f"   E{epoch+1:3d}/{epochs}: {train_loss:.4f}→{val_loss:.4f} ↓{improvement:.4f}")
            else:
                self.patience_counter += 1
                if epoch % print_interval == 0 or epoch == epochs - 1:
                    print(f"   E{epoch+1:3d}/{epochs}: {train_loss:.4f}→{val_loss:.4f} ×{self.patience_counter}/{patience}")
            
            # 早停
            if self.patience_counter >= patience:
                print(f"   早停: {patience}轮无改善 @E{epoch+1}")
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
            'test_predictions': test_pred,
            # 返回实际使用的数据用于评估
            'actual_X_train': self.X_train.cpu().numpy(),
            'actual_y_train': self.y_train.cpu().numpy(),
            'actual_X_test': self.X_test.cpu().numpy(),
            'actual_y_test': self.y_test.cpu().numpy()
        }
        

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
        """验证模型 - 使用批处理避免OOM"""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        # 创建验证数据加载器
        val_dataset = torch.utils.data.TensorDataset(self.X_test, self.y_test)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.batch_size,  # 使用相同的batch_size
            shuffle=False
        )
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                
                # 对于序列输出，需要reshape
                if len(outputs.shape) == 3:
                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    batch_y = batch_y.repeat_interleave(outputs.shape[0] // batch_y.shape[0])
                
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else 0.0
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """生成预测 - 使用批处理避免OOM"""
        self.model.eval()
        all_predictions = []
        
        # 创建数据加载器进行批处理预测
        dataset = torch.utils.data.TensorDataset(X)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=False
        )
        
        with torch.no_grad():
            for (batch_X,) in data_loader:
                outputs = self.model(batch_X)
                
                # 对于序列输出，取最后一个时间步或平均
                if len(outputs.shape) == 3:  # (batch, seq, features)
                    outputs = outputs[:, -1, :]  # 取最后一个时间步
                
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.append(predictions.cpu().numpy())
        
        return np.concatenate(all_predictions)