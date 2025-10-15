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
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))
from epochinfo_loader import EpochInfoLoader

logger = logging.getLogger(__name__)


class SupervisedTrainer:
    """
    监督学习训练器
    
    标准的监督学习训练流程
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str,
                 X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, full_config: Dict[str, Any] = None):
        """
        初始化监督学习训练器
        
        Args:
            model: 待训练的模型
            config: 训练配置
            device: 训练设备
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            full_config: 完整配置（用于访问epochinfo_templates等）
        """
        self.model = model
        self.config = config
        self.full_config = full_config or config
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
        
        # 从训练集中划分验证集
        validation_split = config.get('validation_split', 0.2)
        if validation_split > 0.0 and validation_split < 1.0:
            from sklearn.model_selection import train_test_split
            
            # 检查是否可以使用分层抽样（每个类至少2个样本）
            y_flat = y_train.flatten()
            unique_classes, class_counts = np.unique(y_flat, return_counts=True)
            min_samples_per_class = class_counts.min()
            
            # 如果最少的类别样本数大于1，使用分层抽样；否则使用普通随机抽样
            if min_samples_per_class > 1:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train,
                    test_size=validation_split,
                    stratify=y_flat,
                    random_state=42
                )
                print(f"   使用分层抽样划分验证集")
            else:
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train,
                    test_size=validation_split,
                    random_state=42
                )
                print(f"   警告: 数据集类别样本过少（最少类别仅{min_samples_per_class}个样本），使用普通随机抽样划分验证集")
            
            train_samples = len(X_train_split)
            val_samples = len(X_val)
            print(f"   数据划分: {train_samples:,}训练 + {val_samples:,}验证 + {len(X_test):,}测试")
            
            # 更新训练数据为划分后的训练集
            X_train = X_train_split
            y_train = y_train_split
            
            # 设置验证集
            self.X_val = torch.FloatTensor(X_val).to(device)
            self.y_val = torch.LongTensor(y_val.flatten()).to(device)
        else:
            # 如果没有配置验证集划分，使用测试集作为验证集（保持原有行为）
            print(f"   警告: 未配置validation_split，使用测试集作为验证集（可能导致数据泄露）")
            self.X_val = torch.FloatTensor(X_test).to(device)
            self.y_val = torch.LongTensor(y_test.flatten()).to(device)
        
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
        
        # 初始化epoch信息加载器
        self.epochinfo_loader = EpochInfoLoader()
        self.eval_loader = None  # 将由外部设置
        self.result_manager = None  # 将由外部设置
        self.experiment_name = None  # 将由外部设置
        self.best_checkpoint_path = None  # 记录最优checkpoint
        

    
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
            
            # 早停检查和checkpoint保存
            improvement = None
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # 保存checkpoint（当val_loss改善时）
                if self.result_manager and self.experiment_name:
                    logging_level = self.config.get('logging_level', 'normal')
                    ckpt_path = self.result_manager.save_checkpoint(
                        self.experiment_name, self.model, self.optimizer, 
                        epoch + 1, val_loss,  # epoch+1因为从0开始计数
                        logging_level=logging_level
                    )
                    try:
                        self.best_checkpoint_path = ckpt_path
                    except Exception:
                        self.best_checkpoint_path = None
            else:
                self.patience_counter += 1
            
            # 打印epoch信息（使用模块化系统）
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
                    # 提供计算准确率的方法，但不主动计算
                    'trainer': self
                }
                
                # 获取epoch信息模板配置
                epochinfo_template = self.config.get('epochinfo', 'default')
                # 设置eval_loader如果还没设置
                if self.eval_loader and not self.epochinfo_loader.eval_loader:
                    self.epochinfo_loader.eval_loader = self.eval_loader
                self.epochinfo_loader.print_epoch_info(self.full_config, epochinfo_template, epoch_data)

                # 本轮打印完成
            
            # 早停
            if self.patience_counter >= patience:
                print(f"   早停: {patience}轮无改善 @E{epoch+1}")
                break
        
        # 如配置要求：使用最优checkpoint进行最终评估
        load_best = self.config.get('load_best_checkpoint_for_eval', True)
        selected_checkpoint = None
        if load_best and self.best_checkpoint_path is not None:
            try:
                checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                selected_checkpoint = str(self.best_checkpoint_path)
            except Exception as e:
                logging.getLogger(__name__).warning(f"加载最佳checkpoint失败，将使用当前模型: {e}")

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
            'selected_checkpoint': selected_checkpoint,
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
        """验证模型 - 使用真正的验证集避免数据泄露"""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        # 创建验证数据加载器 - 使用真正的验证集
        val_dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
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
    
    def _calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算准确率
        
        Args:
            X: 输入数据
            y: 真实标签
            
        Returns:
            准确率
        """
        predictions = self.predict(X)
        # 确保y是numpy数组而不是tensor
        if hasattr(y, 'cpu'):
            y = y.cpu().numpy()
        return float(np.mean(predictions == y))
