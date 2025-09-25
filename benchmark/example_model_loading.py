"""简单直接的模型加载测试"""

import os
import sys
import torch
from pathlib import Path

# 切换到benchmark目录作为根目录
benchmark_dir = Path(__file__).parent
os.chdir(benchmark_dir)
sys.path.append(str(benchmark_dir))

from src.data_loader import DataLoader
from src.config_loader import ConfigLoader
from src.model_loader import ModelLoader
from src.eval_loader import EvalLoader
from src.training_loader import TrainingLoader
import numpy as np


def test_model_loading():
    """测试模型加载 - 简单直接"""
    print(f"当前工作目录: {os.getcwd()}")
    print("测试LSTM模型加载...")
    
    # 加载配置 - 现在路径是相对于benchmark目录的
    config_loader = ConfigLoader('configs/default_experiment.yaml')
    config = config_loader.load_config()
    

    
    # 新配置结构：从experiments中获取第一个实验
    first_experiment = config['experiments'][0]
    model_name = first_experiment['model']
    print(f"实验模型: {model_name}")
    
    # 测试数据加载
    print("测试数据加载...")
    dataset_name = first_experiment['dataset']
    print(f"使用数据集: {dataset_name}")
    
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test, metadata = data_loader.prepare_data(config, dataset_name)
    
    print(f"数据集: {metadata.dataset_name}")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"特征维度: {metadata.feature_dim}")

    # 创建模型加载器并加载模型
    model_loader = ModelLoader()
    # 使用真实的输入维度加载模型
    model = model_loader.load_model_from_config(model_name, config, input_dim=metadata.feature_dim)
    
    # 基本信息
    print(f"模型类型: {type(model).__name__}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"设备: {next(model.parameters()).device}")
    
    # 使用真实数据测试推理
    test_batch = torch.FloatTensor(X_test[:4])  # 取前4个样本
    if next(model.parameters()).is_cuda:
        test_batch = test_batch.cuda()
        
    with torch.no_grad():
        output = model(test_batch)
    
    print(f"真实数据推理: {test_batch.shape} -> {output.shape}")
    
    # 测试训练器
    print("\n测试训练器...")
    training_loader = TrainingLoader()
    training_template_name = first_experiment['training']
    
    trainer = training_loader.create_trainer(
        config, training_template_name, model,
        X_train, y_train, X_test, y_test
    )
    
    # 执行训练（使用较少epochs进行快速测试）
    # 临时修改epochs为快速测试
    original_epochs = config['training_templates'][training_template_name]['epochs']
    config['training_templates'][training_template_name]['epochs'] = 5
    
    training_results = trainer.train()
    
    # 恢复原始epochs
    config['training_templates'][training_template_name]['epochs'] = original_epochs
    
    print(f"训练完成，总轮数: {training_results['total_epochs']}")
    print(f"最终训练损失: {training_results['final_train_loss']:.4f}")
    print(f"最终验证损失: {training_results['final_val_loss']:.4f}")
    
    # 测试评估器
    print("\n测试评估器...")
    eval_loader = EvalLoader()
    eval_template_name = first_experiment['evaluation']
    
    # 使用训练器的预测结果
    y_train_pred = training_results['train_predictions']
    y_test_pred = training_results['test_predictions']
    
    results = eval_loader.evaluate(config, eval_template_name,
                                 X_train, y_train, y_train_pred,
                                 X_test, y_test, y_test_pred)
    
    print("评估结果:")
    for metric, score in results.items():
        if score is not None:
            print(f"  {metric}: {score:.4f}")
        else:
            print(f"  {metric}: 失败")
    
    print("✅ 完整流程测试完成!")

if __name__ == "__main__":
    test_model_loading()