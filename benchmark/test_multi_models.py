"""多模型比较测试"""

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


def test_model_comparison():
    """测试多模型比较"""
    print(f"当前工作目录: {os.getcwd()}")
    print("=" * 60)
    print("多模型比较测试")
    print("=" * 60)
    
    # 加载配置
    config_loader = ConfigLoader('configs/default_experiment.yaml')
    config = config_loader.load_config()
    
    # 加载数据（所有实验使用相同数据）
    data_loader = DataLoader()
    dataset_name = config['experiments'][0]['dataset']  # 使用第一个实验的数据集
    X_train, X_test, y_train, y_test, metadata = data_loader.prepare_data(config, dataset_name)
    
    print(f"数据集: {metadata.dataset_name}")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"特征维度: {metadata.feature_dim}, 类别数: {metadata.num_classes}")
    print()
    
    # 创建加载器
    model_loader = ModelLoader()
    training_loader = TrainingLoader()
    eval_loader = EvalLoader()
    
    results_summary = []
    
    # 测试每个实验
    for exp_idx, experiment in enumerate(config['experiments']):
        print(f"实验 {exp_idx + 1}: {experiment['name']}")
        print("-" * 40)
        
        # 加载模型
        model_name = experiment['model']
        model = model_loader.load_model_from_config(model_name, config, input_dim=metadata.feature_dim)
        
        print(f"模型: {type(model).__name__}")
        print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建训练器
        training_template_name = experiment['training']
        trainer = training_loader.create_trainer(
            config, training_template_name, model,
            X_train, y_train, X_test, y_test
        )
        
        # 快速训练（减少epochs）
        original_epochs = config['training_templates'][training_template_name]['epochs']
        config['training_templates'][training_template_name]['epochs'] = 3  # 快速测试
        
        training_results = trainer.train()
        
        # 恢复epochs
        config['training_templates'][training_template_name]['epochs'] = original_epochs
        
        print(f"训练完成，总轮数: {training_results['total_epochs']}")
        print(f"最终验证损失: {training_results['final_val_loss']:.4f}")
        
        # 评估
        eval_template_name = experiment['evaluation']
        y_train_pred = training_results['train_predictions']
        y_test_pred = training_results['test_predictions']
        
        eval_results = eval_loader.evaluate(config, eval_template_name,
                                          X_train, y_train, y_train_pred,
                                          X_test, y_test, y_test_pred)
        
        print("评估结果:")
        for metric, score in eval_results.items():
            if score is not None:
                print(f"  {metric}: {score:.4f}")
        
        # 保存结果
        results_summary.append({
            'experiment': experiment['name'],
            'model': model_name,
            'training': training_template_name,
            'parameters': sum(p.numel() for p in model.parameters()),
            'val_loss': training_results['final_val_loss'],
            'eval_results': eval_results
        })
        
        print()
    
    # 打印对比总结
    print("=" * 60)
    print("实验对比总结")
    print("=" * 60)
    
    print(f"{'模型':<10} {'参数量':<10} {'验证损失':<10} {'F1':<8} {'准确率':<8}")
    print("-" * 60)
    
    for result in results_summary:
        f1_score = result['eval_results'].get('f1', 0)
        accuracy = result['eval_results'].get('accuracy', 0)
        print(f"{result['model']:<10} {result['parameters']:<10,} {result['val_loss']:<10.4f} {f1_score:<8.4f} {accuracy:<8.4f}")
    
    print("\n✅ 多模型比较测试完成!")


if __name__ == "__main__":
    test_model_comparison()