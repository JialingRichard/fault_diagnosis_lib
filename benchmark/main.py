#!/usr/bin/env python3
"""
Fault Diagnosis Library - 主程序入口
基于配置的多模型多数据集故障诊断基准测试框架

使用方法:
    python main.py [config_file]
    
默认配置文件: configs/default_experiment.yaml
"""
import os
import sys
import argparse
import logging

# 确保从benchmark目录运行并设置路径
benchmark_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(benchmark_dir)
sys.path.append(str(benchmark_dir))

from src.config_loader import ConfigLoader
from src.data_loader import DataLoader
from src.model_loader import ModelLoader
from src.training_loader import TrainingLoader
from src.eval_loader import EvalLoader

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def run_experiments(config_file: str = 'configs/default_experiment.yaml'):
    """
    运行完整的实验流程
    
    Args:
        config_file: 配置文件路径
    """
    print("=" * 80)
    print("Fault Diagnosis Library - 故障诊断基准测试")
    print("=" * 80)
    print(f"配置文件: {config_file}")
    print(f"工作目录: {os.getcwd()}")
    print()
    
    # 1. 加载配置
    print("步骤 1: 加载实验配置")
    config_loader = ConfigLoader(config_file)
    config = config_loader.load_config()
    
    experiments = config.get('experiments', [])
    print(f"   发现 {len(experiments)} 个实验配置")
    for i, exp in enumerate(experiments, 1):
        print(f"   {i}. {exp['name']} ({exp['model']} on {exp['dataset']})")
    print()
    
    # 2. 初始化加载器
    print("步骤 2: 初始化系统组件")
    data_loader = DataLoader()
    model_loader = ModelLoader()
    training_loader = TrainingLoader()
    eval_loader = EvalLoader()
    print("   所有组件初始化完成")
    print()
    
    # 3. 运行实验
    results_summary = []
    
    for exp_idx, experiment in enumerate(experiments, 1):
        print(f"步骤 3.{exp_idx}: 执行实验 - {experiment['name']}")
        print("-" * 60)
        
        try:
            # 3.1 数据加载
            print(f"   加载数据集: {experiment['dataset']}")
            X_train, X_test, y_train, y_test, metadata = data_loader.prepare_data(
                config, experiment['dataset']
            )
            print(f"   训练集: {X_train.shape}, 测试集: {X_test.shape}")
            print(f"   特征维度: {metadata.feature_dim}, 类别数: {metadata.num_classes}")
            
            # 3.2 模型加载
            print(f"   加载模型: {experiment['model']}")
            model = model_loader.load_model_from_config(
                experiment['model'], config, input_dim=metadata.feature_dim
            )
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   模型参数数量: {param_count:,}")
            
            # 3.3 训练
            print(f"   开始训练: {experiment['training']}")
            trainer = training_loader.create_trainer(
                config, experiment['training'], model,
                X_train, y_train, X_test, y_test
            )
            training_results = trainer.train()
            
            print(f"   训练完成: {training_results['total_epochs']} epochs")
            print(f"   最终验证损失: {training_results['final_val_loss']:.4f}")
            
            # 3.4 评估
            print(f"   开始评估: {experiment['evaluation']}")
            eval_results = eval_loader.evaluate(
                config, experiment['evaluation'],
                X_train, y_train, training_results['train_predictions'],
                X_test, y_test, training_results['test_predictions']
            )
            
            print(f"   评估结果:")
            for metric, score in eval_results.items():
                if score is not None:
                    print(f"     {metric}: {score:.4f}")
            
            # 保存结果
            results_summary.append({
                'name': experiment['name'],
                'model': experiment['model'],
                'dataset': experiment['dataset'],
                'training': experiment['training'],
                'parameters': param_count,
                'epochs': training_results['total_epochs'],
                'val_loss': training_results['final_val_loss'],
                'eval_results': eval_results
            })
            
            print(f"   实验 {experiment['name']} 完成")
            
        except Exception as e:
            print(f"   实验 {experiment['name']} 失败: {str(e)}")
            logging.error(f"实验失败: {experiment['name']}", exc_info=True)
            
        print()
    
    # 4. 生成总结报告
    print("步骤 4: 生成实验总结报告")
    print("=" * 80)
    print("实验对比总结")
    print("=" * 80)
    
    if results_summary:
        # 表头
        print(f"{'实验名称':<25} {'模型':<8} {'参数量':<10} {'轮数':<6} {'验证损失':<10} {'F1':<8} {'准确率':<8}")
        print("-" * 80)
        
        # 结果行
        for result in results_summary:
            f1_score = result['eval_results'].get('f1', 0)
            accuracy = result['eval_results'].get('accuracy', 0)
            
            print(f"{result['name']:<25} {result['model']:<8} {result['parameters']:<10,} "
                  f"{result['epochs']:<6} {result['val_loss']:<10.4f} "
                  f"{f1_score:<8.4f} {accuracy:<8.4f}")
        
        print()
        
        # 最佳结果
        if len(results_summary) > 1:
            best_f1 = max(results_summary, key=lambda x: x['eval_results'].get('f1', 0))
            best_acc = max(results_summary, key=lambda x: x['eval_results'].get('accuracy', 0))
            
            print("最佳结果:")
            print(f"   最高 F1: {best_f1['name']} (F1: {best_f1['eval_results'].get('f1', 0):.4f})")
            print(f"   最高准确率: {best_acc['name']} (准确率: {best_acc['eval_results'].get('accuracy', 0):.4f})")
    else:
        print("没有成功完成的实验")
    
    print()
    print("所有实验完成")
    print("=" * 80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="故障诊断基准测试框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                          # 使用默认配置
  python main.py configs/my_config.yaml   # 使用自定义配置
        """
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default='configs/default_experiment.yaml',
        help='配置文件路径 (默认: configs/default_experiment.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='启用详细日志输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    setup_logging()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return 1
    
    try:
        run_experiments(args.config)
        return 0
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        logging.error("程序执行失败", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
