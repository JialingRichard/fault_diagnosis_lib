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
from src.result_manager import ResultManager

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
    
    # 2. 初始化系统组件（包括结果管理器）
    print("步骤 2: 初始化系统组件")
    result_manager = ResultManager(config_file)
    data_loader = DataLoader()
    model_loader = ModelLoader()
    training_loader = TrainingLoader()
    eval_loader = EvalLoader()
    print("所有组件就绪")
    print()
    
    # 3. 运行实验
    results_summary = []
    
    for exp_idx, experiment in enumerate(experiments, 1):
        print(f"\n{'='*20} 实验 {exp_idx}/{len(experiments)}: {experiment['name']} {'='*20}")
        
        # 获取训练配置中的epochinfo信息
        training_config = config['training_templates'][experiment['training']]
        epochinfo_name = training_config.get('epochinfo', 'default')
        
        print(f"Model:{experiment['model']} | Data:{experiment['dataset']} | Train:{experiment['training']} | Epoch:{epochinfo_name} | Eval:{experiment['evaluation']}")
        print("-" * 80)
        
        try:
            # 3.1 数据加载
            # 数据加载
            X_train, X_test, y_train, y_test, metadata = data_loader.prepare_data(
                config, experiment['dataset']
            )
            print(f"数据: {X_train.shape[0]:,}训练+{X_test.shape[0]:,}测试 | 特征:{metadata.feature_dim} | 类别:{metadata.num_classes}")
            
            # 模型加载
            model = model_loader.load_model_from_config(
                experiment['model'], config, input_dim=metadata.feature_dim
            )
            param_count = sum(p.numel() for p in model.parameters())
            print(f"模型: {param_count:,}参数")
            
            # 开始训练
            print(f"开始训练...")
            trainer = training_loader.create_trainer(
                config, experiment['training'], model,
                X_train, y_train, X_test, y_test
            )
            # 设置eval_loader和result_manager给训练器
            trainer.eval_loader = eval_loader
            trainer.result_manager = result_manager
            trainer.experiment_name = experiment['name']
            training_results = trainer.train()
            
            # 3.4 评估
            print(f"开始评估...")
            
            # 使用训练器实际使用的数据进行评估（可能是子集）
            actual_X_train = training_results.get('actual_X_train', X_train)
            actual_y_train = training_results.get('actual_y_train', y_train)
            actual_X_test = training_results.get('actual_X_test', X_test)
            actual_y_test = training_results.get('actual_y_test', y_test)
            
            # 为绘图evaluator设置plots目录
            plots_dir = result_manager.get_experiment_plot_dir(experiment['name'])
            
            # 设置绘图evaluator的路径（如果存在的话）
            try:
                from evaluators.plot_label_distribution import set_plots_dir
                set_plots_dir(str(plots_dir))
            except ImportError:
                pass  # 如果模块不存在就忽略
            
            eval_results = eval_loader.evaluate(
                config, experiment['evaluation'],
                actual_X_train, actual_y_train, training_results['train_predictions'],
                actual_X_test, actual_y_test, training_results['test_predictions']
            )
            
            print(f"训练完成: {training_results['total_epochs']}轮 | 验证损失: {training_results['final_val_loss']:.4f}")
            print(f"评估结果: ", end="")
            for metric, score in eval_results.items():
                if score is not None:
                    print(f"{metric}={score:.4f} ", end="")
            print()
            
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
            
            print(f"实验完成\n")
            
        except Exception as e:
            print(f"实验失败: {str(e)}")
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
            
            params = result.get('parameters', 0) or 0  # 处理None值
            epochs = result.get('epochs', 0) or 0  # 处理None值
            val_loss = result.get('val_loss', 0) or 0  # 处理None值
            
            print(f"{result['name']:<25} {result['model']:<8} {params:<10,} "
                  f"{epochs:<6} {val_loss:<10.4f} "
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
