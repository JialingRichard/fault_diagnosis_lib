# 时序异常检测基准测试框架（Python脚本版）
# 直接运行本脚本即可测试主流程，无需Jupyter环境

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
# warnings.filterwarnings('ignore')


# 添加项目路径
# project_root = Path.cwd().parent
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))
# print(f"项目根目录: {project_root}")

project_root = Path.cwd()/ "benchmark"
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
print(f"项目根目录: {project_root}")

# 导入框架组件
from src.config_manager import ConfigManager
from src.dataloaders import DataPipeline
from models.base_model import ModelFactory
from src.trainer import Trainer
from src.metrics import TimeSeriesEvaluator, print_evaluation_report

# 设置随机种子
import random
random.seed(42)
np.random.seed(42)

# 加载实验配置
config_file = "/home/chen/dev/fault_diagnosis_lib/benchmark/configs/default_experiment.yaml"
config_manager = ConfigManager(config_dir = config_file)
config = config_manager.load_config()
print("实验配置:")
print(f"   - 实验名称: {config['experiment']['name']}")
print(f"   - 数据集: {config['data']['dataset']}")
print(f"   - 模型: {list(config['models'].keys())}")
print(f"   - 评估指标: {config['evaluation']['metrics']}")

# 数据加载与预处理
print(f"加载数据集: {config['data']['dataset']}")
data_pipeline = DataPipeline(config)
try:
    train_data, train_labels, test_data, test_labels, metadata = data_pipeline.load_dataset(
        config['data']['dataset'], config['data'])
    print("数据加载成功")
except Exception as e:
    print(f"真实数据加载失败: {e}")
    print("使用模拟数据进行演示")
    from src.dataloaders import DataMetadata
    n_samples = 1000
    n_features = 10
    train_data = np.random.randn(n_samples, n_features)
    train_labels = np.zeros(n_samples)
    test_data = np.random.randn(n_samples, n_features)
    test_labels = np.zeros(n_samples)
    anomaly_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    test_data[anomaly_indices] += 3
    test_labels[anomaly_indices] = 1
    metadata = DataMetadata(
        dataset_name="simulated_data",
        label_granularity="point-wise",
        fault_type="binary",
        num_classes=2,
        sequence_length=n_samples,
        feature_dim=n_features,
    )
print(f"数据统计:")
print(f"   - 训练集: {train_data.shape}")
print(f"   - 测试集: {test_data.shape}")
print(f"   - 特征数: {metadata.feature_dim}")
print(f"   - 故障类型: {metadata.fault_type}")

# 初始化结果存储与评估器
experiment_results = {}
model_objects = {}
evaluator = TimeSeriesEvaluator(tolerance=config['evaluation']['tolerance'])
print("开始模型训练与评估")
print("=" * 60)

# 训练和评估 Isolation Forest
if 'iforest' in config['models']:
    print("\nIsolation Forest")
    print("-" * 40)
    iforest_config = config['models']['iforest']
    model = ModelFactory.create_model('iforest', iforest_config)
    print("训练模型...")
    # 传入metadata需使用关键字，避免被当作y_train
    model.fit(train_data, metadata=metadata)
    print("生成预测...")
    anomaly_scores = model.predict_anomaly_score(test_data)
    print("评估性能...")
    results = evaluator.evaluate(test_labels, anomaly_scores, metadata)
    experiment_results['iforest'] = results
    model_objects['iforest'] = model
    print_evaluation_report(results, "Isolation Forest 评估结果")

# 训练和评估 LSTM AutoEncoder
if 'lstm_ae' in config['models']:
    print("\nLSTM AutoEncoder")
    print("-" * 40)
    lstm_config = config['models']['lstm_ae']
    model = ModelFactory.create_model('lstm_ae', lstm_config)
    print("训练模型...")
    model.fit(train_data, train_labels)
    print("生成预测...")
    anomaly_scores = model.predict_anomaly_score(test_data)
    print("评估性能...")
    results = evaluator.evaluate(test_labels, anomaly_scores, metadata)
    experiment_results['lstm_ae'] = results
    model_objects['lstm_ae'] = model
    print_evaluation_report(results, "LSTM AutoEncoder 评估结果")

# 结果汇总对比
if experiment_results:
    print("\n模型性能对比")
    print("=" * 80)
    comparison_metrics = ['f1', 'precision', 'recall', 'auc', 'f1_point_adjusted']
    comparison_data = []
    for model_name, results in experiment_results.items():
        row = {'模型': model_name}
        for metric in comparison_metrics:
            value = results.get(metric, None)
            if isinstance(value, float) and not np.isnan(value):
                row[metric] = f"{value:.4f}"
            else:
                row[metric] = "N/A"
        comparison_data.append(row)
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    best_model = None
    best_f1 = -1
    for model_name, results in experiment_results.items():
        if 'f1' in results and not np.isnan(results['f1']):
            if results['f1'] > best_f1:
                best_f1 = results['f1']
                best_model = model_name
    if best_model:
        print(f"\n最佳模型: {best_model} (F1 Score: {best_f1:.4f})")
else:
    print("没有成功的实验结果")

# 可视化结果
if experiment_results and config['output']['generate_plots']:
    print("\nGenerating visualization charts")
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Time Series Anomaly Detection Benchmark Results', fontsize=16, fontweight='bold')
    # 1. Performance metrics bar chart
    ax1 = axes[0, 0]
    metrics_to_plot = ['f1', 'precision', 'recall']
    model_names = list(experiment_results.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.35 if len(model_names) == 2 else 0.25
    for i, model_name in enumerate(model_names):
        values = []
        for metric in metrics_to_plot:
            value = experiment_results[model_name].get(metric, 0)
            if isinstance(value, float) and not np.isnan(value):
                values.append(value)
            else:
                values.append(0)
        ax1.bar(x + i * width, values, width, label=model_name, alpha=0.8)
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax1.set_xticklabels(metrics_to_plot)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    # 3. Time series anomaly detection example (first 200 points)
    ax3 = axes[1, 0]
    n_points = min(200, len(test_labels))
    time_indices = np.arange(n_points)
    ax3.plot(time_indices, test_data[:n_points, 0], 'b-', alpha=0.7, label='Time Series')
    anomaly_indices = np.where(test_labels[:n_points] == 1)[0]
    if len(anomaly_indices) > 0:
        ax3.scatter(anomaly_indices, test_data[anomaly_indices, 0], color='red', s=50, label='True Anomaly', zorder=5)
    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('Feature Value')
    ax3.set_title('Time Series Anomaly Detection Example')
    ax3.legend()
    ax3.grid(alpha=0.3)
    # 4. Confusion matrix (for best model)
    if best_model:
        from sklearn.metrics import confusion_matrix
        ax4 = axes[1, 1]
        best_model_obj = model_objects[best_model]
        if best_model == 'iforest':
            best_pred = best_model_obj.predict(test_data)
        elif best_model == 'lstm_ae':
            scores_test = best_model_obj.predict_anomaly_score(test_data)
            scores_train = best_model_obj.predict_anomaly_score(train_data)
            thresh = np.percentile(scores_train, 90)
            best_pred = (scores_test > thresh).astype(int)
        else:
            best_pred = np.zeros_like(test_labels)
        cm = confusion_matrix(test_labels, best_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
        ax4.set_xlabel('Predicted Label')
        ax4.set_ylabel('True Label')
        ax4.set_title(f'{best_model} Confusion Matrix')
        ax4.xaxis.set_ticklabels(['Normal', 'Anomaly'])
        ax4.yaxis.set_ticklabels(['Normal', 'Anomaly'])
    plt.tight_layout()
    plt.show()
    print("Visualization charts generated")

# 保存实验结果
if config['output']['save_results'] and experiment_results:
    print("\nSaving experiment results")
    results_dir = project_root / config['output']['results_dir']
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = config['experiment']['name']
    train_size = train_data.shape[0] if 'train_data' in locals() else None
    test_size = test_data.shape[0] if 'test_data' in locals() else None
    anomaly_rate = (np.sum(test_labels) / len(test_labels)) if 'test_labels' in locals() else None
    detailed_results = {
        'experiment_info': config['experiment'],
        'data_info': {
            'dataset': getattr(metadata, 'dataset_name', 'unknown'),
            'n_features': getattr(metadata, 'feature_dim', None),
            'train_size': train_size,
            'test_size': test_size,
            'anomaly_rate': anomaly_rate,
            'fault_type': getattr(metadata, 'fault_type', None)
        },
        'model_results': experiment_results,
        'config': config
    }
    # 保存为YAML文件
    results_file = results_dir / f"{experiment_name}_{timestamp}.yaml"
    with open(results_file, 'w', encoding='utf-8') as f:
        yaml.dump(detailed_results, f, default_flow_style=False, allow_unicode=True)
    print(f"Results saved to: {results_file}")
    # 保存对比表格为CSV
    if 'comparison_df' in locals():
        csv_file = results_dir / f"{experiment_name}_{timestamp}_comparison.csv"
        comparison_df.to_csv(csv_file, index=False)
        print(f"Comparison table saved to: {csv_file}")
    # 保存可视化图表
    if config['output']['generate_plots']:
        plot_file = results_dir / f"{experiment_name}_{timestamp}_plots.png"
        if 'fig' in locals():
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_file}")

# 实验总结报告
print("\n实验总结报告")
print("=" * 80)
print(f"实验名称: {config['experiment']['name']}")
print(f"数据集: {metadata.dataset_name}")
print(f"评估指标: {', '.join(config['evaluation']['metrics'])}")
if experiment_results:
    print(f"\n主要发现:")
    for model_name, results in experiment_results.items():
        f1_score = results.get('f1', 0)
        precision = results.get('precision', 0)
        recall = results.get('recall', 0)
        print(f" - {model_name}: F1={f1_score:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
print("\n基准测试实验完成")
