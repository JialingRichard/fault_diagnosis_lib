"""
Evaluation Metrics for Time Series Anomaly Detection
====================================================

This module provides comprehensive evaluation metrics specifically designed
for time series anomaly detection tasks.

Author: Fault Diagnosis Benchmark Team
Date: 2025-01-11
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings

from models.base_model import DataMetadata

logger = logging.getLogger(__name__)


class TimeSeriesEvaluator:
    """
    时序异常检测评估器
    
    提供专门针对时序异常检测的评估指标，包括：
    - 传统点级指标 (precision, recall, f1)
    - 时序特定指标 (point-adjusted metrics)
    - 早期检测指标
    - 容忍度评估
    """
    
    def __init__(self, tolerance: int = 0):
        """
        初始化评估器
        
        Args:
            tolerance: 时间容忍度，用于容忍评估
        """
        self.tolerance = tolerance
        self.supported_metrics = [
            'precision', 'recall', 'f1', 'accuracy', 'auc',
            'f1_point_adjusted', 'precision_pa', 'recall_pa',
            'detection_delay', 'early_detection_rate',
            'range_precision', 'range_recall', 'range_f1'
        ]
    
    def evaluate(self, y_true: np.ndarray, anomaly_scores: np.ndarray,
                metadata: DataMetadata, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        根据元数据自动选择合适的评估方法
        
        Args:
            y_true: 真实标签
            anomaly_scores: 异常分数
            metadata: 数据元数据
            threshold: 异常检测阈值（可选）
            
        Returns:
            评估结果字典
        """
        
        logger.info(f"开始评估，数据类型: {metadata.fault_type}, "
                   f"标签粒度: {metadata.label_granularity}")
        
        # 根据标签粒度和故障类型选择评估策略
        if metadata.label_granularity == "point-wise":
            if metadata.fault_type == "binary":
                return self._evaluate_binary_pointwise(y_true, anomaly_scores, threshold)
            else:
                return self._evaluate_multiclass_pointwise(y_true, anomaly_scores, threshold)
        else:  # sequence-wise
            if metadata.fault_type == "binary":
                return self._evaluate_binary_sequence(y_true, anomaly_scores, threshold)
            else:
                return self._evaluate_multiclass_sequence(y_true, anomaly_scores, threshold)
    
    def _evaluate_binary_pointwise(self, y_true: np.ndarray, 
                                  anomaly_scores: np.ndarray,
                                  threshold: Optional[float] = None) -> Dict[str, float]:
        """二分类逐点评估"""
        
        # 确定阈值
        if threshold is None:
            threshold, _ = self._find_best_threshold(y_true, anomaly_scores)
        
        y_pred = (anomaly_scores > threshold).astype(int)
        
        # 基础指标
        results = {
            'threshold': threshold,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
        }
        
        # AUC（如果可能）
        try:
            if len(np.unique(y_true)) > 1:
                results['auc'] = roc_auc_score(y_true, anomaly_scores)
            else:
                results['auc'] = np.nan
        except ValueError:
            results['auc'] = np.nan
        
        # Point-Adjusted指标
        pa_metrics = self._compute_point_adjusted_metrics(y_true, y_pred)
        results.update(pa_metrics)
        
        # 时序特定指标
        ts_metrics = self._compute_time_series_metrics(y_true, y_pred, anomaly_scores)
        results.update(ts_metrics)
        
        return results
    
    def _compute_point_adjusted_metrics(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> Dict[str, float]:
        """计算Point-Adjusted指标"""
        
        # 找到真实和预测的异常区间
        true_ranges = self._find_anomaly_ranges(y_true)
        pred_ranges = self._find_anomaly_ranges(y_pred)
        
        if len(true_ranges) == 0 and len(pred_ranges) == 0:
            # 没有异常
            return {
                'f1_point_adjusted': 1.0,
                'precision_pa': 1.0,
                'recall_pa': 1.0,
                'range_precision': 1.0,
                'range_recall': 1.0,
                'range_f1': 1.0
            }
        
        # 计算区间级别的TP, FP, FN
        tp_ranges = 0
        matched_true_ranges = set()
        
        for pred_range in pred_ranges:
            for i, true_range in enumerate(true_ranges):
                if self._ranges_overlap(pred_range, true_range):
                    tp_ranges += 1
                    matched_true_ranges.add(i)
                    break
        
        fp_ranges = len(pred_ranges) - tp_ranges
        fn_ranges = len(true_ranges) - len(matched_true_ranges)
        
        # Range-based指标
        range_precision = tp_ranges / max(len(pred_ranges), 1)
        range_recall = tp_ranges / max(len(true_ranges), 1)
        range_f1 = 2 * range_precision * range_recall / max(range_precision + range_recall, 1e-10)
        
        # Point-Adjusted指标（更严格的版本）
        if tp_ranges + fp_ranges == 0:
            precision_pa = 0.0
        else:
            precision_pa = tp_ranges / (tp_ranges + fp_ranges)
        
        if tp_ranges + fn_ranges == 0:
            recall_pa = 0.0 
        else:
            recall_pa = tp_ranges / (tp_ranges + fn_ranges)
        
        if precision_pa + recall_pa == 0:
            f1_pa = 0.0
        else:
            f1_pa = 2 * precision_pa * recall_pa / (precision_pa + recall_pa)
        
        return {
            'f1_point_adjusted': f1_pa,
            'precision_pa': precision_pa,
            'recall_pa': recall_pa,
            'range_precision': range_precision,
            'range_recall': range_recall,
            'range_f1': range_f1
        }
    
    def _compute_time_series_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   anomaly_scores: np.ndarray) -> Dict[str, float]:
        """计算时序特定指标"""
        
        true_ranges = self._find_anomaly_ranges(y_true)
        
        if len(true_ranges) == 0:
            return {
                'detection_delay': 0.0,
                'early_detection_rate': 1.0
            }
        
        # 检测延迟
        total_delay = 0
        detected_ranges = 0
        early_detections = 0
        
        for true_start, true_end in true_ranges:
            # 在真实异常区间内找到第一个预测异常点
            anomaly_region = y_pred[true_start:true_end+1]
            
            if np.any(anomaly_region):
                # 找到第一个预测异常点
                first_detection = np.argmax(anomaly_region) + true_start
                delay = max(0, first_detection - true_start)
                total_delay += delay
                detected_ranges += 1
                
                # 早期检测（在异常开始前就检测到）
                if first_detection <= true_start:
                    early_detections += 1
        
        avg_delay = total_delay / max(detected_ranges, 1)
        early_detection_rate = early_detections / len(true_ranges) if true_ranges else 0.0
        
        return {
            'detection_delay': avg_delay,
            'early_detection_rate': early_detection_rate
        }
    
    def _find_best_threshold(self, y_true: np.ndarray, 
                           anomaly_scores: np.ndarray) -> Tuple[float, float]:
        """找到最佳F1分数对应的阈值"""
        
        if len(np.unique(y_true)) < 2:
            # 只有一个类别
            return np.percentile(anomaly_scores, 95), 0.0
        
        # 使用precision-recall曲线找最佳阈值
        precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)
        
        # 计算F1分数
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        
        # 找到最佳F1对应的阈值
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        
        if best_idx < len(thresholds):
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = np.percentile(anomaly_scores, 95)
        
        return best_threshold, best_f1
    
    def _find_anomaly_ranges(self, y: np.ndarray) -> List[Tuple[int, int]]:
        """找到异常区间"""
        ranges = []
        start = None
        
        for i, val in enumerate(y):
            if val == 1 and start is None:  # 异常开始
                start = i
            elif val == 0 and start is not None:  # 异常结束
                ranges.append((start, i-1))
                start = None
        
        # 处理序列末尾的异常
        if start is not None:
            ranges.append((start, len(y)-1))
        
        return ranges
    
    def _ranges_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
        """判断两个区间是否重叠"""
        return max(range1[0], range2[0]) <= min(range1[1], range2[1])
    
    def _evaluate_multiclass_pointwise(self, y_true: np.ndarray, 
                                     anomaly_scores: np.ndarray,
                                     threshold: Optional[float] = None) -> Dict[str, float]:
        """多类别逐点评估"""
        
        # 对于多类别情况，这里提供一个基础实现
        # 可以根据具体需求扩展
        
        logger.warning("多类别点级评估尚未完全实现，使用基础指标")
        
        if threshold is None:
            threshold = np.percentile(anomaly_scores, 95)
        
        # 将多类别问题简化为二分类（正常 vs 任意异常）
        y_binary = (y_true > 0).astype(int)
        y_pred = (anomaly_scores > threshold).astype(int)
        
        return {
            'threshold': threshold,
            'accuracy': accuracy_score(y_binary, y_pred),
            'precision': precision_score(y_binary, y_pred, zero_division=0),
            'recall': recall_score(y_binary, y_pred, zero_division=0),
            'f1': f1_score(y_binary, y_pred, zero_division=0)
        }
    
    def _evaluate_binary_sequence(self, y_true: np.ndarray, 
                                anomaly_scores: np.ndarray,
                                threshold: Optional[float] = None) -> Dict[str, float]:
        """二分类序列级评估"""
        
        logger.warning("序列级评估尚未完全实现，使用点级评估")
        return self._evaluate_binary_pointwise(y_true, anomaly_scores, threshold)
    
    def _evaluate_multiclass_sequence(self, y_true: np.ndarray, 
                                    anomaly_scores: np.ndarray,
                                    threshold: Optional[float] = None) -> Dict[str, float]:
        """多类别序列级评估"""
        
        logger.warning("多类别序列级评估尚未完全实现，使用基础指标")
        return self._evaluate_multiclass_pointwise(y_true, anomaly_scores, threshold)
    
    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算混淆矩阵"""
        return confusion_matrix(y_true, y_pred)
    
    def compute_roc_curve(self, y_true: np.ndarray, anomaly_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算ROC曲线"""
        from sklearn.metrics import roc_curve
        
        if len(np.unique(y_true)) < 2:
            logger.warning("只有一个类别，无法计算ROC曲线")
            return np.array([]), np.array([]), np.array([])
        
        return roc_curve(y_true, anomaly_scores)
    
    def compute_pr_curve(self, y_true: np.ndarray, anomaly_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算Precision-Recall曲线"""
        
        if len(np.unique(y_true)) < 2:
            logger.warning("只有一个类别，无法计算PR曲线")
            return np.array([]), np.array([]), np.array([])
        
        return precision_recall_curve(y_true, anomaly_scores)


class MetricsCalculator:
    """指标计算工具类"""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            anomaly_scores: np.ndarray) -> Dict[str, Any]:
        """计算所有可能的指标"""
        
        metrics = {}
        
        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp) 
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # 其他指标
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # AUC指标
        try:
            if len(np.unique(y_true)) > 1:
                metrics['auc'] = roc_auc_score(y_true, anomaly_scores)
            else:
                metrics['auc'] = np.nan
        except ValueError:
            metrics['auc'] = np.nan
        
        return metrics
    
    @staticmethod
    def print_metrics_report(metrics: Dict[str, Any], title: str = "评估结果") -> None:
        """打印格式化的指标报告"""
        
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print(f"{'='*60}")
        
        # 基础指标
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        print("📈 基础指标:")
        for metric in basic_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, float):
                    if np.isnan(value):
                        print(f"   - {metric}: N/A")
                    else:
                        print(f"   - {metric}: {value:.4f}")
                else:
                    print(f"   - {metric}: {value}")
        
        # Point-Adjusted指标
        pa_metrics = ['f1_point_adjusted', 'precision_pa', 'recall_pa']
        pa_values = [metrics.get(m) for m in pa_metrics]
        if any(v is not None for v in pa_values):
            print("\n🎯 Point-Adjusted指标:")
            for metric in pa_metrics:
                if metric in metrics and metrics[metric] is not None:
                    print(f"   - {metric}: {metrics[metric]:.4f}")
        
        # 时序指标
        ts_metrics = ['detection_delay', 'early_detection_rate']
        ts_values = [metrics.get(m) for m in ts_metrics]
        if any(v is not None for v in ts_values):
            print("\n⏰ 时序指标:")
            for metric in ts_metrics:
                if metric in metrics and metrics[metric] is not None:
                    print(f"   - {metric}: {metrics[metric]:.4f}")
        
        # 阈值信息
        if 'threshold' in metrics:
            print(f"\n🎚️  最佳阈值: {metrics['threshold']:.4f}")
        
        print(f"{'='*60}")


# 便捷函数
def evaluate_model(y_true: np.ndarray, anomaly_scores: np.ndarray, 
                   metadata: DataMetadata, threshold: Optional[float] = None) -> Dict[str, float]:
    """
    便捷的模型评估函数
    
    Args:
        y_true: 真实标签
        anomaly_scores: 异常分数
        metadata: 数据元数据
        threshold: 检测阈值
        
    Returns:
        评估结果字典
    """
    evaluator = TimeSeriesEvaluator()
    return evaluator.evaluate(y_true, anomaly_scores, metadata, threshold)


def print_evaluation_report(results: Dict[str, float], title: str = "模型评估报告") -> None:
    """打印评估报告"""
    MetricsCalculator.print_metrics_report(results, title)
