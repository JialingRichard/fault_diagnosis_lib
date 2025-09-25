"""
SKLearn 评估器集合
包含多个基于 sklearn 的评估函数
"""

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np


def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """默认评估函数 - F1分数"""
    return f1_score(y_test.flatten(), y_test_pred.flatten(), average='macro')


def f1_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """F1分数评估"""
    return f1_score(y_test.flatten(), y_test_pred.flatten(), average='macro')


def precision_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """精确率评估"""
    return precision_score(y_test.flatten(), y_test_pred.flatten(), average='macro')


def recall_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """召回率评估"""
    return recall_score(y_test.flatten(), y_test_pred.flatten(), average='macro')


def accuracy_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """准确率评估"""
    return accuracy_score(y_test.flatten(), y_test_pred.flatten())


def f1_micro_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """F1分数 - micro平均"""
    return f1_score(y_test.flatten(), y_test_pred.flatten(), average='micro')


def train_test_gap_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """训练测试性能差距评估 - 演示使用训练集数据"""
    train_f1 = f1_score(y_train.flatten(), y_train_pred.flatten(), average='macro')
    test_f1 = f1_score(y_test.flatten(), y_test_pred.flatten(), average='macro')
    
    # 返回性能差距（越小越好）
    return abs(train_f1 - test_f1)