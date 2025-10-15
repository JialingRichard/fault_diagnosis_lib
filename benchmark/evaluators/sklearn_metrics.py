"""
SKLearn-based evaluator collection.
Contains multiple metrics implemented via scikit-learn.
"""

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np


def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """Default evaluator - macro F1 score."""
    return f1_score(y_test.flatten(), y_test_pred.flatten(), average='macro')


def f1_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """Macro F1 score."""
    return f1_score(y_test.flatten(), y_test_pred.flatten(), average='macro', zero_division=0)


def precision_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """Macro precision."""
    return precision_score(y_test.flatten(), y_test_pred.flatten(), average='macro', zero_division=0)


def recall_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """Macro recall."""
    return recall_score(y_test.flatten(), y_test_pred.flatten(), average='macro', zero_division=0)


def accuracy_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """Accuracy."""
    return accuracy_score(y_test.flatten(), y_test_pred.flatten())


def f1_micro_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """Micro F1 score."""
    return f1_score(y_test.flatten(), y_test_pred.flatten(), average='micro')


def train_test_gap_evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """Absolute gap between train and test macro F1 (smaller is better)."""
    train_f1 = f1_score(y_train.flatten(), y_train_pred.flatten(), average='macro')
    test_f1 = f1_score(y_test.flatten(), y_test_pred.flatten(), average='macro')
    
    # Return absolute performance gap (smaller is better)
    return abs(train_f1 - test_f1)
