"""
Models Module for Time Series Anomaly Detection Benchmark
=========================================================

This module contains all the model implementations for the benchmark framework.

Available Models:
- IsolationForestModel: Isolation Forest based anomaly detection
- LSTMAutoEncoderModel: LSTM AutoEncoder based anomaly detection
"""

from .base_model import BaseModel, DataMetadata, ModelFactory
from .iforest import IsolationForestModel
from .lstm_ae import LSTMAutoEncoderModel

__all__ = [
    'BaseModel',
    'DataMetadata', 
    'ModelFactory',
    'IsolationForestModel',
    'LSTMAutoEncoderModel'
]

# 所有模型会在导入时自动注册到ModelFactory
