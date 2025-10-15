"""
Plot Label Distribution evaluator.
Draws the label distribution of the selected split (passed as X_test/y_test/y_test_pred).

Return: (matplotlib Figure, total samples). Figure will be saved by EvalLoader.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def evaluate(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    """
    Draw label distribution for the provided split.
    """
    # Compute distribution
    y_test_flat = y_test.flatten()
    label_counts = Counter(y_test_flat)
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    
    # Plot bars
    labels = sorted(label_counts.keys())
    counts = [label_counts[label] for label in labels]
    
    bars = plt.bar(labels, counts, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Add counts on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Class Label', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Test Set Label Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Layout
    plt.tight_layout()
    
    # Return figure and sample count
    return (fig, len(y_test_flat))
