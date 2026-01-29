"""
Metrics and evaluation utilities for model performance assessment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = 'binary'
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        average: Averaging method for multi-class ('binary', 'micro', 'macro', 'weighted')

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Add ROC AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            pass

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None
):
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of target class names
    """
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    figsize: tuple = (8, 6)
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of label names
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=labels if labels else 'auto',
        yticklabels=labels if labels else 'auto'
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate business-specific metrics for financial use cases.

    Args:
        y_true: True labels (1 = fraud/default, 0 = legitimate/no default)
        y_pred: Predicted labels
        amounts: Optional transaction/loan amounts

    Returns:
        Dictionary of business metrics
    """
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
    }

    # Calculate financial impact if amounts provided
    if amounts is not None:
        # Amount saved by catching fraud/defaults
        caught_amount = amounts[(y_true == 1) & (y_pred == 1)].sum()
        # Amount lost due to missed fraud/defaults
        missed_amount = amounts[(y_true == 1) & (y_pred == 0)].sum()
        # Amount blocked incorrectly (false positives)
        blocked_amount = amounts[(y_true == 0) & (y_pred == 1)].sum()

        metrics['amount_caught'] = float(caught_amount)
        metrics['amount_missed'] = float(missed_amount)
        metrics['amount_blocked'] = float(blocked_amount)
        metrics['recovery_rate'] = float(caught_amount / (caught_amount + missed_amount)) if (caught_amount + missed_amount) > 0 else 0

    return metrics


def calculate_model_performance_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    amounts: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Generate a comprehensive model performance summary.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        amounts: Optional transaction/loan amounts

    Returns:
        DataFrame with all metrics
    """
    # Standard classification metrics
    standard_metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)

    # Business metrics
    business_metrics = calculate_business_metrics(y_true, y_pred, amounts)

    # Combine all metrics
    all_metrics = {**standard_metrics, **business_metrics}

    # Remove confusion matrix for cleaner display
    cm = all_metrics.pop('confusion_matrix', None)

    # Convert to DataFrame
    df = pd.DataFrame([all_metrics]).T
    df.columns = ['Value']

    return df


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample data
    y_true = np.random.randint(0, 2, 1000)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(1000, 100, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    y_pred_proba = np.random.random(1000)
    amounts = np.random.uniform(100, 10000, 1000)

    # Calculate metrics
    print("Classification Metrics:")
    print("=" * 60)
    metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value:.4f}")

    print("\n\nBusiness Metrics:")
    print("=" * 60)
    business_metrics = calculate_business_metrics(y_true, y_pred, amounts)
    for key, value in business_metrics.items():
        print(f"{key}: {value}")

    print("\n\nPerformance Summary:")
    print("=" * 60)
    summary = calculate_model_performance_summary(y_true, y_pred, y_pred_proba, amounts)
    print(summary)
