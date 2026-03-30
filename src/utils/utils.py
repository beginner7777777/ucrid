"""
Utility Functions for Training and Evaluation
"""

import os
import random
import numpy as np
import torch
import json
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    **kwargs
):
    """
    Save model checkpoint

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load model checkpoint

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        optimizer: Optional optimizer to load state
        device: Device to load model to

    Returns:
        Dictionary containing checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")

    return checkpoint


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    oos_label: int = 150
) -> Dict[str, float]:
    """
    Compute evaluation metrics

    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        oos_label: Label ID for OOS class

    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(labels, predictions)

    # Binary OOS detection metrics
    is_oos_pred = (predictions == oos_label).astype(int)
    is_oos_true = (labels == oos_label).astype(int)

    oos_precision = precision_score(is_oos_true, is_oos_pred, zero_division=0)
    oos_recall = recall_score(is_oos_true, is_oos_pred, zero_division=0)
    oos_f1 = f1_score(is_oos_true, is_oos_pred, zero_division=0)

    # ID accuracy (accuracy on in-domain samples only)
    id_mask = (labels != oos_label)
    if id_mask.sum() > 0:
        id_accuracy = accuracy_score(labels[id_mask], predictions[id_mask])
    else:
        id_accuracy = 0.0

    metrics = {
        'accuracy': accuracy,
        'id_accuracy': id_accuracy,
        'oos_precision': oos_precision,
        'oos_recall': oos_recall,
        'oos_f1': oos_f1
    }

    return metrics


def compute_oos_metrics_detailed(
    predictions: np.ndarray,
    labels: np.ndarray,
    oos_types: np.ndarray = None,
    oos_label: int = 150
) -> Dict[str, float]:
    """
    Compute detailed OOS metrics including near/far domain

    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        oos_types: OOS type labels ('near' or 'far'), optional
        oos_label: Label ID for OOS class

    Returns:
        Dictionary of detailed metrics
    """
    metrics = compute_metrics(predictions, labels, oos_label)

    # If OOS types are provided, compute near/far domain metrics
    if oos_types is not None:
        oos_mask = (labels == oos_label)

        if oos_mask.sum() > 0:
            oos_predictions = predictions[oos_mask]
            oos_labels = labels[oos_mask]
            oos_types_filtered = oos_types[oos_mask]

            # Near domain OOS recall
            near_mask = (oos_types_filtered == 'near')
            if near_mask.sum() > 0:
                near_correct = ((oos_predictions[near_mask] == oos_label).sum())
                near_recall = near_correct / near_mask.sum()
                metrics['near_oos_recall'] = near_recall

            # Far domain OOS recall
            far_mask = (oos_types_filtered == 'far')
            if far_mask.sum() > 0:
                far_correct = ((oos_predictions[far_mask] == oos_label).sum())
                far_recall = far_correct / far_mask.sum()
                metrics['far_oos_recall'] = far_recall

    return metrics


def save_results(results: Dict[str, Any], save_path: str):
    """
    Save results to JSON file

    Args:
        results: Results dictionary
        save_path: Path to save results
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {save_path}")


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for printing (e.g., "Train", "Val", "Test")
    """
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print("-" * 50)


def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get torch device

    Args:
        gpu_id: GPU ID to use

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether to minimize or maximize metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training

        Args:
            score: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test set_seed
    set_seed(42)
    print("✓ Seed set")

    # Test metrics
    predictions = np.array([0, 1, 2, 150, 150, 1, 2, 150])
    labels = np.array([0, 1, 2, 150, 1, 1, 2, 150])

    metrics = compute_metrics(predictions, labels, oos_label=150)
    print_metrics(metrics, "Test")

    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='min')
    scores = [0.5, 0.4, 0.45, 0.46, 0.47, 0.48]

    print("\nTesting Early Stopping:")
    for i, score in enumerate(scores):
        should_stop = early_stopping(score)
        print(f"  Epoch {i+1}: score={score:.2f}, stop={should_stop}")
