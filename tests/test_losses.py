import pytest
import sys, torch
sys.path.insert(0, 'src')
from losses.multi_task_loss import MultiTaskLoss, SupConLoss, BoundaryLoss

def test_multitask_loss_ce_only():
    criterion = MultiTaskLoss(lambda_contrastive=0.0, lambda_boundary=0.0)
    logits = torch.randn(8, 150)
    hidden = torch.randn(8, 768)
    labels = torch.randint(0, 150, (8,))
    out = criterion(logits, hidden, labels)
    assert 'loss' in out
    assert out['loss'].item() > 0

def test_multitask_loss_with_supcon():
    criterion = MultiTaskLoss(lambda_contrastive=0.3, lambda_boundary=0.0)
    logits = torch.randn(8, 150)
    hidden = torch.randn(8, 768)
    # Use repeated labels to guarantee positive pairs exist for SupCon loss
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out = criterion(logits, hidden, labels)
    assert out['loss_contrastive'] > 0

def test_boundary_loss_no_oos():
    loss_fn = BoundaryLoss()
    hidden = torch.randn(4, 768)
    labels = torch.tensor([0, 1, 2, 3])  # 全 in-scope
    proto_bank = torch.randn(4, 1, 768)
    loss = loss_fn(hidden, labels, proto_bank, oos_label=150)
    assert loss.item() == 0.0  # 无OOS时boundary loss为0
