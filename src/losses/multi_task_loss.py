"""
Loss Functions for LLM-Enhanced OOS Detection
Implements multi-task losses: CE, SupCon, Boundary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossEntropyLoss(nn.Module):
    """Standard Cross-Entropy Loss"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss

        Args:
            logits: Predicted logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        return F.cross_entropy(logits, labels, reduction=self.reduction)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    Reference: https://arxiv.org/abs/2004.11362
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss

        Args:
            features: Hidden features [batch_size, hidden_dim]
            labels: Ground truth labels [batch_size]
            mask: Optional mask for valid samples [batch_size]

        Returns:
            Loss value
        """
        device = features.device
        batch_size = features.shape[0]

        if mask is None:
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # [batch_size, batch_size]

        # Create label mask: 1 if same label, 0 otherwise
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)  # [batch_size, batch_size]

        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(label_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        # Apply mask
        label_mask = label_mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive samples
        mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-8)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[mask].mean()

        return loss


class BoundaryLoss(nn.Module):
    """
    Boundary-Aware Loss for OOS Detection (UCRID L_Boundary)
    Forces OOS embeddings to be at least Δ away from all ID prototypes (Euclidean).
    L_Boundary = mean_{x in OOS} max(0, Δ - d(h_x, p_{c*}))
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        prototype_bank: torch.Tensor,
        oos_label: int = 150
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, hidden_dim]
            labels: [batch_size]
            prototype_bank: [num_intents, 1, hidden_dim]  (mean prototypes)
            oos_label: OOS class index
        """
        oos_mask = (labels == oos_label)
        if not oos_mask.any():
            return torch.tensor(0.0, device=hidden_states.device)

        oos_hidden = hidden_states[oos_mask]  # [num_oos, hidden_dim]

        # Euclidean distance to each intent prototype
        # prototype_bank: [num_intents, 1, hidden_dim] -> squeeze to [num_intents, hidden_dim]
        protos = prototype_bank.squeeze(1)  # [num_intents, hidden_dim]
        # [num_oos, num_intents]
        dists = torch.cdist(oos_hidden, protos, p=2)
        d_min = dists.min(dim=1)[0]  # [num_oos]

        return F.relu(self.margin - d_min).mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss combining CE, SupCon, and Boundary losses
    """

    def __init__(
        self,
        lambda_contrastive: float = 0.3,
        lambda_boundary: float = 0.1,
        temperature: float = 0.07,
        margin: float = 1.0,
        oos_label: int = 150
    ):
        super().__init__()

        self.lambda_contrastive = lambda_contrastive
        self.lambda_boundary = lambda_boundary
        self.oos_label = oos_label

        self.ce_loss = CrossEntropyLoss()
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.boundary_loss = BoundaryLoss(margin=margin)

    def forward(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        prototype_bank: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute multi-task loss

        Args:
            logits: Predicted logits [batch_size, num_classes]
            hidden_states: Hidden states [batch_size, hidden_dim]
            labels: Ground truth labels [batch_size]
            prototype_bank: Mean prototype bank [num_intents, 1, hidden_dim] (optional)

        Returns:
            Dictionary containing individual losses and total loss
        """
        # Cross-entropy loss
        loss_ce = self.ce_loss(logits, labels)

        # Contrastive loss
        loss_contrastive = self.supcon_loss(hidden_states, labels)

        # Boundary loss (only if prototype bank is provided)
        if prototype_bank is not None and self.lambda_boundary > 0:
            loss_boundary = self.boundary_loss(
                hidden_states, labels, prototype_bank, self.oos_label
            )
        else:
            loss_boundary = torch.tensor(0.0, device=logits.device)

        # Total loss
        total_loss = (
            loss_ce +
            self.lambda_contrastive * loss_contrastive +
            self.lambda_boundary * loss_boundary
        )

        return {
            'loss': total_loss,
            'loss_ce': loss_ce.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss_boundary': loss_boundary.item() if isinstance(loss_boundary, torch.Tensor) else 0.0
        }


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")

    batch_size = 8
    num_classes = 151
    hidden_dim = 768

    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    hidden_states = torch.randn(batch_size, hidden_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test CE loss
    ce_loss = CrossEntropyLoss()
    loss_ce = ce_loss(logits, labels)
    print(f"CE Loss: {loss_ce.item():.4f}")

    # Test SupCon loss
    supcon_loss = SupConLoss(temperature=0.07)
    loss_supcon = supcon_loss(hidden_states, labels)
    print(f"SupCon Loss: {loss_supcon.item():.4f}")

    # Test Boundary loss
    num_intents = 150
    num_sub_prototypes = 1
    prototype_bank = torch.randn(num_intents, num_sub_prototypes, hidden_dim)

    boundary_loss = BoundaryLoss(margin=1.0)
    loss_boundary = boundary_loss(hidden_states, labels, prototype_bank, oos_label=150)
    print(f"Boundary Loss: {loss_boundary.item():.4f}")

    # Test Multi-task loss
    multi_loss = MultiTaskLoss(
        lambda_contrastive=0.3,
        lambda_boundary=0.1,
        temperature=0.07,
        margin=1.0
    )

    losses = multi_loss(logits, hidden_states, labels, prototype_bank)
    print(f"\nMulti-Task Loss:")
    print(f"  Total: {losses['loss'].item():.4f}")
    print(f"  CE: {losses['loss_ce']:.4f}")
    print(f"  Contrastive: {losses['loss_contrastive']:.4f}")
    print(f"  Boundary: {losses['loss_boundary']:.4f}")
