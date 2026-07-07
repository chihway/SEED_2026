"""CORN (COnsistent Rank logits via Ordinal regression, conditional training)
loss for the 4-class blend-severity task. Chosen over CORAL because CORN's
K-1 binary sub-tasks are trained conditionally (rank >= k) and don't need a
shared-weight-vector constraint to keep predictions rank-monotonic.

Reference: Shi, Cao & Raschka (2021), "Deep Neural Networks for Rank
Consistent Ordinal Regression Based on Conditional Probabilities" (CORN).
"""
import torch
import torch.nn.functional as F


def corn_loss(logits, targets, num_classes):
    """logits: (batch, num_classes - 1). targets: (batch,) int ranks in [0, num_classes - 1]."""
    losses = []
    n_examples = 0
    for k in range(num_classes - 1):
        mask = targets >= k
        if mask.sum() == 0:
            continue
        target_k = (targets[mask] > k).float()
        logit_k = logits[mask, k]
        n_examples += mask.sum().item()
        losses.append(F.binary_cross_entropy_with_logits(logit_k, target_k, reduction="sum"))
    return sum(losses) / max(n_examples, 1)


def corn_predict(logits):
    """(batch, num_classes - 1) logits -> predicted rank in [0, num_classes - 1]."""
    probas = torch.sigmoid(logits)
    cum_probas = torch.cumprod(probas, dim=1)
    return (cum_probas > 0.5).sum(dim=1)
