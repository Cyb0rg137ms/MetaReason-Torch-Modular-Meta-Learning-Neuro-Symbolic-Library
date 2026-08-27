"""
proto_maml.py
=============
Prototypical Networks and ANIL (Almost No Inner Loop) implementations.

References:
  - Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
    https://arxiv.org/abs/1703.05175
  - Raghu et al., "Rapid Learning or Feature Reuse? Towards Understanding
    the Effectiveness of MAML", ICLR 2020.
    https://arxiv.org/abs/1909.09157

Prototypical Networks compute class prototypes as means of support set
embeddings, then classify queries by nearest-prototype distance.

ANIL (Almost No Inner Loop) is a MAML variant that adapts only the final
classification head (not the feature extractor) during the inner loop.
This significantly reduces second-order gradient computation while
preserving most of MAML's performance.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared feature extractor
# ---------------------------------------------------------------------------

class ConvEmbedder(nn.Module):
    """
    4-layer convolutional feature extractor for image-based few-shot tasks.

    Architecture follows the standard Omniglot/miniImageNet backbone:
    4 × [Conv(64) → BN → ReLU → MaxPool]

    For tabular/vector inputs, use LinearEmbedder instead.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.encoder = nn.Sequential(
            conv_block(in_channels, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(start_dim=1)


class LinearEmbedder(nn.Module):
    """
    MLP feature extractor for tabular/vector inputs.

    Suitable for synthetic few-shot benchmarks when image data is unavailable.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Prototypical Networks
# ---------------------------------------------------------------------------

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for N-way K-shot classification.

    Algorithm:
      1. Embed all support and query samples via shared encoder.
      2. Compute class prototypes c_k = mean of K support embeddings per class.
      3. Classify queries as: argmin_k d(f(x_query), c_k)
         where d(·,·) is squared Euclidean distance.

    Loss: cross-entropy over negative squared distances (equivalent to softmax
    over distances as logits).
    """

    def __init__(self, embedder: nn.Module) -> None:
        super().__init__()
        self.embedder = embedder

    def compute_prototypes(
        self,
        support_x: torch.Tensor,  # (n_way * k_shot, d)
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:
        """
        Compute class prototypes as the mean embedding per class.

        Args:
            support_x: Support set embeddings, shape (n_way * k_shot, embed_dim).
            n_way: Number of classes.
            k_shot: Samples per class.

        Returns:
            Prototypes tensor of shape (n_way, embed_dim).
        """
        # Reshape to (n_way, k_shot, embed_dim)
        embeddings = support_x.view(n_way, k_shot, -1)
        return embeddings.mean(dim=1)  # (n_way, embed_dim)

    def forward(
        self,
        support_x: torch.Tensor,   # (n_way * k_shot, ...)
        support_y: torch.Tensor,   # (n_way * k_shot,) — class indices 0..n_way-1
        query_x: torch.Tensor,     # (n_query, ...)
        n_way: int,
        k_shot: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing query logits and episode loss.

        Returns:
            (logits, loss) where logits has shape (n_query, n_way).
        """
        # Embed support and query
        z_support = self.embedder(support_x)  # (n_way*k_shot, D)
        z_query = self.embedder(query_x)       # (n_query, D)

        # Compute prototypes in order of class index
        # Re-sort support by label to ensure correct prototype computation
        _, sorted_idx = support_y.sort()
        z_support_sorted = z_support[sorted_idx]
        prototypes = self.compute_prototypes(z_support_sorted, n_way, k_shot)  # (n_way, D)

        # Squared Euclidean distances: (n_query, n_way)
        # ||z_q - c_k||^2 = ||z_q||^2 + ||c_k||^2 - 2 * z_q · c_k
        z_q_sq = (z_query ** 2).sum(dim=1, keepdim=True)  # (n_query, 1)
        c_sq = (prototypes ** 2).sum(dim=1).unsqueeze(0)   # (1, n_way)
        cross = torch.mm(z_query, prototypes.t())           # (n_query, n_way)
        dists = z_q_sq + c_sq - 2 * cross                  # (n_query, n_way)

        # Logits: negative distance (smaller dist = higher logit)
        logits = -dists

        # Query labels: assume queries are ordered (n_way groups)
        n_query = query_x.size(0)
        query_y = torch.arange(n_way, device=query_x.device).repeat_interleave(
            n_query // n_way
        )

        loss = F.cross_entropy(logits, query_y)
        return logits, loss

    def predict(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:
        """Returns predicted class indices for each query sample."""
        logits, _ = self.forward(support_x, support_y, query_x, n_way, k_shot)
        return logits.argmax(dim=1)


# ---------------------------------------------------------------------------
# ANIL — Almost No Inner Loop
# ---------------------------------------------------------------------------

class ANILModel(nn.Module):
    """
    ANIL model: shared frozen-during-inner-loop feature extractor + task head.

    During the MAML inner loop, ONLY the head parameters are updated.
    The feature extractor is updated only during the outer (meta) loop.
    This exploits the empirical finding that MAML's performance comes
    primarily from the learned representation, not the inner-loop adaptation.
    """

    def __init__(self, embedder: nn.Module, embed_dim: int, n_way: int) -> None:
        super().__init__()
        self.embedder = embedder
        self.head = nn.Linear(embed_dim, n_way)

    def forward(self, x: torch.Tensor, head_params: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
            head_params: Optional dict with 'weight' and 'bias' for functional head forward.
                         If None, uses self.head.weight and self.head.bias.
        """
        with torch.no_grad():
            # Feature extractor runs WITHOUT gradient tracking (frozen in inner loop)
            z = self.embedder(x)
        z = z.detach()

        if head_params is not None:
            return F.linear(z, head_params["weight"], head_params["bias"])
        return self.head(z)


class ANIL:
    """
    ANIL training loop.

    Inner loop: adapts only the head (linear layer) on support data.
    Outer loop: updates the full model (embedder + head) via meta-gradient.
    """

    def __init__(self, model: ANILModel, inner_lr: float = 0.1, inner_steps: int = 1) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs `inner_steps` gradient steps on the head using support data.

        Returns adapted head parameters {weight, bias}.
        """
        head_w = self.model.head.weight.clone()
        head_b = self.model.head.bias.clone()

        for _ in range(self.inner_steps):
            logits = self.model(support_x, head_params={"weight": head_w, "bias": head_b})
            loss = loss_fn(logits, support_y)
            grads = torch.autograd.grad(loss, [head_w, head_b], create_graph=True)
            head_w = head_w - self.inner_lr * grads[0]
            head_b = head_b - self.inner_lr * grads[1]

        return {"weight": head_w, "bias": head_b}

    def meta_loss(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        loss_fn: nn.Module,
    ) -> torch.Tensor:
        """
        Computes meta-loss: adapt on support, evaluate on query.
        Gradients flow through to outer loop parameters.
        """
        adapted = self.adapt(support_x, support_y, loss_fn)
        query_logits = self.model(query_x, head_params=adapted)
        return loss_fn(query_logits, query_y)


# ---------------------------------------------------------------------------
# Few-shot accuracy utilities
# ---------------------------------------------------------------------------

def compute_episode_accuracy(
    predictions: torch.Tensor,
    query_y: torch.Tensor,
) -> float:
    """Computes accuracy for a single few-shot episode."""
    correct = (predictions == query_y).float().mean().item()
    return correct * 100.0


def compute_confidence_interval(
    accuracies: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Computes mean and 95% confidence interval for a list of episode accuracies.

    Returns (mean, half_interval) suitable for reporting as mean ± CI.
    """
    n = len(accuracies)
    mean = sum(accuracies) / n
    std = math.sqrt(sum((a - mean) ** 2 for a in accuracies) / max(n - 1, 1))
    # t-distribution approximation: use 1.96 for large n
    z = 1.96 if n >= 30 else 2.045
    half_ci = z * std / math.sqrt(n)
    return mean, half_ci
