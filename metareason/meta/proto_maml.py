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
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    @staticmethod
    def compute_prototypes(
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        n_way: int | None = None,
        k_shot: int | None = None,
    ) -> torch.Tensor:
        """Mean embedding per class, for labels in ``0..n_way-1`` in any order.

        Works for unbalanced shots; raises if a class has no support example.
        Legacy call ``compute_prototypes(z, n_way=.., k_shot=..)`` (no labels)
        assumes ``z`` is grouped by class in blocks of ``k_shot``.
        """
        if n_way is None:
            raise TypeError("n_way is required")
        if labels is None:
            if k_shot is None or embeddings.size(0) != n_way * k_shot:
                raise ValueError("without labels, embeddings must be n_way blocks of k_shot rows")
            return embeddings.view(n_way, k_shot, -1).mean(dim=1)
        if labels.numel() == 0 or int(labels.min()) < 0 or int(labels.max()) >= n_way:
            raise ValueError("support labels must lie in 0..n_way-1")
        counts = torch.bincount(labels, minlength=n_way).to(embeddings.dtype)
        if bool((counts == 0).any()):
            raise ValueError("every class 0..n_way-1 needs at least one support example")
        sums = torch.zeros(n_way, embeddings.size(1), dtype=embeddings.dtype, device=embeddings.device)
        sums.index_add_(0, labels, embeddings)
        return sums / counts.unsqueeze(1)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_way: int,
        k_shot: int | None = None,
        query_y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(logits, loss)``; logits are negative squared distances.

        ``k_shot`` is accepted for backward compatibility and ignored. ``loss``
        is ``None`` when ``query_y`` is not given and cannot be inferred; the
        1.0 behaviour (queries grouped by class) is kept with a warning.
        """
        z_support = self.embedder(support_x)
        z_query = self.embedder(query_x)
        prototypes = self.compute_prototypes(z_support, support_y, n_way)
        logits = -torch.cdist(z_query, prototypes).pow(2)
        if query_y is None and query_x.size(0) % n_way == 0:
            # Legacy 1.0 behaviour: queries assumed grouped by class.
            warnings.warn(
                "PrototypicalNetwork: pass query_y; assuming queries are grouped by class",
                DeprecationWarning,
                stacklevel=2,
            )
            query_y = torch.arange(n_way, device=query_x.device).repeat_interleave(query_x.size(0) // n_way)
        loss = F.cross_entropy(logits, query_y) if query_y is not None else None
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

    def forward(self, x: torch.Tensor, head_params: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
            head_params: Optional dict with 'weight' and 'bias' for functional head forward.
                         If None, uses self.head.weight and self.head.bias.
        """
        # The body is not adapted in the inner loop (only the head is), but it
        # *is* trained by the outer loop, so gradients must flow through it.
        z = self.embedder(x)

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
    ) -> dict[str, torch.Tensor]:
        """
        Runs `inner_steps` gradient steps on the head using support data.

        Returns adapted head parameters {weight, bias}.
        """
        head_w: torch.Tensor = self.model.head.weight
        head_b: torch.Tensor = self.model.head.bias
        z = self.model.embedder(support_x)  # computed once: the body is fixed in the inner loop
        for _ in range(self.inner_steps):
            logits = F.linear(z, head_w, head_b)
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


def compute_confidence_interval(accuracies: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Mean and Student-t half-width of the confidence interval.

    Uses ``scipy.stats.t`` when SciPy is installed, otherwise the normal
    approximation (exact enough for the usual 600+ episodes).
    """
    n = len(accuracies)
    if n == 0:
        raise ValueError("need at least one accuracy")
    mean = sum(accuracies) / n
    if n == 1:
        return mean, float("nan")
    std = math.sqrt(sum((a - mean) ** 2 for a in accuracies) / (n - 1))
    try:
        from scipy.stats import t as student_t

        q = float(student_t.ppf(0.5 + confidence / 2, df=n - 1))
    except ImportError:  # pragma: no cover
        q = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.5 + confidence / 2)))
    return mean, q * std / math.sqrt(n)
