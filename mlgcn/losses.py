# -*- coding: utf-8 -*-
"""
Loss functions for multi-label classification.

Reference:
  - Asymmetric Loss (ASL): https://github.com/Alibaba-MIIL/ASL.git  (Ben-Baruch et al., ECCV 2020)
  - Adapted from the prior working implementation in
    D:\\freematch_mlgcn\\semilearn\\core\\criterions\\asl_loss.py

Provided
--------
  * AsymmetricLoss       — hard-label ASL (for the supervised / labeled part).
  * AsymmetricLossSoft   — soft-label ASL (for the unsupervised consistency part, no hard clipping).
  * build_loss           — factory returning any of 'asl' / 'bce' / 'bce_w' (weighted BCE).
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (hard-label version) for the labeled data.

    Core ideas:
      * Asymmetric focusing: negatives use gamma_neg (>> 1) to down-weight easy negatives,
        positives use a small gamma_pos.
      * Hard probability clipping: negatives with p < clip are fully excluded.

    Supports a progressive warmup: while ``_progress < 1`` the effective gamma_neg / clip are
    scaled linearly from 0 to their target, effectively behaving like BCE early in training.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction="mean"):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self._progress = 1.0

    def set_progress(self, progress):
        """Set training progress in [0, 1] for the ASL warmup (0 → behaves like BCE)."""
        self._progress = min(max(float(progress), 0.0), 1.0)

    def forward(self, logits, targets):
        # float32 to avoid AMP float16 underflow (eps / clamp).
        logits = logits.float()
        targets = targets.float()

        gamma_neg = self.gamma_neg * self._progress
        clip = self.clip * self._progress

        prob = torch.sigmoid(logits)
        prob = torch.clamp(prob, self.eps, 1.0 - self.eps)

        # Positive loss:  L+ = -(1-p)^{gamma+} log(p)
        pos_weight = (1.0 - prob).pow(self.gamma_pos)
        pos_loss = -targets * pos_weight * torch.log(prob)

        # Negative loss with hard clipping.
        prob_shifted = torch.clamp(prob - clip, min=0.0) if clip > 0 else prob
        neg_weight = prob_shifted.pow(gamma_neg)
        neg_loss = -(1.0 - targets) * neg_weight * torch.log(1.0 - prob_shifted + self.eps)

        loss = pos_loss + neg_loss  # [B, C]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'


class AsymmetricLossSoft(nn.Module):
    """
    Asymmetric Loss (soft / pseudo-label version) for the unsupervised consistency part.

    Unlike the hard version, no hard clipping is applied (a pseudo-label of 0.05 need not be a
    true negative) — only asymmetric focusing re-weights samples/classes.
    """

    def __init__(self, gamma_neg=2, gamma_pos=0, eps=1e-8, reduction="none"):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        prob = torch.sigmoid(logits)
        prob = torch.clamp(prob, self.eps, 1.0 - self.eps)

        pos_weight = (1.0 - prob).pow(self.gamma_pos)
        pos_loss = -targets * pos_weight * torch.log(prob)

        neg_weight = prob.pow(self.gamma_neg)
        neg_loss = -(1.0 - targets) * neg_weight * torch.log(1.0 - prob + self.eps)

        loss = pos_loss + neg_loss  # [B, C]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'  → combine with per-(sample, class) mask on top


class WeightedBCEWithLogits(nn.Module):
    """
    Weighted binary cross-entropy with logits. ``weight`` is broadcast over the class dim
    ([B, C] or [C]) and can be used to up-weight rare positive classes.
    """

    def __init__(self, reduction="mean", pos_weight=None):
        super().__init__()
        self.reduction = reduction
        self.register_buffer("pos_weight", None)
        self._pos_weight = pos_weight

    def set_progress(self, progress):
        pass  # BCE has no warmup.

    def forward(self, logits, targets, weight=None):
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.float(), targets.float(),
            pos_weight=self._pos_weight,
            reduction="none",
        )  # [B, C]
        if weight is not None:
            loss = loss * weight
            if self.reduction == "mean":
                return loss.sum() / weight.sum().clamp(min=1.0)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(kind, **kwargs):
    """Loss factory. kind in {asl, bce}."""
    if kind == "asl":
        return AsymmetricLoss(
            gamma_neg=kwargs.get("gamma_neg", 4.0),
            gamma_pos=kwargs.get("gamma_pos", 1.0),
            clip=kwargs.get("clip", 0.05),
            eps=kwargs.get("eps", 1e-8),
            reduction=kwargs.get("reduction", "mean"),
        )
    elif kind == "bce":
        return WeightedBCEWithLogits(reduction=kwargs.get("reduction", "mean"))
    else:
        raise ValueError(f"Unknown loss kind '{kind}'.")


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 15)
    y = (torch.rand(4, 15) > 0.6).float()
    for kind in ("asl", "bce"):
        l = build_loss(kind, reduction="none")
        v = l(x, y)
        print(f"{kind}: loss shape {tuple(v.shape)} mean {v.mean().item():.4f}")
