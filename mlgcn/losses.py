# -*- coding: utf-8 -*-

"""
Loss functions for semi-supervised multi-label classification.

Supervised:
    AsymmetricLoss
        hard multi-hot labels {0,1}

Unsupervised:
    AsymmetricLossSoft
        soft pseudo-labels in [0,1]

Reference:
    Alibaba-MIIL/ASL
    https://github.com/Alibaba-MIIL/ASL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. ASL for labeled / supervised samples
# ============================================================

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for supervised multi-label classification.

    Input
    -----
    logits:
        [B, C], raw model outputs.
        DO NOT apply sigmoid before passing them here.

    targets:
        [B, C], binary multi-label targets in {0, 1}.

    Output
    ------
    reduction='none':
        [B, C]

    reduction='mean':
        scalar

    reduction='sum':
        scalar
    """

    def __init__(
        self,
        gamma_neg=4.0,
        gamma_pos=0.0,
        clip=0.05,
        eps=1e-8,
        reduction="mean",
        disable_torch_grad_focal_loss=True,
    ):
        super().__init__()

        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip) if clip is not None else 0.0
        self.eps = eps
        self.reduction = reduction

        self.disable_torch_grad_focal_loss = (
            disable_torch_grad_focal_loss
        )

        # Used by FreeMatch ASL warmup.
        self._progress = 1.0


    def set_progress(self, progress):
        """
        progress:
            0 -> approximately BCE
            1 -> full ASL
        """
        self._progress = max(
            0.0,
            min(float(progress), 1.0)
        )


    def forward(self, logits, targets):

        # AMP safe
        logits = logits.float()
        targets = targets.float()

        # ----------------------------------------------------
        # ASL warmup
        # ----------------------------------------------------

        gamma_neg = self.gamma_neg * self._progress
        gamma_pos = self.gamma_pos * self._progress
        clip = self.clip * self._progress

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        xs_pos = torch.sigmoid(logits)

        xs_pos = torch.clamp(
            xs_pos,
            min=self.eps,
            max=1.0 - self.eps
        )

        xs_neg = 1.0 - xs_pos

        # ----------------------------------------------------
        # Asymmetric clipping
        #
        # official ASL:
        #
        # xs_neg = (xs_neg + clip).clamp(max=1)
        # ----------------------------------------------------

        if clip > 0:
            xs_neg = (
                xs_neg + clip
            ).clamp(max=1.0)

        # ----------------------------------------------------
        # Basic binary cross entropy
        #
        # Positive:
        #     y log(p)
        #
        # Negative:
        #     (1-y) log(p_neg)
        # ----------------------------------------------------

        loss_pos = (
            targets *
            torch.log(xs_pos.clamp(min=self.eps))
        )

        loss_neg = (
            (1.0 - targets) *
            torch.log(xs_neg.clamp(min=self.eps))
        )

        loss = loss_pos + loss_neg

        # ----------------------------------------------------
        # Asymmetric focusing
        # ----------------------------------------------------

        if gamma_neg > 0 or gamma_pos > 0:

            if self.disable_torch_grad_focal_loss:

                with torch.no_grad():

                    pt = (
                        xs_pos * targets
                        +
                        xs_neg * (1.0 - targets)
                    )

                    one_sided_gamma = (
                        gamma_pos * targets
                        +
                        gamma_neg * (1.0 - targets)
                    )

                    asymmetric_weight = torch.pow(
                        1.0 - pt,
                        one_sided_gamma
                    )

            else:

                pt = (
                    xs_pos * targets
                    +
                    xs_neg * (1.0 - targets)
                )

                one_sided_gamma = (
                    gamma_pos * targets
                    +
                    gamma_neg * (1.0 - targets)
                )

                asymmetric_weight = torch.pow(
                    1.0 - pt,
                    one_sided_gamma
                )

            loss = loss * asymmetric_weight

        # official implementation calculates negative log-likelihood
        loss = -loss

        # loss shape before reduction:
        #
        # [B, C]

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "none":
            return loss

        raise ValueError(
            f"Unsupported reduction: {self.reduction}"
        )


# ============================================================
# 2. ASL for soft pseudo labels
# ============================================================
class AsymmetricLossSoft(nn.Module):
    """
    Numerically stable ASL for soft pseudo-labels.
    targets: [B, C] continuous values in [0, 1]
    """

    def __init__(
        self,
        gamma_neg=2.0,
        gamma_pos=0.0,
        eps=1e-8,
        reduction="none",
        detach_focal_weight=True,
    ):
        super().__init__()

        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.eps = float(eps)
        self.reduction = reduction
        self.detach_focal_weight = detach_focal_weight

    def forward(self, logits, targets):

        # Always calculate ASL in fp32
        logits = logits.float()
        targets = targets.float()

        # ------------------------------------------
        # check inputs
        # ------------------------------------------
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "AsymmetricLossSoft received NaN/Inf logits."
            )

        if not torch.isfinite(targets).all():
            raise RuntimeError(
                "AsymmetricLossSoft received NaN/Inf targets."
            )

        # Avoid malformed pseudo labels
        targets = targets.clamp(0.0, 1.0)

        # ------------------------------------------
        # probability only for focal weights
        # ------------------------------------------
        prob = torch.sigmoid(logits)

        if self.detach_focal_weight:
            with torch.no_grad():

                pos_weight = (
                    1.0 - prob
                ).pow(self.gamma_pos)

                neg_weight = (
                    prob
                ).pow(self.gamma_neg)

        else:

            pos_weight = (
                1.0 - prob
            ).pow(self.gamma_pos)

            neg_weight = (
                prob
            ).pow(self.gamma_neg)

        # ==========================================
        # IMPORTANT:
        # numerically stable log probabilities
        #
        # log(sigmoid(x))
        #     = logsigmoid(x)
        #
        # log(1 - sigmoid(x))
        #     = logsigmoid(-x)
        # ==========================================

        log_pos = F.logsigmoid(logits)
        log_neg = F.logsigmoid(-logits)

        # ------------------------------------------
        # Soft ASL
        # ------------------------------------------

        pos_loss = (
            -targets
            * pos_weight
            * log_pos
        )

        neg_loss = (
            -(1.0 - targets)
            * neg_weight
            * log_neg
        )

        loss = pos_loss + neg_loss

        # ------------------------------------------
        # final safety check
        # ------------------------------------------
        if not torch.isfinite(loss).all():

            bad = ~torch.isfinite(loss)

            raise RuntimeError(
                "NaN/Inf detected in AsymmetricLossSoft: "
                f"bad={bad.sum().item()}, "
                f"logit_min={logits.min().item():.4f}, "
                f"logit_max={logits.max().item():.4f}, "
                f"target_min={targets.min().item():.6f}, "
                f"target_max={targets.max().item():.6f}"
            )

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "none":
            return loss

        raise ValueError(
            f"Unsupported reduction: {self.reduction}"
        )

# ============================================================
# 3. BCE baseline
# ============================================================

class BCEWithLogitsLoss(nn.Module):

    def __init__(
        self,
        reduction="mean",
        pos_weight=None
    ):
        super().__init__()

        self.reduction = reduction

        if pos_weight is not None:
            pos_weight = torch.as_tensor(
                pos_weight,
                dtype=torch.float32
            )

        self.register_buffer(
            "pos_weight",
            pos_weight
        )


    def set_progress(self, progress):
        # compatibility with FreeMatch
        pass


    def forward(
        self,
        logits,
        targets,
    ):

        return F.binary_cross_entropy_with_logits(
            logits.float(),
            targets.float(),
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


# ============================================================
# 4. Loss factory
# ============================================================

def build_loss(kind="asl", **kwargs):

    kind = kind.lower()

    # --------------------------------------------------------
    # supervised ASL
    # --------------------------------------------------------

    if kind in (
        "asl",
        "asl_hard",
    ):

        return AsymmetricLoss(
            gamma_neg=kwargs.get(
                "gamma_neg", 4.0
            ),
            gamma_pos=kwargs.get(
                "gamma_pos", 0.0
            ),
            clip=kwargs.get(
                "clip", 0.05
            ),
            eps=kwargs.get(
                "eps", 1e-8
            ),
            reduction=kwargs.get(
                "reduction", "mean"
            ),
            disable_torch_grad_focal_loss=kwargs.get(
                "disable_torch_grad_focal_loss",
                True
            ),
        )

    # --------------------------------------------------------
    # unsupervised / soft pseudo-label ASL
    # --------------------------------------------------------

    elif kind in (
        "asl_soft",
        "soft_asl",
    ):

        return AsymmetricLossSoft(
            gamma_neg=kwargs.get(
                "gamma_neg", 2.0
            ),
            gamma_pos=kwargs.get(
                "gamma_pos", 0.0
            ),
            eps=kwargs.get(
                "eps", 1e-8
            ),
            reduction=kwargs.get(
                "reduction", "none"
            ),
            detach_focal_weight=kwargs.get(
                "detach_focal_weight",
                True
            ),
        )

    # --------------------------------------------------------
    # BCE baseline
    # --------------------------------------------------------

    elif kind == "bce":

        return BCEWithLogitsLoss(
            reduction=kwargs.get(
                "reduction", "mean"
            ),
            pos_weight=kwargs.get(
                "pos_weight", None
            ),
        )

    else:

        raise ValueError(
            f"Unknown loss kind: {kind}. "
            "Choose from "
            "{'asl', 'asl_hard', "
            "'asl_soft', 'bce'}."
        )


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(0)

    B = 8
    C = 15

    logits = torch.randn(B, C)

    targets = (
        torch.rand(B, C) > 0.8
    ).float()

    pseudo_targets = torch.rand(B, C)

    # supervised ASL
    sup_criterion = build_loss(
        "asl",
        reduction="none"
    )

    sup_loss = sup_criterion(
        logits,
        targets
    )

    print(
        "supervised ASL:",
        sup_loss.shape,
        sup_loss.mean().item()
    )

    # unsupervised soft ASL
    unsup_criterion = build_loss(
        "asl_soft",
        reduction="none"
    )

    unsup_loss = unsup_criterion(
        logits,
        pseudo_targets
    )

    print(
        "soft ASL:",
        unsup_loss.shape,
        unsup_loss.mean().item()
    )