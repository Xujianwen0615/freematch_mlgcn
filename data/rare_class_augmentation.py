# -*- coding: utf-8 -*-
"""Rare-class-aware batch augmentation for multi-label training.

The module implements two paper-inspired, on-the-fly augmentations:

* CutMix: paste a rectangular patch from a second image and mix the two
  multi-hot targets by the exact visible area.
* ULDA/DSTIM: use TIMadd or TIMsub to move a rare-class sample toward or away
  from another sample while keeping the anchor target.

Only samples containing an automatically detected (or explicitly configured)
rare class are changed.  A donor that shares one of the anchor's rare labels is
preferred by default; donor-only positive labels are masked out of the
supervised loss so mixed-in objects are not learned as false negatives.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import torch


SUPPORTED_METHODS = {"none", "cutmix", "ulda", "cutmix_ulda"}
SUPPORTED_PAIRINGS = {"random", "shared_rare", "label_subset"}
SUPPORTED_ULDA_OPERATORS = {"random", "timadd", "timsub"}


def _parse_class_ids(value: Optional[Iterable[int] | str]) -> list[int]:
    """Accept YAML lists as well as a comma-separated CLI string."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(class_id) for class_id in value]


class RareClassBatchAugmenter:
    """Apply CutMix and/or ULDA-DSTIM to rare-class anchors in a batch.

    Parameters
    ----------
    class_counts:
        Positive counts for each class in the labeled training subset.
    method:
        ``none``, ``cutmix``, ``ulda`` or ``cutmix_ulda``.  The hybrid mode
        randomly chooses CutMix or ULDA for each selected anchor.
    rare_class_ids:
        Optional explicit class indices.  When omitted, classes at or below
        ``rare_quantile`` of the positive count distribution are selected.
    probability:
        Independent probability of augmenting each rare-class anchor.

    Notes
    -----
    The input tensors are expected to have already been normalized.  CutMix
    can operate directly in that space because it only copies pixels.  DSTIM
    is computed after undoing normalization, clipped to [0, 1], and normalized
    again to avoid invalid image ranges.
    """

    def __init__(
        self,
        class_counts: Sequence[float] | torch.Tensor,
        *,
        method: str = "none",
        rare_class_ids: Optional[Iterable[int] | str] = None,
        rare_quantile: float = 0.25,
        probability: float = 1.0,
        pairing: str = "shared_rare",
        ulda_pairing: str = "shared_rare",
        cutmix_alpha: float = 1.0,
        cutmix_preserve_anchor: bool = True,
        ulda_alpha_add: float = 0.6,
        ulda_alpha_sub: float = 0.8,
        ulda_operator: str = "random",
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.method = str(method).lower()
        self.pairing = str(pairing).lower()
        self.ulda_pairing = str(ulda_pairing).lower()
        self.ulda_operator = str(ulda_operator).lower()
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported rare augmentation method {method!r}; "
                f"choose from {sorted(SUPPORTED_METHODS)}."
            )
        if self.pairing not in SUPPORTED_PAIRINGS:
            raise ValueError(
                f"Unsupported donor pairing {pairing!r}; "
                f"choose from {sorted(SUPPORTED_PAIRINGS)}."
            )
        if self.ulda_pairing not in SUPPORTED_PAIRINGS:
            raise ValueError(
                f"Unsupported ULDA donor pairing {ulda_pairing!r}; "
                f"choose from {sorted(SUPPORTED_PAIRINGS)}."
            )
        if self.ulda_operator not in SUPPORTED_ULDA_OPERATORS:
            raise ValueError(
                f"Unsupported ULDA operator {ulda_operator!r}; "
                f"choose from {sorted(SUPPORTED_ULDA_OPERATORS)}."
            )
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("probability must be in [0, 1].")
        if not 0.0 <= float(rare_quantile) <= 1.0:
            raise ValueError("rare_quantile must be in [0, 1].")
        for name, alpha in (
            ("cutmix_alpha", cutmix_alpha),
            ("ulda_alpha_add", ulda_alpha_add),
            ("ulda_alpha_sub", ulda_alpha_sub),
        ):
            if float(alpha) <= 0:
                raise ValueError(f"{name} must be > 0.")

        counts = torch.as_tensor(class_counts, dtype=torch.float32).flatten().cpu()
        if counts.numel() == 0:
            raise ValueError("class_counts must contain at least one class.")
        if (counts < 0).any():
            raise ValueError("class_counts cannot contain negative values.")

        explicit_ids = _parse_class_ids(rare_class_ids)
        rare_mask = torch.zeros(counts.numel(), dtype=torch.bool)
        if explicit_ids:
            invalid = [idx for idx in explicit_ids if idx < 0 or idx >= counts.numel()]
            if invalid:
                raise ValueError(
                    f"rare_class_ids contains out-of-range indices {invalid}; "
                    f"num_classes={counts.numel()}."
                )
            rare_mask[torch.tensor(sorted(set(explicit_ids)), dtype=torch.long)] = True
            self.rare_count_threshold = None
        else:
            positive = counts[counts > 0]
            if positive.numel() == 0:
                # A class with no labeled anchor cannot be augmented safely.
                rare_mask[:] = False
                self.rare_count_threshold = 0.0
            else:
                threshold = torch.quantile(positive, float(rare_quantile))
                rare_mask = (counts > 0) & (counts <= threshold)
                self.rare_count_threshold = float(threshold.item())

        self.class_counts = counts
        self.rare_class_mask = rare_mask
        self.rare_class_ids = rare_mask.nonzero(as_tuple=False).flatten().tolist()
        self.rare_quantile = float(rare_quantile)
        self.probability = float(probability)
        self.cutmix_alpha = float(cutmix_alpha)
        self.cutmix_preserve_anchor = bool(cutmix_preserve_anchor)
        self.ulda_alpha_add = float(ulda_alpha_add)
        self.ulda_alpha_sub = float(ulda_alpha_sub)

        self.mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        self.std = torch.as_tensor(std, dtype=torch.float32).flatten()
        if self.mean.numel() != self.std.numel() or (self.std <= 0).any():
            raise ValueError("mean/std must have equal length and std must be positive.")

    @property
    def enabled(self) -> bool:
        return self.method != "none" and self.probability > 0 and bool(self.rare_class_ids)

    def selected_classes(self, class_names: Optional[Sequence[str]] = None) -> list[dict]:
        """Return log-friendly metadata for the selected rare classes."""
        result = []
        for class_id in self.rare_class_ids:
            name = (
                str(class_names[class_id])
                if class_names is not None and class_id < len(class_names)
                else str(class_id)
            )
            result.append(
                {"id": class_id, "name": name, "count": int(self.class_counts[class_id].item())}
            )
        return result

    @staticmethod
    def _sample_beta(alpha: float, *, device: torch.device) -> float:
        concentration = torch.tensor(float(alpha), device=device)
        return float(torch.distributions.Beta(concentration, concentration).sample().item())

    def _choose_donor(
        self,
        anchor: int,
        targets: torch.Tensor,
        rare_mask: torch.Tensor,
        pairing: str,
    ) -> Optional[int]:
        batch_size = targets.shape[0]
        if batch_size < 2:
            return None

        candidates = torch.arange(batch_size, device=targets.device)
        candidates = candidates[candidates != anchor]
        if pairing in {"shared_rare", "label_subset"}:
            anchor_rare = (targets[anchor] > 0.5) & rare_mask
            if anchor_rare.any():
                compatible = (targets[candidates][:, anchor_rare] > 0.5).any(dim=1)
                if pairing == "label_subset":
                    # A TIM donor must not introduce a positive label that is negative on the
                    # anchor, because ULDA keeps y_i unchanged.  This adapts instance-level
                    # DSTIM to real multi-label supervision without creating false negatives.
                    anchor_positive = targets[anchor] > 0.5
                    donor_positive = targets[candidates] > 0.5
                    no_extra_positive = (~anchor_positive & donor_positive).sum(dim=1) == 0
                    compatible = compatible & no_extra_positive
                if compatible.any():
                    candidates = candidates[compatible]
                elif pairing == "label_subset":
                    return None

        pick = torch.randint(candidates.numel(), (1,), device=targets.device)
        return int(candidates[pick].item())

    def _cutmix_one(
        self,
        output_images: torch.Tensor,
        output_targets: torch.Tensor,
        source_images: torch.Tensor,
        source_targets: torch.Tensor,
        mix_targets: torch.Tensor,
        mix_lam: torch.Tensor,
        anchor: int,
        donor: int,
    ) -> bool:
        _, height, width = source_images[anchor].shape
        lam = self._sample_beta(self.cutmix_alpha, device=source_images.device)
        if self.cutmix_preserve_anchor:
            # Rare-class adaptation: retain at least half of the rare anchor.
            lam = max(lam, 1.0 - lam)

        cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
        cut_h = max(1, int(round(height * cut_ratio)))
        cut_w = max(1, int(round(width * cut_ratio)))
        center_y = int(torch.randint(height, (1,), device=source_images.device).item())
        center_x = int(torch.randint(width, (1,), device=source_images.device).item())
        y1 = max(0, center_y - cut_h // 2)
        y2 = min(height, y1 + cut_h)
        x1 = max(0, center_x - cut_w // 2)
        x2 = min(width, x1 + cut_w)
        if y2 <= y1 or x2 <= x1:
            return False

        output_images[anchor, :, y1:y2, x1:x2] = source_images[donor, :, y1:y2, x1:x2]
        patch_ratio = ((y2 - y1) * (x2 - x1)) / float(height * width)
        lam_exact = 1.0 - patch_ratio
        # Keep two hard targets so ASL can compute the exact CutMix objective
        # lam * L(logits, y_a) + (1-lam) * L(logits, y_b).  Feeding a soft target
        # directly to hard-label ASL changes its focal weighting semantics.
        output_targets[anchor] = source_targets[anchor]
        mix_targets[anchor] = source_targets[donor]
        mix_lam[anchor] = lam_exact
        return True

    def _pick_ulda_operator(self, *, device: torch.device) -> str:
        if self.ulda_operator != "random":
            return self.ulda_operator
        return "timadd" if bool(torch.rand((), device=device) < 0.5) else "timsub"

    def _ulda_one(
        self,
        output_images: torch.Tensor,
        source_images: torch.Tensor,
        anchor: int,
        donor: int,
        operator: str,
    ) -> None:
        channels = source_images.shape[1]
        if channels != self.mean.numel():
            raise ValueError(
                f"image channels ({channels}) do not match mean/std ({self.mean.numel()})."
            )
        mean = self.mean.to(device=source_images.device, dtype=source_images.dtype)[:, None, None]
        std = self.std.to(device=source_images.device, dtype=source_images.dtype)[:, None, None]
        anchor_pixels = (source_images[anchor] * std + mean).clamp(0.0, 1.0)
        donor_pixels = (source_images[donor] * std + mean).clamp(0.0, 1.0)

        if operator == "timadd":
            raw = self._sample_beta(self.ulda_alpha_add, device=source_images.device)
            lam = max(raw, 1.0 - raw)  # [0.5, 1.0], ULDA Eq. (2)
            mixed = lam * anchor_pixels + (1.0 - lam) * donor_pixels
        else:
            raw = self._sample_beta(self.ulda_alpha_sub, device=source_images.device)
            lam = 0.5 + max(raw, 1.0 - raw)  # [1.0, 1.5], ULDA Eq. (3)
            mixed = lam * anchor_pixels - (1.5 - lam) * donor_pixels

        # DSTIM can leave the valid image domain, especially TIMsub.
        mixed = mixed.clamp(0.0, 1.0)
        output_images[anchor] = (mixed - mean) / std

    def __call__(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int],
    ]:
        """Return images, y_a, y_b, lambda, supervised mask and operation counts."""
        stats = {
            "eligible": 0,
            "augmented": 0,
            "cutmix": 0,
            "timadd": 0,
            "timsub": 0,
            "ignored_labels": 0,
        }
        if not self.enabled:
            mix_lam = torch.ones(targets.shape[0], dtype=targets.dtype, device=targets.device)
            loss_mask = torch.ones_like(targets)
            return images, targets, targets, mix_lam, loss_mask, stats
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B,C,H,W], got {tuple(images.shape)}")
        if targets.ndim != 2 or targets.shape[0] != images.shape[0]:
            raise ValueError(
                f"targets must have shape [B,K] with matching B; got {tuple(targets.shape)}"
            )
        if targets.shape[1] != self.class_counts.numel():
            raise ValueError(
                f"target classes ({targets.shape[1]}) != class_counts ({self.class_counts.numel()})."
            )

        source_images = images.detach()
        source_targets = targets.detach()
        output_images = images.clone()
        output_targets = targets.clone()
        mix_targets = targets.clone()
        mix_lam = torch.ones(targets.shape[0], dtype=targets.dtype, device=targets.device)
        loss_mask = torch.ones_like(targets)
        rare_mask = self.rare_class_mask.to(targets.device)
        eligible = ((targets > 0.5) & rare_mask.unsqueeze(0)).any(dim=1).nonzero(as_tuple=False)
        eligible = eligible.flatten()
        stats["eligible"] = int(eligible.numel())

        for anchor_tensor in eligible:
            if float(torch.rand((), device=images.device).item()) >= self.probability:
                continue
            operation = self.method
            if operation == "cutmix_ulda":
                operation = "cutmix" if bool(torch.rand((), device=images.device) < 0.5) else "ulda"
            anchor = int(anchor_tensor.item())
            pairing = self.pairing if operation == "cutmix" else self.ulda_pairing
            donor = self._choose_donor(anchor, source_targets, rare_mask, pairing)
            if donor is None:
                continue

            if operation == "cutmix":
                changed = self._cutmix_one(
                    output_images,
                    output_targets,
                    source_images,
                    source_targets,
                    mix_targets,
                    mix_lam,
                    anchor,
                    donor,
                )
                if not changed:
                    continue
                stats["cutmix"] += 1
            else:
                operator = self._pick_ulda_operator(device=images.device)
                self._ulda_one(output_images, source_images, anchor, donor, operator)
                # ULDA's TIMadd/TIMsub are identity-preserving: keep the anchor target.
                output_targets[anchor] = source_targets[anchor]
                # In a real multi-label image, unlike ULDA's instance pseudo-class setup, the
                # donor may contain additional annotated objects.  Ignore those donor-only
                # output entries rather than teaching them as false negatives.
                donor_only = (source_targets[donor] > 0.5) & (source_targets[anchor] <= 0.5)
                loss_mask[anchor, donor_only] = 0.0
                stats["ignored_labels"] += int(donor_only.sum().item())
                stats[operator] += 1

            stats["augmented"] += 1

        return output_images, output_targets, mix_targets, mix_lam, loss_mask, stats

