# -*- coding: utf-8 -*-
"""Grad-CAM and hard-example helpers for multi-label classifiers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


DIAGNOSTIC_GROUPS = ("tp", "fn", "fp", "boundary")


class GradCAM:
    """Generate class-specific Grad-CAM maps from a spatial feature layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activation = None
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            shape = getattr(output, "shape", None)
            raise ValueError(
                "Grad-CAM target layer must return a [B, C, H, W] tensor; "
                f"got {type(output).__name__} with shape {shape}"
            )
        self._activation = output

    def __call__(self, input_tensor: torch.Tensor, class_id: int):
        """Return ``(logits, normalized_cam, raw_cam_max, raw_cam_range)``.

        ``normalized_cam`` has shape ``[B, H, W]`` and is independently scaled to
        ``[0, 1]`` for each image. Raw statistics are retained for diagnostics because
        normalized maps should not be compared by visual intensity.
        """
        if self._handle is None:
            raise RuntimeError("GradCAM has already been closed")
        if input_tensor.ndim != 4 or input_tensor.shape[0] == 0:
            raise ValueError(
                f"input_tensor must be a non-empty [B, C, H, W] tensor; got {tuple(input_tensor.shape)}"
            )

        self._activation = None
        detached_input = input_tensor.detach().requires_grad_(True)
        try:
            with torch.enable_grad():
                logits = self.model(detached_input)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise ValueError("model must return a [B, num_classes] logits tensor")
                if not 0 <= int(class_id) < logits.shape[1]:
                    raise ValueError(
                        f"class_id {class_id} is outside [0, {logits.shape[1] - 1}]"
                    )
                if self._activation is None:
                    raise RuntimeError("target layer was not used during the model forward pass")

                score = logits[:, int(class_id)].sum()
                gradients = torch.autograd.grad(score, self._activation)[0]
                activations = self._activation

                if not torch.isfinite(logits).all() or not torch.isfinite(gradients).all():
                    raise ValueError("non-finite logits or Grad-CAM gradients detected")

                weights = gradients.mean(dim=(-2, -1), keepdim=True)
                raw_cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
                raw_cam = F.interpolate(
                    raw_cam,
                    size=input_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if not torch.isfinite(raw_cam).all():
                    raise ValueError("non-finite Grad-CAM values detected")

                raw_min = raw_cam.amin(dim=(-2, -1), keepdim=True)
                raw_max = raw_cam.amax(dim=(-2, -1), keepdim=True)
                raw_range = raw_max - raw_min
                normalized = (raw_cam - raw_min) / raw_range.clamp_min(1e-8)
                return (
                    logits.detach(),
                    normalized[:, 0].detach(),
                    raw_max[:, 0, 0, 0].detach(),
                    raw_range[:, 0, 0, 0].detach(),
                )
        finally:
            self._activation = None

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._activation = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def rank_hard_classes(
    metrics: dict,
    class_names: Sequence[str],
    eval_name: str = "val",
    min_positive: int = 1,
):
    """Return per-class metric rows ordered from hardest to easiest.

    AP is the primary, threshold-independent criterion. Recall and class id provide
    deterministic tie-breaks. Classes below ``min_positive`` remain in the returned
    table but are marked ineligible and placed after eligible classes.
    """
    if min_positive < 1:
        raise ValueError("min_positive must be at least 1")

    rows = []
    for class_id, class_name in enumerate(class_names):
        prefix = f"{eval_name}/per_class/{class_name}"
        try:
            ap = float(metrics[f"{prefix}/AP"])
            precision = float(metrics[f"{prefix}/precision"])
            recall = float(metrics[f"{prefix}/recall"])
            positive_count = int(metrics[f"{prefix}/pos_count"])
        except KeyError as exc:
            raise KeyError(f"missing per-class metric: {exc.args[0]}") from exc

        values = np.asarray([ap, precision, recall], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite metrics for class {class_name!r}")

        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "positive_count": positive_count,
            "AP": ap,
            "precision": precision,
            "recall": recall,
            "eligible": positive_count > 0 and positive_count >= min_positive,
        })

    rows.sort(
        key=lambda row: (
            not row["eligible"],
            row["AP"],
            row["recall"],
            row["class_id"],
        )
    )
    rank = 0
    for row in rows:
        if row["eligible"]:
            rank += 1
            row["difficulty_rank"] = rank
        else:
            row["difficulty_rank"] = ""
    return rows


def select_diagnostic_samples(
    targets,
    probabilities,
    class_id: int,
    threshold: float = 0.5,
    samples_per_group: int = 3,
    groups: Iterable[str] = DIAGNOSTIC_GROUPS,
    logits=None,
):
    """Select deterministic TP/FN/FP/boundary examples for one class.

    Pass raw ``logits`` to keep ordering meaningful when sigmoid probabilities
    saturate to exactly zero or one. Probabilities are retained for display.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 2 or probabilities.ndim != 2 or targets.shape != probabilities.shape:
        raise ValueError(
            "targets and probabilities must have the same [N, C] shape; "
            f"got {targets.shape} and {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or Inf")
    if not np.isfinite(targets).all():
        raise ValueError("targets contain NaN or Inf")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if samples_per_group < 0:
        raise ValueError("samples_per_group must be non-negative")
    if not 0 <= int(class_id) < targets.shape[1]:
        raise ValueError(f"class_id {class_id} is outside [0, {targets.shape[1] - 1}]")

    requested_groups = tuple(dict.fromkeys(str(group).strip().lower() for group in groups))
    unknown = sorted(set(requested_groups) - set(DIAGNOSTIC_GROUPS))
    if unknown:
        raise ValueError(f"unknown diagnostic groups: {', '.join(unknown)}")

    class_targets = targets[:, int(class_id)] > 0.5
    class_probs = probabilities[:, int(class_id)]
    if logits is not None:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != probabilities.shape:
            raise ValueError(
                "logits must have the same [N, C] shape as probabilities; "
                f"got {logits.shape} and {probabilities.shape}"
            )
        if not np.isfinite(logits).all():
            raise ValueError("logits contain NaN or Inf")
        ranking_scores = logits[:, int(class_id)]
        decision_threshold = float(np.log(threshold) - np.log1p(-threshold))
    else:
        ranking_scores = class_probs
        decision_threshold = threshold
    class_preds = ranking_scores > decision_threshold
    indices = np.arange(targets.shape[0])

    masks = {
        "tp": class_targets & class_preds,
        "fn": class_targets & ~class_preds,
        "fp": ~class_targets & class_preds,
        "boundary": np.ones(targets.shape[0], dtype=bool),
    }

    def sort_key(group, index):
        score = float(ranking_scores[index])
        if group in ("tp", "fp"):
            return (-score, int(index))
        if group == "fn":
            return (score, int(index))
        return (abs(score - decision_threshold), int(index))

    selected = []
    for group in requested_groups:
        candidates = indices[masks[group]].tolist()
        candidates.sort(key=lambda index: sort_key(group, index))
        for index in candidates[:samples_per_group]:
            selected.append({
                "group": group,
                "dataset_index": int(index),
                "target": int(class_targets[index]),
                "prediction": int(class_preds[index]),
                "probability": float(class_probs[index]),
                "margin": float(class_probs[index] - threshold),
                "ranking_score": float(ranking_scores[index]),
                "ranking_margin": float(ranking_scores[index] - decision_threshold),
            })
    return selected


def heatmap_to_rgb(heatmap) -> np.ndarray:
    """Map a normalized heatmap to a blue-cyan-yellow-red RGB palette."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("heatmap must be a finite [H, W] array")
    x = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.rint(np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def save_cam_visualization(
    base_image: Image.Image,
    heatmap,
    output_path,
    title: str,
    alpha: float = 0.45,
):
    """Save original, heatmap, and overlay as a labelled three-panel PNG."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    base = base_image.convert("RGB")
    heatmap = np.asarray(heatmap, dtype=np.float32)
    flat_heatmap = float(np.ptp(heatmap)) <= 1e-8
    heat_rgb = Image.fromarray(heatmap_to_rgb(heatmap), mode="RGB")
    if heat_rgb.size != base.size:
        resampling = getattr(Image, "Resampling", Image)
        heat_rgb = heat_rgb.resize(base.size, resampling.BILINEAR)
    if flat_heatmap:
        heat_rgb = Image.new("RGB", base.size, "black")
        overlay = base.copy()
    else:
        overlay = Image.blend(base, heat_rgb, alpha=alpha)

    title_height = 42
    canvas = Image.new("RGB", (base.width * 3, base.height + title_height), "white")
    canvas.paste(base, (0, title_height))
    canvas.paste(heat_rgb, (base.width, title_height))
    canvas.paste(overlay, (base.width * 2, title_height))

    draw = ImageDraw.Draw(canvas)
    labels = ("original", "grad-cam", "overlay")
    for panel_index, label in enumerate(labels):
        draw.text((panel_index * base.width + 6, 4), label, fill="black")
    max_title_chars = max(1, (base.width * 3 - 12) // 7)
    clipped_title = title if len(title) <= max_title_chars else title[:max_title_chars - 3] + "..."
    draw.text((6, 22), clipped_title, fill="black")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path
