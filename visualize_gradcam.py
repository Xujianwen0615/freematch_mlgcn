#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Find hard validation classes and generate class-specific Grad-CAM diagnostics.

Example
-------
    python visualize_gradcam.py \
        --checkpoint outputs/freematch_mlgcn/best.pth \
        --phase val --top_k 3 --samples_per_group 3
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data import MultiLabelDataset, collate_fn
from mlgcn import build_model, compute_metrics, format_report
from mlgcn.gradcam import (
    DIAGNOSTIC_GROUPS,
    GradCAM,
    rank_hard_classes,
    save_cam_visualization,
    select_diagnostic_samples,
)


DEFAULT_MEAN = [0.2318, 0.2274, 0.2761]
DEFAULT_STD = [0.2077, 0.2220, 0.2262]
MODEL_DEFAULTS = {
    "data_dir": r"dataset/custom_muti",
    "num_classes": 15,
    "backbone": "resnet101",
    "variant": "gcn",
    "gcn_t": 0.4,
    "gcn_hidden": 1024,
    "embedding": "glove",
    "emb_model": "openai/clip-vit-base-patch32",
    "img_size": 224,
    "labeled_ratio": 0.2,
    "split_mode": "by_prefix",
    "seed": 42,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Rank hard multi-label classes and generate Grad-CAM diagnostics."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="", help="fallback YAML for old checkpoints")
    parser.add_argument("--phase", choices=["val", "test"], default="val")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default="")

    # Model/data values default to None so explicit CLI values can override checkpoint metadata.
    parser.add_argument("--data_dir", "--data-dir", dest="data_dir", default=None)
    parser.add_argument("--num_classes", "--num-classes", dest="num_classes", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--variant", choices=["gcn", "gap", "gmp"], default=None)
    parser.add_argument("--gcn_t", "--gcn-t", dest="gcn_t", type=float, default=None)
    parser.add_argument("--gcn_hidden", "--gcn-hidden", dest="gcn_hidden", type=int, default=None)
    parser.add_argument("--embedding", choices=["glove", "clip", "random"], default=None)
    parser.add_argument("--emb_model", "--emb-model", dest="emb_model", default=None)
    parser.add_argument("--img_size", "--img-size", dest="img_size", type=int, default=None)
    parser.add_argument("--labeled_ratio", "--labeled-ratio", dest="labeled_ratio", type=float, default=None)
    parser.add_argument("--split_mode", "--split-mode", dest="split_mode", choices=["random", "by_prefix"], default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=16)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    parser.add_argument("--use_ema", "--use-ema", dest="use_ema", type=str2bool, default=None,
                        help="default: checkpoint eval_with_ema, otherwise true")
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--class_ids", "--class-ids", dest="class_ids", default="",
                        help="comma-separated class ids; empty selects bottom-AP classes")
    parser.add_argument("--top_k", "--top-k", dest="top_k", type=int, default=3)
    parser.add_argument("--min_positive", "--min-positive", dest="min_positive", type=int, default=5)
    parser.add_argument("--samples_per_group", "--samples-per-group",
                        dest="samples_per_group", type=int, default=3)
    parser.add_argument("--groups", default=",".join(DIAGNOSTIC_GROUPS),
                        help="comma-separated subset of tp,fn,fp,boundary")
    parser.add_argument("--target_layer", "--target-layer", dest="target_layer",
                        choices=["layer3", "layer4"], default="layer4")
    parser.add_argument("--alpha", type=float, default=0.45, help="heatmap overlay opacity")
    return parser.parse_args()


def _source_value(source, name):
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def resolve_model_args(args, saved_args=None, config_values=None, inferred_values=None):
    """Resolve settings with CLI > checkpoint > YAML > state inference > defaults."""
    for name, default in MODEL_DEFAULTS.items():
        if getattr(args, name, None) is not None:
            continue
        value = _source_value(saved_args, name)
        if value is None:
            value = _source_value(config_values, name)
        if value is None:
            value = _source_value(inferred_values, name)
        setattr(args, name, default if value is None else value)
    return args


def extract_model_state(checkpoint, use_ema):
    """Normalize current, legacy, and common raw checkpoint layouts."""
    state = checkpoint.get("algorithm", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    weights_source = "checkpoint root"
    if isinstance(state, Mapping):
        preferred_key = "ema_model" if use_ema else "model"
        if preferred_key in state and isinstance(state[preferred_key], Mapping):
            state = state[preferred_key]
            weights_source = preferred_key
        elif "model" in state and isinstance(state["model"], Mapping):
            state = state["model"]
            weights_source = "model"
        elif "state_dict" in state and isinstance(state["state_dict"], Mapping):
            state = state["state_dict"]
            weights_source = "state_dict"
        elif "model_state_dict" in state and isinstance(state["model_state_dict"], Mapping):
            state = state["model_state_dict"]
            weights_source = "model_state_dict"

    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint does not contain a model state dict")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError(
            "could not identify a pure model state dict in the checkpoint; "
            "expected algorithm.model/ema_model, model, state_dict, or model_state_dict"
        )

    keys = list(state.keys())
    if keys and all(key.startswith("module.") for key in keys):
        state = {key[len("module."):]: value for key, value in state.items()}
        weights_source += " (module prefix removed)"
    return state, weights_source


def infer_model_settings(model_state):
    """Infer architecture values that are encoded unambiguously in tensor shapes."""
    inferred = {}
    if "gc1.weight" in model_state and "gc2.weight" in model_state:
        inferred["variant"] = "gcn"
        inferred["gcn_hidden"] = int(model_state["gc1.weight"].shape[1])
        if "inp" in model_state:
            inferred["num_classes"] = int(model_state["inp"].shape[0])
            inferred["embedding_dim"] = int(model_state["inp"].shape[1])
        elif "A" in model_state:
            inferred["num_classes"] = int(model_state["A"].shape[0])
            inferred["embedding_dim"] = int(model_state["gc1.weight"].shape[0])
    elif "fc.weight" in model_state:
        inferred["variant"] = "gap" if "fc.bias" in model_state else "gmp"
        inferred["num_classes"] = int(model_state["fc.weight"].shape[0])

    stage_counts = {}
    has_bottleneck = False
    for key in model_state:
        match = re.match(r"^features\.(4|5|6|7)\.(\d+)\.", key)
        if match:
            stage = int(match.group(1))
            stage_counts[stage] = max(stage_counts.get(stage, 0), int(match.group(2)) + 1)
        if re.match(r"^features\.[4-7]\.\d+\.conv3\.weight$", key):
            has_bottleneck = True
    counts = tuple(stage_counts.get(stage, 0) for stage in (4, 5, 6, 7))
    backbone_by_counts = {
        (2, 2, 2, 2): "resnet18",
        (3, 4, 6, 3): "resnet34",
        (3, 4, 6, 3, True): "resnet50",
        (3, 4, 23, 3, True): "resnet101",
    }
    if has_bottleneck:
        inferred["backbone"] = backbone_by_counts.get(counts + (True,))
    else:
        inferred["backbone"] = backbone_by_counts.get(counts)
    if inferred.get("backbone") is None:
        inferred.pop("backbone", None)
    return inferred


def validate_model_settings(args, inferred):
    for name in ("variant", "num_classes", "backbone"):
        inferred_value = inferred.get(name)
        if inferred_value is not None and getattr(args, name) != inferred_value:
            raise ValueError(
                f"resolved {name}={getattr(args, name)!r}, but checkpoint weights imply "
                f"{name}={inferred_value!r}"
            )
    if args.variant == "gcn" and int(args.gcn_hidden) != inferred.get("gcn_hidden", args.gcn_hidden):
        raise ValueError(
            f"resolved gcn_hidden={args.gcn_hidden}, but checkpoint weights imply "
            f"gcn_hidden={inferred['gcn_hidden']}"
        )


def parse_id_list(raw_value):
    if raw_value is None or raw_value == "":
        return []
    values = raw_value if isinstance(raw_value, (list, tuple)) else str(raw_value).split(",")
    result = []
    for value in values:
        class_id = int(value)
        if class_id not in result:
            result.append(class_id)
    return result


def parse_groups(raw_value):
    groups = tuple(dict.fromkeys(part.strip().lower() for part in raw_value.split(",") if part.strip()))
    unknown = sorted(set(groups) - set(DIAGNOSTIC_GROUPS))
    if not groups:
        raise ValueError("at least one diagnostic group is required")
    if unknown:
        raise ValueError(f"unknown diagnostic groups: {', '.join(unknown)}")
    return groups


def resolve_device(raw_value):
    raw_value = str(raw_value).lower()
    if raw_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw_value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if (device.type == "cuda" and device.index is not None
            and device.index >= torch.cuda.device_count()):
        raise RuntimeError(
            f"CUDA device index {device.index} is unavailable; "
            f"found {torch.cuda.device_count()} device(s)"
        )
    return device


def safe_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "item"


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint_model(args, model_state, inferred, dataset, device):
    """Build with checkpoint-shaped placeholders, then strictly restore all weights."""
    inp = None
    if args.variant == "gcn":
        embedding_dim = inferred.get("embedding_dim")
        if embedding_dim is None:
            raise ValueError("could not infer the GCN embedding dimension from checkpoint weights")
        inp = np.zeros((dataset.num_classes, embedding_dim), dtype=np.float32)

    model = build_model(
        dataset.num_classes,
        backbone=args.backbone,
        variant=args.variant,
        pretrained=False,
        t=args.gcn_t,
        adj_file=None,
        inp=inp,
        gcn_hidden=args.gcn_hidden,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


def collect_predictions(model, loader, device):
    all_logits, all_targets = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device, non_blocking=True))
            all_logits.append(logits.detach().cpu())
            all_targets.append(torch.as_tensor(targets).cpu())
    if not all_logits:
        raise ValueError("evaluation dataset is empty")
    return torch.cat(all_logits).numpy(), torch.cat(all_targets).numpy()


def target_layer_from_model(model, layer_name):
    core = model.module if hasattr(model, "module") else model
    if not hasattr(core, "features") or len(core.features) < 2:
        raise ValueError("model does not expose the expected ResNet 'features' sequence")
    stage = core.features[-1] if layer_name == "layer4" else core.features[-2]
    try:
        return stage[-1]
    except (TypeError, IndexError):
        return stage


def validate_args(args):
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if args.config and not Path(args.config).is_file():
        raise FileNotFoundError(f"config not found: {args.config}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive")
    if args.min_positive < 1:
        raise ValueError("min_positive must be at least 1")
    if args.samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if args.output_dir:
        requested_output = Path(args.output_dir)
        if requested_output.exists() and not requested_output.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {requested_output}")
        if requested_output.is_dir() and any(requested_output.iterdir()):
            raise FileExistsError(
                f"output directory is not empty: {requested_output}; choose a new directory"
            )


def choose_output_dir(args):
    if args.output_dir:
        return Path(args.output_dir)
    root = Path(args.checkpoint).resolve().parent / f"gradcam_{args.phase}"
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.target_layer}"
    candidate = root / run_name
    suffix = 2
    while candidate.exists():
        candidate = root / f"{run_name}_{suffix}"
        suffix += 1
    return candidate


def main():
    args = get_args()
    validate_args(args)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args") if isinstance(checkpoint, Mapping) else None
    config_values = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config_values = yaml.safe_load(handle) or {}
    if args.use_ema is None:
        saved_eval_with_ema = _source_value(saved_args, "eval_with_ema")
        if saved_eval_with_ema is None:
            saved_eval_with_ema = _source_value(config_values, "eval_with_ema")
        args.use_ema = True if saved_eval_with_ema is None else str2bool(saved_eval_with_ema)
    model_state, weights_source = extract_model_state(checkpoint, args.use_ema)
    inferred = infer_model_settings(model_state)
    resolve_model_args(args, saved_args, config_values, inferred)
    validate_model_settings(args, inferred)

    device = resolve_device(args.device)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])
    dataset = MultiLabelDataset(
        args.data_dir,
        phase=args.phase,
        is_unlabeled=False,
        labeled_ratio=1.0,
        weak_transform=transform,
    )
    if dataset.num_classes != int(args.num_classes):
        raise ValueError(
            f"checkpoint/config expects {args.num_classes} classes but dataset has {dataset.num_classes}"
        )
    class_names = [dataset.idx2cat[index] for index in range(dataset.num_classes)]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = load_checkpoint_model(args, model_state, inferred, dataset, device)
    del model_state
    del checkpoint
    print(
        f"[gradcam] device={device} weights={weights_source} phase={args.phase} "
        f"samples={len(dataset)}"
    )

    all_logits, all_targets = collect_predictions(model, loader, device)
    probabilities = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
    metrics = compute_metrics(
        all_targets,
        all_logits,
        threshold=args.threshold,
        class_names=class_names,
        eval_name=args.phase,
        from_logits=True,
    )
    print(format_report(metrics, args.phase))

    ranking = rank_hard_classes(
        metrics,
        class_names,
        eval_name=args.phase,
        min_positive=args.min_positive,
    )
    requested_ids = parse_id_list(args.class_ids)
    for class_id in requested_ids:
        if not 0 <= class_id < dataset.num_classes:
            raise ValueError(f"class id {class_id} is outside [0, {dataset.num_classes - 1}]")
    selected_ids = requested_ids or [
        row["class_id"] for row in ranking if row["eligible"]
    ][:args.top_k]
    if not selected_ids:
        raise ValueError(
            "no classes meet min_positive; lower --min-positive or pass --class-ids"
        )

    output_dir = choose_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "class_ranking.csv",
        ranking,
        ["difficulty_rank", "class_id", "class_name", "positive_count",
         "AP", "precision", "recall", "eligible"],
    )

    groups = parse_groups(args.groups)
    rank_by_id = {row["class_id"]: row for row in ranking}
    planned = []
    for class_id in selected_ids:
        for sample in select_diagnostic_samples(
            all_targets,
            probabilities,
            class_id,
            threshold=args.threshold,
            samples_per_group=args.samples_per_group,
            groups=groups,
            logits=all_logits,
        ):
            sample.update({
                "class_id": class_id,
                "class_name": class_names[class_id],
                "difficulty_rank": rank_by_id[class_id]["difficulty_rank"],
            })
            planned.append(sample)

    selected_text = ", ".join(
        f"{class_id}:{class_names[class_id]}" for class_id in selected_ids
    )
    print(f"[gradcam] selected classes: {selected_text}")
    print(f"[gradcam] rendering {len(planned)} diagnostic panels")

    metadata = []
    cam_cache = {}
    target_layer = target_layer_from_model(model, args.target_layer)
    resampling = getattr(Image, "Resampling", Image)
    with GradCAM(model, target_layer) as cam:
        for position, sample in enumerate(planned, start=1):
            class_id = sample["class_id"]
            index = sample["dataset_index"]
            item = dataset.img_list[index]
            file_name = item["file_name"]
            image_path = Path(args.data_dir) / "data" / args.phase / file_name
            cache_key = (class_id, index)

            if cache_key not in cam_cache:
                input_tensor, _target = dataset[index]
                cam_logits, heatmaps, raw_max, raw_range = cam(
                    input_tensor.unsqueeze(0).to(device), class_id
                )
                cam_cache[cache_key] = (
                    heatmaps[0].cpu().numpy(),
                    float(raw_max[0].cpu()),
                    float(raw_range[0].cpu()),
                    float(cam_logits[0, class_id].cpu()),
                )
            heatmap, raw_cam_max, raw_cam_range, cam_logit = cam_cache[cache_key]

            gt_ids = np.flatnonzero(all_targets[index] > 0.5).tolist()
            decision_logit = float(np.log(args.threshold) - np.log1p(-args.threshold))
            pred_ids = np.flatnonzero(all_logits[index] > decision_logit).tolist()
            gt_names = [class_names[value] for value in gt_ids]
            pred_names = [class_names[value] for value in pred_ids]

            class_dir = f"class_{class_id:02d}_{safe_name(class_names[class_id])}"
            image_stem = safe_name(Path(file_name).stem)
            output_path = output_dir / class_dir / sample["group"] / (
                f"{index:05d}_{image_stem}.png"
            )
            title = (
                f"class={class_names[class_id]} | {sample['group'].upper()} | "
                f"p={sample['probability']:.4f} | y={sample['target']} "
                f"pred={sample['prediction']} | {file_name}"
            )
            with Image.open(image_path) as image:
                base_image = image.convert("RGB").resize(
                    (args.img_size, args.img_size), resampling.BILINEAR
                )
            display_heatmap = heatmap if raw_cam_range > 1e-8 else np.zeros_like(heatmap)
            save_cam_visualization(
                base_image,
                display_heatmap,
                output_path,
                title=title,
                alpha=args.alpha,
            )

            metadata.append({
                **sample,
                "file_name": file_name,
                "image_path": str(image_path.resolve()),
                "target_logit": float(all_logits[index, class_id]),
                "cam_forward_logit": cam_logit,
                "threshold": args.threshold,
                "gt_labels": " | ".join(gt_names),
                "predicted_labels": " | ".join(pred_names),
                "target_layer": args.target_layer,
                "raw_cam_max": raw_cam_max,
                "raw_cam_range": raw_cam_range,
                "cam_is_flat": raw_cam_range <= 1e-8,
                "output_file": str(output_path.relative_to(output_dir)),
            })
            if position % 10 == 0 or position == len(planned):
                print(f"[gradcam] rendered {position}/{len(planned)}")

    metadata_fields = [
        "difficulty_rank", "class_id", "class_name", "group", "dataset_index",
        "file_name", "target", "prediction", "probability", "margin", "threshold",
        "ranking_score", "ranking_margin",
        "target_logit", "cam_forward_logit", "gt_labels", "predicted_labels",
        "target_layer", "raw_cam_max", "raw_cam_range", "cam_is_flat",
        "image_path", "output_file",
    ]
    write_csv(output_dir / "samples.csv", metadata, metadata_fields)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights_source": weights_source,
        "phase": args.phase,
        "device": str(device),
        "data_dir": str(Path(args.data_dir).resolve()),
        "num_samples": len(dataset),
        "threshold": args.threshold,
        "target_layer": args.target_layer,
        "selected_classes": [
            {"class_id": class_id, "class_name": class_names[class_id]}
            for class_id in selected_ids
        ],
        "groups": list(groups),
        "samples_per_group": args.samples_per_group,
        "rendered_panels": len(metadata),
        "mAP": metrics[f"{args.phase}/mAP"],
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[gradcam] outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
