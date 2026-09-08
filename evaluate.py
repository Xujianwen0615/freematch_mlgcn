#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py — unified evaluation of a saved checkpoint on val / test data.

Loads a checkpoint saved by ``train.py`` (final.pth or best.pth), rebuilds the model and data
with the saved config, computes multi-label metrics (mAP, macro/micro F1, precision/recall,
hamming) and writes a CSV report.

Usage
-----
    python evaluate.py --checkpoint outputs/freematch_mlgcn_resnet50/best.pth --phase val
    python evaluate.py --checkpoint outputs/.../final.pth --phase test --config configs/....yaml
"""
import argparse
import os
import sys

import numpy as np
import torch
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

DEFAULT_MEAN = [0.2318, 0.2274, 0.2761]
DEFAULT_STD = [0.2077, 0.2220, 0.2262]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="saved checkpoint (.pth)")
    p.add_argument("--config", type=str, default="", help="config YAML (used if checkpoint lacks args)")
    p.add_argument("--phase", type=str, default="val", choices=["val", "test"])
    p.add_argument("--data_dir", type=str, default=r"dataset/custom_muti")
    p.add_argument("--num_classes", type=int, default=15)
    p.add_argument("--backbone", type=str, default="resnet101")
    p.add_argument("--variant", type=str, default="gcn")
    p.add_argument("--pretrained", type=str2bool, default=False)  # evaluate: don't re-download
    p.add_argument("--gcn_t", type=float, default=0.4)
    p.add_argument("--embedding", type=str, default="glove", choices=["glove", "clip", "random"])
    p.add_argument("--emb_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--use_ema", type=str2bool, default=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out_csv", type=str, default="")
    return p.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "yes", "on")


def main():
    args = get_args()

    # Load checkpoint; if it contains the original args, reuse them for the model.
    ckpt_dir = os.path.dirname(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", None)
    if saved_args:
        for k in ("num_classes", "backbone", "variant", "gcn_t", "img_size", "data_dir",
                  "embedding", "emb_model"):
            if hasattr(saved_args, k) and getattr(saved_args, k) is not None:
                setattr(args, k, getattr(saved_args, k))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data loader.
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])
    ds = MultiLabelDataset(args.data_dir, phase=args.phase, is_unlabeled=False,
                           labeled_ratio=1.0, weak_transform=tf)
    ds.load_class_embeddings(os.path.join(args.data_dir, "custom_glove_word2vec.pkl"))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    class_names = [ds.idx2cat[i] for i in range(ds.num_classes)]

    # Reproduce the same class-embedding source used at training time (GloVe default, or CLIP /
    # random). This keeps the first GCN layer's weight shape consistent with the checkpoint.
    if getattr(args, "embedding", "glove") == "clip":
        from mlgcn.models import load_clip_class_embeddings
        inp = load_clip_class_embeddings(class_names, model_name=args.emb_model)
    elif getattr(args, "embedding", "glove") == "random":
        inp = np.random.randn(ds.num_classes, 300).astype(np.float32)
    else:
        inp = ds.inp   # GloVe [C, 300]

    # Build model and load weights.
    # NOTE: the adjacency matrix (A/adj) is registered as a model buffer and restored from the
    # checkpoint via load_state_dict, so the adjacency source below only needs matching shape.
    model = build_model(ds.num_classes, backbone=args.backbone, variant=args.variant,
                        pretrained=args.pretrained, t=args.gcn_t,
                        adj_file=os.path.join(args.data_dir, "custom_adj.pkl"),
                        inp=inp).to(device)
    sd = ckpt["algorithm"] if "algorithm" in ckpt else ckpt
    model_sd = sd.get("ema_model" if args.use_ema else "model", sd.get("model", sd))
    model.load_state_dict(model_sd)
    model.eval()

    # Forward and evaluate.
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            out = model(x)
            all_logits.append(out.detach().cpu().numpy())
            all_targets.append(y.numpy())
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    probs = 1.0 / (1.0 + np.exp(-all_logits))

    metrics = compute_metrics(all_targets, probs, threshold=args.threshold,
                              class_names=class_names, eval_name=args.phase)
    print(format_report(metrics, args.phase))

    # CSV report
    out_csv = args.out_csv or os.path.join(ckpt_dir, f"metrics_{args.phase}.csv")
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            w.writerow([k, v])
    print(f"[evaluate] metrics saved to {out_csv}")


if __name__ == "__main__":
    main()
