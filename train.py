#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py — unified semi-supervised multi-label training for FreeMatch + ML-GCN + ASL.

Pipeline
--------
  * Load labeled / unlabeled (from the labeled pool with labels hidden) / validation data.
  * Build an ML-GCN (or plain ResNet baseline) model + GloVe class embeddings.
  * Build the loss (ASL by default, or BCE) + a per-element consistency loss.
  * Wrap everything in FreeMatchMultiLabel (per-class threshold, EMA teacher, entropy reg).
  * Train with AMP on a single GPU, evaluate validation mAP periodically, and early-stop.

Usage
-----
    python train.py --c configs/freematch_mlgcn.yaml
    python train.py --c configs/freematch_mlgcn.yaml --save_name freematch_mlgcn_clip
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

# Windows console UTF-8 guard (emoji / Chinese in logs).
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data import MultiLabelDataset, collate_fn
from mlgcn import build_model, load_class_embeddings, build_loss, FreeMatchMultiLabel
from mlgcn.metrics import format_report

# Single source of image normalisation stats (from the private dataset).
DEFAULT_MEAN = [0.2318, 0.2274, 0.2761]
DEFAULT_STD = [0.2077, 0.2220, 0.2262]


# DEFAULT_MEAN = [0.2913, 0.2876, 0.3352]
# DEFAULT_STD = [0.2158, 0.2275, 0.2356]


# --------------------------------------------------------------------------- args
def add_argument(parser):
    parser.add_argument("--c", type=str, default="", help="YAML config path")
    parser.add_argument("--overwrite", type=str, default="",
                        help="space-separated k=v overrides, e.g. 'lr=0.003 batch_size=4'")

    # data
    parser.add_argument("--data_dir", type=str, default=r"dataset/custom_muti")
    parser.add_argument("--num_classes", type=int, default=15)
    parser.add_argument("--adj_file", type=str, default=None)
    parser.add_argument("--inp_file", type=str, default=None)
    parser.add_argument("--labeled_ratio", type=float, default=0.2)
    parser.add_argument("--split_mode", type=str, default="by_prefix", choices=["random", "by_prefix"])
    parser.add_argument("--train_phase", type=str, default="train")
    parser.add_argument("--val_phase", type=str, default="val")
    parser.add_argument("--test_phase", type=str, default=None)

    # model
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--variant", type=str, default="gcn", choices=["gcn", "gap", "gmp"])
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--gcn_t", type=float, default=0.4)
    parser.add_argument("--gcn_hidden", type=int, default=1024)

    # label-correlation graph & class-embedding source (ML-GCN head)
    parser.add_argument("--embedding", type=str, default="glove",
                        choices=["glove", "clip", "random"],
                        help="class-embedding source for the GCN head")
    parser.add_argument("--emb_model", type=str, default="openai/clip-vit-base-patch32",
                        help="HuggingFace CLIP model used when --embedding clip")
    parser.add_argument("--recompute_adj", type=str2bool, default=False,
                        help="rebuild label-correlation adjacency from train_anno.json instead of custom_adj.pkl")
    parser.add_argument("--adj_smoothing", type=float, default=0.0,
                        help="additive smoothing for co-occurrence when --recompute_adj")

    # loss
    parser.add_argument("--loss", type=str, default="asl", choices=["asl", "bce"])
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_gamma_pos", type=float, default=1.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--use_class_balancing", type=str2bool, default=True)
    parser.add_argument("--oversample", type=str2bool, default=False,
                        help="class-balanced over-sampling (effective-number) of labeled data")
    parser.add_argument("--cb_beta", type=float, default=0.999,
                        help="beta for effective-number class balancing (Cui et al., CVPR 2019)")

    # ---- P1: pseudo-label thresholds / consistency weighting / entropy fairness ----
    parser.add_argument("--time_p_init", type=float, default=0.3,
                        help="initial adaptive pseudo-label threshold (lower => more pseudo-labels)")
    parser.add_argument("--thresh_q", type=float, default=0.7,
                        help="quantile of unlabeled probs used to update the threshold")
    parser.add_argument("--thresh_mod_min", type=float, default=0.15,
                        help="lower clamp of the per-class threshold modulation p_model/max")
    parser.add_argument("--thresh_lo", type=float, default=0.15)
    parser.add_argument("--thresh_hi", type=float, default=0.85)
    parser.add_argument("--clip_lo", type=float, default=0.4,
                        help="lower clamp applied to time_p when clip_thresh is on")
    parser.add_argument("--clip_hi", type=float, default=0.90)
    parser.add_argument("--conf_pow", type=float, default=2.0,
                        help="confidence exponent for soft-weighting the unlabeled loss (0 = hard mask)")
    parser.add_argument("--unsup_class_balance", type=str2bool, default=True,
                        help="tail-aware re-weighting of pseudo-labels in the unlabeled loss")
    parser.add_argument("--fairness_weight", type=float, default=0.15,
                        help="weight of the fairness term in the entropy regularizer")
    parser.add_argument("--fairness_class_weight_max", type=float, default=20.0,
                        help="cap of inverse-frequency weighting inside the fairness term")

    # algorithm (FreeMatch)
    parser.add_argument("--algorithm", type=str, default="freematch_mlgcn")
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--hard_label", type=str2bool, default=False)
    parser.add_argument("--ema_p", type=float, default=0.999)
    parser.add_argument("--ema_m", type=float, default=0.999)
    parser.add_argument("--use_ema_teacher", type=str2bool, default=True)
    parser.add_argument("--use_quantile", type=str2bool, default=True)
    parser.add_argument("--clip_thresh", type=str2bool, default=True)
    parser.add_argument("--lambda_u", type=float, default=1.0)
    parser.add_argument("--lambda_e", type=float, default=0.02)
    parser.add_argument("--asl_warmup_ratio", type=float, default=0.03)

    # training
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--resize_mode", type=str, default="square", choices=["square", "aspect"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--uratio", type=int, default=3)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--num_train_iter", type=int, default=None)
    parser.add_argument("--num_eval_iter", type=int, default=1000)
    parser.add_argument("--num_log_iter", type=int, default=100)
    parser.add_argument("--amp", type=str2bool, default=True)

    # optimizer
    parser.add_argument("--optim", type=str, default="SGD", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--clip_grad", type=float, default=10.0)
    parser.add_argument("--layer_decay", type=float, default=0.5)

    # early stopping
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--metric", type=str, default="eval/mAP")

    # run
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=r"outputs")
    parser.add_argument("--save_name", type=str, default="freematch_mlgcn")
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--load_path", type=str, default="")
    parser.add_argument("--overwrite_config", type=str2bool, default=True)
    parser.add_argument("--no_amp", action="store_true")


def _apply_config(args, config_path, overrides=None):
    """Apply a YAML config dict + an optional override dict onto args (config/override wins over default)."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    if overrides:
        for k, v in overrides.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args


def make_args(config_path=None, overrides=None):
    """Build an args namespace from defaults + YAML config + override dict (no CLI parsing)."""
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args([])
    _apply_config(args, config_path, overrides)
    # resolve adj / inp defaults
    if args.adj_file is None:
        args.adj_file = os.path.join(args.data_dir, "custom_adj.pkl")
    if args.inp_file is None:
        args.inp_file = os.path.join(args.data_dir, "custom_glove_word2vec.pkl")
    return args


def get_args():
    """CLI entry: precedence is CLI > YAML/config > defaults."""
    parser = argparse.ArgumentParser(description="FreeMatch + ML-GCN multi-label semi-supervised training")
    add_argument(parser)
    args = parser.parse_args()  # defaults
    args = _apply_config(args, args.c)  # YAML config on top
    # CLI overrides (space separated k=v)
    if args.overwrite:
        for kv in args.overwrite.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                if hasattr(args, k):
                    cur = getattr(args, k)
                    try:
                        if isinstance(cur, bool):
                            setattr(args, k, v.lower() in ("1", "true", "yes", "on"))
                        elif isinstance(cur, int):
                            setattr(args, k, int(v))
                        elif isinstance(cur, float):
                            setattr(args, k, float(v))
                        else:
                            setattr(args, k, v)
                    except ValueError:
                        setattr(args, k, v)
    args = parser.parse_args(namespace=args)  # CLI on top of everything
    # resolve adj / inp defaults
    if args.adj_file is None:
        args.adj_file = os.path.join(args.data_dir, "custom_adj.pkl")
    if args.inp_file is None:
        args.inp_file = os.path.join(args.data_dir, "custom_glove_word2vec.pkl")
    if getattr(args, "no_amp", False):
        args.amp = False
    return args


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- data
def build_transform(args, kind):
    resize = transforms.Resize((args.img_size, args.img_size))
    norm = transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
    if kind == "weak":
        return transforms.Compose([resize, transforms.RandomHorizontalFlip(0.5),
                                   transforms.ToTensor(), norm])
    if kind == "strong":
        return transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), norm,
        ])
        # return transforms.Compose([
        #     transforms.RandomResizedCrop(size=args.img_size, scale=(0.3, 1.0)),
        #     transforms.RandomHorizontalFlip(0.5),
        #     transforms.RandAugment(num_ops=3, magnitude=11),
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(), norm,
        # ])
    # val / test
    return transforms.Compose([resize, transforms.ToTensor(), norm])


def build_loaders(args):
    weak_tf = build_transform(args, "weak")
    strong_tf = build_transform(args, "strong")
    val_tf = build_transform(args, "val")

    train_lb = MultiLabelDataset(args.data_dir, phase=args.train_phase, is_unlabeled=False,
                                 labeled_ratio=args.labeled_ratio, seed=args.seed,
                                 split_mode=args.split_mode, weak_transform=weak_tf)
    train_ulb = MultiLabelDataset(args.data_dir, phase=args.train_phase, is_unlabeled=True,
                                  labeled_ratio=args.labeled_ratio, seed=args.seed,
                                  split_mode=args.split_mode, weak_transform=weak_tf,
                                  strong_transform=strong_tf)
    train_lb.load_class_embeddings(args.inp_file)

    val_ds = MultiLabelDataset(args.data_dir, phase=args.val_phase, is_unlabeled=False,
                               labeled_ratio=1.0, seed=args.seed, weak_transform=val_tf)
    test_ds = None
    if args.test_phase:
        test_ds = MultiLabelDataset(args.data_dir, phase=args.test_phase, is_unlabeled=False,
                                    labeled_ratio=1.0, seed=args.seed, weak_transform=val_tf)

    dl_kwargs = dict(num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    lb_loader = DataLoader(train_lb, batch_size=args.batch_size, shuffle=True,
                           drop_last=True, **dl_kwargs)
    # If there is no unlabeled data (e.g. labeled_ratio=1.0), avoid an empty RandomSampler.
    ulb_shuffle = len(train_ulb) > 0
    ulb_loader = DataLoader(train_ulb, batch_size=args.batch_size * args.uratio, shuffle=ulb_shuffle,
                            drop_last=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False,
                            drop_last=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                             drop_last=False, **dl_kwargs) if test_ds else None

    # Per-class positive counts of the *labeled* pool, used for effective-number re-balancing
    # (P0: not the model-estimated label_hist, so the weight reflects real rarity).
    num_classes = train_lb.num_classes
    label_counts = np.zeros(num_classes, dtype=np.float32)
    for item in train_lb.img_list:
        for l in item["labels"]:
            label_counts[int(l)] += 1.0
    print(f"[data] labeled per-class counts: {list(label_counts.astype(int))}")
    # Optional class-balanced over-sampling (P0): draw rare-class images more often.
    if getattr(args, "oversample", False) and train_lb.num_classes > 0 and len(train_lb) > 0:
        w = _oversampling_sampler(train_lb, num_classes, beta=getattr(args, "cb_beta", 0.999))
        sampler = torch.utils.data.WeightedRandomSampler(
            w, num_samples=len(train_lb), replacement=True)
        lb_loader = DataLoader(train_lb, batch_size=args.batch_size, sampler=sampler,
                               drop_last=True, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=collate_fn)
        print(f"[data] class-balanced over-sampling enabled (effective-number, beta={args.cb_beta})")

    class_names = [train_lb.idx2cat[i] for i in range(train_lb.num_classes)]

    # Choose the class-embedding source for the GCN head (GloVe default, or CLIP / random).
    emb_kind = getattr(args, "embedding", "glove")
    if emb_kind == "clip":
        from mlgcn.models import load_clip_class_embeddings
        inp = load_clip_class_embeddings(
            class_names, model_name=getattr(args, "emb_model", "openai/clip-vit-base-patch32"))
        print(f"[data] loaded CLIP class embeddings: {inp.shape}")
    elif emb_kind == "random":
        inp = np.random.randn(train_lb.num_classes, 300).astype(np.float32)
        print(f"[data] random class embedding init: {inp.shape}")
    else:
        inp = train_lb.inp  # [C, d] GloVe embeddings or None
    return lb_loader, ulb_loader, val_loader, test_loader, class_names, inp, label_counts


# --------------------------------------------------------------------------- model
def _oversampling_sampler(dataset, num_classes, beta=0.999):
    """Class-balanced (effective-number) sampling weights for a multi-label dataset.

    Each image gets weight = mean of ``1 / effective_number`` over its labels, so images that
    contain rare classes are sampled much more often (Cui et al., CVPR 2019). Returns a numpy
    weight vector, one entry per dataset sample.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for it in dataset.img_list:
        for l in it["labels"]:
            counts[int(l)] += 1.0
    en = (1.0 - beta ** np.maximum(counts, 1.0)) / (1.0 - beta + 1e-7)  # effective number E_n
    inv = 1.0 / np.maximum(en, 1e-7)  # class-balanced weight = 1/E_n
    w = np.empty(len(dataset), dtype=np.float64)
    for i, it in enumerate(dataset.img_list):
        if it["labels"]:
            w[i] = inv[list(it["labels"])].mean()  # mean over the image's labels
        else:
            w[i] = 1.0
    w = w / (w.sum() + 1e-7) * len(w)  # normalise so the sampler scale is ~1x
    return w


_FEAT = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}


def build_model_from_args(args, inp, num_classes, adj_source=None):
    if adj_source is None:
        adj_source = args.adj_file
    return build_model(num_classes, backbone=args.backbone, variant=args.variant,
                       pretrained=args.pretrained, t=args.gcn_t, adj_file=adj_source,
                       inp=inp, gcn_hidden=args.gcn_hidden, feature_dim=_FEAT.get(args.backbone, 2048))


def _resolve_adj_source(args, save_path=None):
    """Return the label-correlation adjacency source."""
    if getattr(args, "recompute_adj", False):
        anno_path = os.path.join(args.data_dir, "data", "train_anno.json")
        from mlgcn.models import label_cooccurrence_from_annotations
        P = label_cooccurrence_from_annotations(
            anno_path, args.num_classes, smoothing=getattr(args, "adj_smoothing", 0.0))
        print(f"[train] rebuilt label-correlation adjacency from {anno_path}")
        if save_path:
            import pickle as _pkl
            out = os.path.join(save_path, "rebuilt_adj.pkl")
            with open(out, "wb") as f:
                _pkl.dump({"P": P}, f)
            print(f"[train] saved rebuilt adjacency to {out}")
        return P
    return getattr(args, "adj_file", None)


def build_optimizer(args, model):
    weight_decay = float(args.weight_decay)  # 强制转换
    momentum = float(args.momentum)
    lr = float(args.lr)
    if hasattr(model, "get_config_optim"):
        params = model.get_config_optim(lr, args.layer_decay)
    else:
        params = [{"params": model.parameters(), "lr": args.lr}]
    if args.optim == "SGD":
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum,
                              weight_decay=weight_decay, nesterov=True)
    elif args.optim == "Adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total_steps = args.num_train_iter or (args.epochs * _lb_len)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    return opt, sched


_lb_len = 100  # placeholder overwritten in main


# --------------------------------------------------------------------------- train
def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("[train] WARNING: CUDA unavailable; training on CPU (slow).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    save_path = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_path, exist_ok=True)

    # -------------------- 日志重定向开始 --------------------
    log_path = os.path.join(save_path, "train.log")
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout

    class Tee:
        """将输出同时发送到终端和文件"""

        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(original_stdout, log_file)

    # 即使后续出现异常，也要在退出前恢复 stdout 并关闭文件
    try:
        from mlgcn.metrics import compute_metrics, format_report

        # --- data ---
        lb_loader, ulb_loader, val_loader, test_loader, class_names, inp, label_counts = build_loaders(args)
        global _lb_len
        _lb_len = len(lb_loader)
        print(f"[train] labeled={len(lb_loader.dataset)}, unlabeled={len(ulb_loader.dataset)}, "
              f"val={len(val_loader.dataset)}")

        # # --- model ---
        # model = build_model_from_args(args, inp, args.num_classes).to(device)

        # --- model ---
        # Rebuild the label-correlation adjacency from train annotations when requested
        # (instead of a stale custom_adj.pkl), then build the model with the chosen embeddings.
        adj_source = _resolve_adj_source(args, save_path)
        model = build_model_from_args(args, inp, args.num_classes, adj_source=adj_source).to(device)

        print(f"[train] model={args.variant}/{args.backbone}, params={sum(p.numel() for p in model.parameters()):,}")

        # --- losses ---
        criterion = build_loss(args.loss, gamma_neg=args.asl_gamma_neg,
                               gamma_pos=args.asl_gamma_pos, clip=args.asl_clip, reduction="none")
        consistency_criterion = nn.BCEWithLogitsLoss(reduction="none")

        # --- algorithm ---
        algo = FreeMatchMultiLabel(
            model, args.num_classes, criterion, consistency_criterion,
            T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, ema_m=args.ema_m,
            use_ema_teacher=args.use_ema_teacher, use_quantile=args.use_quantile,
            clip_thresh=args.clip_thresh, lambda_u=args.lambda_u, lambda_e=args.lambda_e,
            use_class_balancing=args.use_class_balancing, asl_warmup_ratio=args.asl_warmup_ratio,
            label_counts=label_counts, cb_beta=args.cb_beta,
            time_p_init=args.time_p_init,
            thresh_q=args.thresh_q, thresh_mod_min=args.thresh_mod_min,
            thresh_lo=args.thresh_lo, thresh_hi=args.thresh_hi,
            clip_lo=args.clip_lo, clip_hi=args.clip_hi,
            conf_pow=args.conf_pow, unsup_class_balance=args.unsup_class_balance,
            fairness_weight=args.fairness_weight,
            fairness_class_weight_max=args.fairness_class_weight_max,
            device=device, use_amp=(args.amp and torch.cuda.is_available()), logger=print,
        )
        algo.class_names = class_names
        algo.num_train_iter = args.num_train_iter or (args.epochs * len(lb_loader))
        print(f"[train] thresh_q={algo.thresh_q}")

        # --- optimizer / scheduler ---
        opt, sched = build_optimizer(args, model)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and torch.cuda.is_available()))
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and torch.cuda.is_available()))

        start_epoch = 0
        it = 0
        best_metric = -1e9
        no_improve = 0
        global_step = 0

        # resume
        resume_path = args.load_path if args.resume else None
        if resume_path and os.path.exists(resume_path):
            sd = torch.load(resume_path, map_location=device, weights_only=False)
            algo.load_state_dict(sd["algorithm"] if "algorithm" in sd else sd)
            opt.load_state_dict(sd["optimizer"])
            it = algo.it
            if "best_metric" in sd:
                best_metric = sd["best_metric"]
            print(f"[train] resumed from {resume_path}" if resume_path != save_path + "/best.pth" else
                  f"[train] resumed from {resume_path}")

        print("[train] start training")
        has_ulb = len(ulb_loader.dataset) > 0 and args.lambda_u > 0
        for epoch in range(start_epoch, args.epochs):
            algo.model.train()
            algo.ema_model.train()
            if has_ulb:
                batches = zip(lb_loader, ulb_loader)
            else:
                batches = ((b, None) for b in lb_loader)
            for data_lb, data_ulb in batches:
                if it >= algo.num_train_iter:
                    break
                x_lb, y_lb = data_lb
                if data_ulb is not None:
                    x_u_w, x_u_s = data_ulb
                    log_dict, loss = algo.train_step(x_lb, y_lb, x_u_w, x_u_s)
                else:
                    log_dict, loss = algo.sup_train_step(x_lb, y_lb)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
                sched.step()
                algo.update()

                it += 1
                global_step += 1
                if global_step % args.num_log_iter == 0:
                    print(f"[train] epoch {epoch} it {it} "
                          f"| sup {log_dict['sup_loss']:.4f} unsup {log_dict['unsup_loss']:.4f} "
                          f"ent {log_dict['ent_loss']:.4f} util {log_dict['util_ratio']:.3f} "
                          f"lr {opt.param_groups[0]['lr']:.5f}")

                # eval + early stopping
                if it % args.num_eval_iter == 0:
                    metrics, _, _ = algo.evaluate(val_loader, use_ema=True)
                    ap = metrics.get(args.metric, metrics.get("eval/mAP", 0.0))
                    print(format_report(metrics, "eval"))
                    if ap > best_metric:
                        best_metric = ap
                        no_improve = 0
                        torch.save({
                            "algorithm": algo.state_dict(),
                            "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(),
                            "args": vars(args),
                            "best_metric": best_metric,
                        }, os.path.join(save_path, "best.pth"))
                        print(f"[train] new best {args.metric}={ap:.4f} -> saved best.pth")
                    else:
                        no_improve += 1
                        if args.patience and no_improve >= args.patience:
                            print(f"[train] early stopping at it {it} (no improvement for {no_improve} evals)")
                            break
            if it >= algo.num_train_iter:
                break
            if args.patience and no_improve >= args.patience:
                break

        # --- final eval on the best model ---
        best_path = os.path.join(save_path, "best.pth")
        if os.path.exists(best_path):
            print("[train] loading best model for final eval")
            sd = torch.load(best_path, map_location=device, weights_only=False)
            algo.load_state_dict(sd["algorithm"])
        metrics, _, _ = algo.evaluate(val_loader, use_ema=True)
        print(format_report(metrics, "eval"))

        if test_loader is not None:
            tmetrics, _, _ = algo.evaluate(test_loader, use_ema=True)
            print(format_report(tmetrics, "test"))

        torch.save({
            "algorithm": algo.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "args": vars(args),
            "best_metric": best_metric,
            "val_metrics": metrics,
        }, os.path.join(save_path, "final.pth"))
        print(f"[train] done. best {args.metric}={best_metric:.4f}. saved to {save_path}")
        return metrics, best_metric

    finally:
        # -------------------- 恢复 stdout 并关闭日志文件 --------------------
        sys.stdout = original_stdout
        log_file.close()
        print(f"[train] 完整日志已保存至 {log_path}")  # 此时 stdout 已恢复，只输出到终端


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
