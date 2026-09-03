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
    python train.py --c configs/freematch_mlgcn_resnet50.yaml
    python train.py --c configs/freematch_mlgcn_resnet50.yaml --overwrite "lr=0.003 batch_size=4"
    python train.py --data_dir D:/CV/custom_muti --labeled_ratio 0.2 --backbone resnet50 ...
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


# --------------------------------------------------------------------------- args
def add_argument(parser):
    parser.add_argument("--c", type=str, default="", help="YAML config path")
    parser.add_argument("--overwrite", type=str, default="",
                        help="space-separated k=v overrides, e.g. 'lr=0.003 batch_size=4'")

    # data
    parser.add_argument("--data_dir", type=str, default=r"/root/autodl-tmp/my_semi_learing/data/custom_muti")
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

    # loss
    parser.add_argument("--loss", type=str, default="asl", choices=["asl", "bce"])
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_gamma_pos", type=float, default=1.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument("--use_class_balancing", type=str2bool, default=True)

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
    parser.add_argument("--lambda_e", type=float, default=0.005)
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
    parser.add_argument("--save_name", type=str, default="freematch_mlgcn_resnet50")
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
    args = parser.parse_args()                       # defaults
    args = _apply_config(args, args.c)                 # YAML config on top
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
    args = parser.parse_args(namespace=args)           # CLI on top of everything
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

    class_names = [train_lb.idx2cat[i] for i in range(train_lb.num_classes)]
    inp = train_lb.inp  # [C, d] GloVe embeddings or None
    return lb_loader, ulb_loader, val_loader, test_loader, class_names, inp


# --------------------------------------------------------------------------- model
def build_model_from_args(args, inp, num_classes):
    return build_model(num_classes, backbone=args.backbone, variant=args.variant,
                       pretrained=args.pretrained, t=args.gcn_t, adj_file=args.adj_file,
                       inp=inp, gcn_hidden=args.gcn_hidden, feature_dim=2048)


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
        lb_loader, ulb_loader, val_loader, test_loader, class_names, inp = build_loaders(args)
        global _lb_len
        _lb_len = len(lb_loader)
        print(f"[train] labeled={len(lb_loader.dataset)}, unlabeled={len(ulb_loader.dataset)}, "
              f"val={len(val_loader.dataset)}")

        # --- model ---
        model = build_model_from_args(args, inp, args.num_classes).to(device)
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
            device=device, use_amp=(args.amp and torch.cuda.is_available()), logger=print,
        )
        algo.class_names = class_names
        algo.num_train_iter = args.num_train_iter or (args.epochs * len(lb_loader))

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
        
# def train(args):
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     if not torch.cuda.is_available():
#         print("[train] WARNING: CUDA unavailable; training on CPU (slow).")

#     random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
#     cudnn.deterministic = True; cudnn.benchmark = True

#     save_path = os.path.join(args.save_dir, args.save_name)
#     os.makedirs(save_path, exist_ok=True)

#     from mlgcn.metrics import compute_metrics, format_report

#     # --- data ---
#     lb_loader, ulb_loader, val_loader, test_loader, class_names, inp = build_loaders(args)
#     global _lb_len
#     _lb_len = len(lb_loader)
#     print(f"[train] labeled={len(lb_loader.dataset)}, unlabeled={len(ulb_loader.dataset)}, "
#           f"val={len(val_loader.dataset)}")

#     # --- model ---
#     model = build_model_from_args(args, inp, args.num_classes).to(device)
#     print(f"[train] model={args.variant}/{args.backbone}, params={sum(p.numel() for p in model.parameters()):,}")

#     # --- losses ---
#     criterion = build_loss(args.loss, gamma_neg=args.asl_gamma_neg,
#                            gamma_pos=args.asl_gamma_pos, clip=args.asl_clip, reduction="none")
#     consistency_criterion = nn.BCEWithLogitsLoss(reduction="none")

#     # --- algorithm ---
#     algo = FreeMatchMultiLabel(
#         model, args.num_classes, criterion, consistency_criterion,
#         T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, ema_m=args.ema_m,
#         use_ema_teacher=args.use_ema_teacher, use_quantile=args.use_quantile,
#         clip_thresh=args.clip_thresh, lambda_u=args.lambda_u, lambda_e=args.lambda_e,
#         use_class_balancing=args.use_class_balancing, asl_warmup_ratio=args.asl_warmup_ratio,
#         device=device, use_amp=(args.amp and torch.cuda.is_available()), logger=print,
#     )
#     algo.class_names = class_names
#     algo.num_train_iter = args.num_train_iter or (args.epochs * len(lb_loader))

#     # --- optimizer / scheduler ---
#     opt, sched = build_optimizer(args, model)
#     try:
#         scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and torch.cuda.is_available()))
#     except TypeError:
#         scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and torch.cuda.is_available()))

#     start_epoch = 0
#     it = 0
#     best_metric = -1e9
#     no_improve = 0
#     global_step = 0

#     # resume
#     resume_path = args.load_path if args.resume else None
#     if resume_path and os.path.exists(resume_path):
#         sd = torch.load(resume_path, map_location=device, weights_only=False)
#         algo.load_state_dict(sd["algorithm"] if "algorithm" in sd else sd)
#         opt.load_state_dict(sd["optimizer"])
#         it = algo.it
#         if "best_metric" in sd:
#             best_metric = sd["best_metric"]
#         print(f"[train] resumed from {resume_path}" if resume_path != save_path + "/best.pth" else
#               f"[train] resumed from {resume_path}")

#     print("[train] start training")
#     has_ulb = len(ulb_loader.dataset) > 0 and args.lambda_u > 0
#     for epoch in range(start_epoch, args.epochs):
#         algo.model.train()
#         algo.ema_model.train()
#         if has_ulb:
#             batches = zip(lb_loader, ulb_loader)
#         else:
#             batches = ((b, None) for b in lb_loader)
#         for data_lb, data_ulb in batches:
#             if it >= algo.num_train_iter:
#                 break
#             x_lb, y_lb = data_lb
#             if data_ulb is not None:
#                 x_u_w, x_u_s = data_ulb
#                 log_dict, loss = algo.train_step(x_lb, y_lb, x_u_w, x_u_s)
#             else:
#                 log_dict, loss = algo.sup_train_step(x_lb, y_lb)

#             opt.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.unscale_(opt)
#             if args.clip_grad:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
#             scaler.step(opt)
#             scaler.update()
#             sched.step()
#             algo.update()  # EMA teacher

#             it += 1
#             global_step += 1
#             if global_step % args.num_log_iter == 0:
#                 print(f"[train] epoch {epoch} it {it} "
#                       f"| sup {log_dict['sup_loss']:.4f} unsup {log_dict['unsup_loss']:.4f} "
#                       f"ent {log_dict['ent_loss']:.4f} util {log_dict['util_ratio']:.3f} "
#                       f"lr {opt.param_groups[0]['lr']:.5f}")

#             # eval + early stopping
#             if it % args.num_eval_iter == 0:
#                 metrics, _, _ = algo.evaluate(val_loader, use_ema=True)
#                 ap = metrics.get(args.metric, metrics.get("eval/mAP", 0.0))
#                 print(format_report(metrics, "eval"))
#                 if ap > best_metric:
#                     best_metric = ap
#                     no_improve = 0
#                     torch.save({
#                         "algorithm": algo.state_dict(),
#                         "optimizer": opt.state_dict(),
#                         "scheduler": sched.state_dict(),
#                         "args": vars(args),
#                         "best_metric": best_metric,
#                     }, os.path.join(save_path, "best.pth"))
#                     print(f"[train] new best {args.metric}={ap:.4f} -> saved best.pth")
#                 else:
#                     no_improve += 1
#                     if args.patience and no_improve >= args.patience:
#                         print(f"[train] early stopping at it {it} (no improvement for {no_improve} evals)")
#                         break
#         if it >= algo.num_train_iter:
#             break
#         if args.patience and no_improve >= args.patience:
#             break

#     # --- final eval on the best model ---
#     best_path = os.path.join(save_path, "best.pth")
#     if os.path.exists(best_path):
#         print("[train] loading best model for final eval")
#         sd = torch.load(best_path, map_location=device, weights_only=False)
#         algo.load_state_dict(sd["algorithm"])
#     metrics, _, _ = algo.evaluate(val_loader, use_ema=True)
#     print(format_report(metrics, "eval"))

#     if test_loader is not None:
#         tmetrics, _, _ = algo.evaluate(test_loader, use_ema=True)
#         print(format_report(tmetrics, "test"))

#     torch.save({"algorithm": algo.state_dict(), "optimizer": opt.state_dict(),
#                 "scheduler": sched.state_dict(), "args": vars(args),
#                 "best_metric": best_metric, "val_metrics": metrics},
#                os.path.join(save_path, "final.pth"))
#     print(f"[train] done. best {args.metric}={best_metric:.4f}. saved to {save_path}")
#     return metrics, best_metric


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
