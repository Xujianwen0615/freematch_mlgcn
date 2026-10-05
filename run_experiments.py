#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_experiments.py — run the full set of baselines / ablations and summarize results.

Each experiment overrides a few keys on top of the base config and runs ``train.train``.
Metrics are collected from the ``final.pth`` written by train.py and summarized into a table
(+ optional CSV) so you can compare mAP / F1 across methods.

Usage
-----
    # run everything, write a CSV summary
    python run_experiments.py --out_csv results_summary.csv
    python run_experiments.py --out_csv aug_results_summary.csv

    # only run a subset
    python run_experiments.py --only C_ssl_mlgcn,D_ssl_mlgcn --out_csv results_bce_asl_summary.csv
    python run_experiments.py --only C_ssl_mlgcn,C_ssl_mlgcn_oversample,D_ssl_mlgcn,D_ssl_mlgcn_oversample --out_csv results_oversample_summary.csv
    python run_experiments.py --only A_sup_mlgcn_bce,C_ssl_mlgcn,H_weak_sup_bce --out_csv bce_results_summary.csv
    python run_experiments.py --only D_ssl_mlgcn,E_ssl_gap_asl,F_ssl_gmp_asl --out_csv mlgcn_results_summary.csv
    python run_experiments.py --only B_sup_mlgcn_asl,D_ssl_mlgcn,H_weak_sup_asl --out_csv asl_results_summary.csv
    python run_experiments.py --only D_ssl_mlgcn,I_tail_sampler,J_tail_cutmix,K_tail_ulda,L_tail_hybrid --out_csv rare_aug_results.csv
"""

import argparse
import gc
import os
import sys
import time

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import train as train_mod


# Each experiment: name -> (description, override-dict). Keys override the base config.
def build_matrix():
    base = {}
    M = {}
    M["A_sup_mlgcn_bce"] = ("Sup-only ML-GCN + BCE                    ", {
        "epochs": 23, "labeled_ratio": 1.0, "loss": "bce",
        "lambda_u": 0.0, "lambda_e": 0.0, "use_ema_teacher": False,
        "save_name": "A_sup_mlgcn_bce",
    })
    M["B_sup_mlgcn_asl"] = ("Sup-only ML-GCN + ASL                    ", {
        "labeled_ratio": 1.0, "loss": "asl",
        "lambda_u": 0.0, "lambda_e": 0.0, "use_ema_teacher": False,
        "save_name": "B_sup_mlgcn_asl",
    })

    M["C_ssl_mlgcn"] = ("SSL FreeMatch+ML-GCN, BCE loss           ", {
        "loss": "bce", "save_name": "C_ssl_mlgcn",
    })
    M["C_ssl_mlgcn_oversample"] = ("SSL FreeMatch+ML-GCN, BCE loss           ", {
        "loss": "bce", "save_name": "C_ssl_mlgcn_oversample",
        "oversample": True
    })

    M["D_ssl_mlgcn"] = ("SSL FreeMatch+ML-GCN, ASL loss (full)    ", {
        "save_name": "D_ssl_mlgcn",
    })
    M["I_tail_sampler"] = ("Tail baseline: rare-class sampler             ", {
        "oversample": True,
        "save_name": "I_tail_sampler",
    })
    M["J_tail_cutmix"] = ("Tail augmentation: sampler + CutMix           ", {
        "oversample": True, "rare_aug_method": "cutmix",
        "save_name": "J_tail_cutmix",
    })
    M["K_tail_ulda"] = ("Tail augmentation: sampler + ULDA-DSTIM        ", {
        "oversample": True, "rare_aug_method": "ulda",
        "save_name": "K_tail_ulda",
    })
    M["L_tail_hybrid"] = ("Tail augmentation: sampler + CutMix/DSTIM     ", {
        "oversample": True, "rare_aug_method": "cutmix_ulda",
        "save_name": "L_tail_hybrid",
    })
    M["H_weak_sup_asl"] = ("Weak-sup ML-GCN+ASL, only 20% labels     ", {
        "labeled_ratio": 0.2, "lambda_u": 0.0, "lambda_e": 0.0,
        "use_ema_teacher": False, "save_name": "H_weak_sup_mlgcn_asl_020",
    })
    M["H_weak_sup_bce"] = ("Weak-sup ML-GCN+ASL, only 20% labels     ", {
        "labeled_ratio": 0.2, "loss": "bce", "lambda_u": 0.0, "lambda_e": 0.0,
        "use_ema_teacher": False, "save_name": "H_weak_sup_mlgcn_bce_020",
    })

    M["E_ssl_gap_asl"] = ("Ablation: no graph (GAP) + ASL           ", {
        "variant": "gap", "save_name": "E_ssl_gap_asl",
    })
    M["F_ssl_gmp_asl"] = ("Ablation: no GCN head (GMP) + ASL        ", {
        "variant": "gmp", "save_name": "F_ssl_gmp_asl",
    })
    # SSL label-ratio sweep (full method)
    for ratio in (0.05, 0.1, 0.2, 0.5):
        key = f"G_ratio_{int(ratio * 100):02d}"
        M[key] = (f"Full method, labeled_ratio={ratio}              ", {
            "labeled_ratio": ratio, "save_name": f"G_ratio_{int(ratio * 100):02d}",
        })
    return M


def run_one(base_config, name, desc, overrides):
    print("\n" + "=" * 90)
    print(f"  RUN  {name}: {desc.strip()}")
    print("=" * 90)
    args = train_mod.make_args(base_config, overrides)
    metrics, best = train_mod.train(args)
    return name, desc, metrics, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/freematch_mlgcn.yaml")
    parser.add_argument("--only", default="", help="comma-separated experiment names to run")
    parser.add_argument("--skip", default="", help="comma-separated experiment names to skip")
    parser.add_argument("--out_csv", default="", help="write a summary CSV")
    parser.add_argument("--dry", action="store_true", help="print the matrix and exit")
    a = parser.parse_args()

    matrix = build_matrix()
    if a.dry:
        for k, (d, o) in matrix.items():
            print(f"  {k:<18} {d.strip()}")
        return

    only = set(x.strip() for x in a.only.split(",") if x.strip())
    skip = set(x.strip() for x in a.skip.split(",") if x.strip())
    keys = [k for k in matrix if (not only or k in only) and k not in skip]

    results = []
    for k in keys:
        d, ov = matrix[k]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append(run_one(a.base, k, d, ov))

    # Summary table
    print("\n" + "=" * 100)
    print("  RESULTS SUMMARY")
    print("=" * 100)
    hdr = (
        f"  {'Exp':<18}{'mAP':>9}{'Tail-mAP':>10}{'Tail-R':>9}"
        f"{'Macro-F1':>10}{'Micro-F1':>10}{'Hamming':>9}{'Best-ckpt':>12}"
    )
    print(hdr)
    rows = []
    for name, desc, metrics, best in results:
        mAP = metrics.get("eval/mAP", 0.0)
        tail_map = metrics.get("eval/tail-mAP", 0.0)
        tail_recall = metrics.get("eval/tail-recall", 0.0)
        mf1 = metrics.get("eval/macro-F1", 0.0)
        mif1 = metrics.get("eval/micro-F1", 0.0)
        ham = metrics.get("eval/hamming_loss", 0.0)
        rows.append((name, mAP, tail_map, tail_recall, mf1, mif1, ham, f"{best:.4f}"))
        print(
            f"  {name:<18}{mAP:>9.4f}{tail_map:>10.4f}{tail_recall:>9.4f}"
            f"{mf1:>10.4f}{mif1:>10.4f}{ham:>9.4f}{best:>12.4f}"
        )
    print("=" * 100)

    if a.out_csv:
        import csv
        with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "exp", "mAP", "tail_mAP", "tail_recall", "macro_F1",
                "micro_F1", "hamming", "best_metric",
            ])
            for name, mAP, tail_map, tail_recall, mf1, mif1, ham, best in rows:
                w.writerow([name, mAP, tail_map, tail_recall, mf1, mif1, ham, best])
        print(f"[run_experiments] summary CSV saved to {a.out_csv}")


if __name__ == "__main__":
    main()
