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
    python run_experiments.py --only C_ssl_mlgcn_bce,D_ssl_mlgcn --out_csv results_bce_asl_summary.csv
    python run_experiments.py --only C_ssl_mlgcn_bce,C_ssl_mlgcn_bce_oversample --out_csv results_bce_oversample_summary.csv
    python run_experiments.py --only C_ssl_mlgcn,C_ssl_mlgcn_q75,C_ssl_mlgcn_q8,C_ssl_mlgcn_q9 --out_csv results_thresh_summary.csv
    python run_experiments.py --dry
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
        "labeled_ratio": 1.0, "loss": "bce", "use_class_balancing": False,
        "lambda_u": 0.0, "lambda_e": 0.0, "use_ema_teacher": False,
        "save_name": "A_sup_mlgcn_bce",
    })
    M["B_sup_mlgcn_asl"] = ("Sup-only ML-GCN + ASL                    ", {
        "epochs": 23, "labeled_ratio": 1.0, "loss": "asl", "lambda_u": 0.0, "lambda_e": 0.0,
        "use_ema_teacher": False, "save_name": "B_sup_mlgcn_asl",
    })
    M["H_weak_sup_asl"] = ("Weak-sup ML-GCN+ASL, only 20% labels     ", {
        "labeled_ratio": 0.2, "loss": "asl", "lambda_u": 0.0, "lambda_e": 0.0,
        "use_ema_teacher": False, "save_name": "H_weak_sup_mlgcn_asl_020",
    })

    M["C_ssl_mlgcn"] = ("SSL FreeMatch+ML-GCN, BCE loss           ", {
        "loss": "bce", "save_name": "C_ssl_mlgcn",
        "oversample": True
    })
    M["C_ssl_mlgcn_q75"] = ("SSL FreeMatch+ML-GCN, BCE loss           ", {
        "loss": "bce", "save_name": "C_ssl_mlgcn_q75",
        "oversample": True, "thresh_q": 0.75
    })
    M["C_ssl_mlgcn_q8"] = ("SSL FreeMatch+ML-GCN, BCE loss           ", {
        "loss": "bce", "save_name": "C_ssl_mlgcn_q8",
        "oversample": True, "thresh_q": 0.80
    })
    M["C_ssl_mlgcn_q9"] = ("SSL FreeMatch+ML-GCN, BCE loss           ", {
        "loss": "bce", "save_name": "C_ssl_mlgcn_q9",
        "oversample": True, "thresh_q": 0.90
    })
    M["D_ssl_mlgcn"] = ("SSL FreeMatch+ML-GCN, ASL loss (full)    ", {
        "save_name": "D_ssl_mlgcn",
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
    hdr = f"  {'Exp':<18}{'mAP':>9}{'Macro-F1':>10}{'Micro-F1':>10}{'Hamming':>9}{'Best-ckpt':>12}"
    print(hdr)
    rows = []
    for name, desc, metrics, best in results:
        mAP = metrics.get("eval/mAP", 0.0)
        mf1 = metrics.get("eval/macro-F1", 0.0)
        mif1 = metrics.get("eval/micro-F1", 0.0)
        ham = metrics.get("eval/hamming_loss", 0.0)
        rows.append((name, mAP, mf1, mif1, ham, f"{best:.4f}"))
        print(f"  {name:<18}{mAP:>9.4f}{mf1:>10.4f}{mif1:>10.4f}{ham:>9.4f}{best:>12.4f}")
    print("=" * 100)

    if a.out_csv:
        import csv
        with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["exp", "mAP", "macro_F1", "micro_F1", "hamming", "best_metric"])
            for name, mAP, mf1, mif1, ham, best in rows:
                w.writerow([name, mAP, mf1, mif1, ham, best])
        print(f"[run_experiments] summary CSV saved to {a.out_csv}")


if __name__ == "__main__":
    main()
