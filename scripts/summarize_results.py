#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summarize_results.py — parse every experiment's train.log and report the final metrics.

Looks under <root>/outputs/*/train.log, extracts the LAST evaluation report (which is the
best-model evaluation on val) plus the saved ``best eval/mAP``, and prints a comparison table.

Usage:
    python scripts/summarize_results.py            # default root = D:\\free_mlgcn
    python scripts/summarize_results.py --root D:/free_mlgcn/outputs
    python scripts/summarize_results.py --out_csv results_table.csv
"""
import argparse
import glob
import os
import re
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


REPORT_RE = re.compile(
    r"mAP=([\d.]+)\s+Macro-F1=([\d.]+)\s+Micro-F1=([\d.]+)\s+Hamming=([\d.]+)"
)
BEST_RE = re.compile(r"done\.\s*best eval/mAP=([\d.]+)")
ECHO_RE = re.compile(r"epoch (\d+) it (\d+)")


def parse_log(path):
    """Return (final_report, best_mAP, last_iter) or None. final_report = (mAP, mf1, mif1, ham)."""
    final_report, best_map, last_iter = None, None, None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return None
    matches = REPORT_RE.findall(text)
    if matches:
        final_report = tuple(float(x) for x in matches[-1])
    m = BEST_RE.search(text)
    if m:
        best_map = float(m.group(1).rstrip("."))
    iters = ECHO_RE.findall(text)
    if iters:
        last_iter = int(iters[-1][1])
    return final_report, best_map, last_iter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"D:\free_mlgcn\outputs")
    p.add_argument("--out_csv", default="")
    a = p.parse_args()

    logs = sorted(glob.glob(os.path.join(a.root, "*", "train.log")))
    if not logs:
        print(f"no train.log found under {a.root}")
        return

    rows = []
    for lp in logs:
        exp = os.path.basename(os.path.dirname(lp))
        res = parse_log(lp)
        if res is None:
            print(f"  {exp:<30}  (cannot parse)")
            continue
        report, best, iters = res
        if report is None:
            print(f"  {exp:<30}  (no eval report found)")
            continue
        mAP, mf1, mif1, ham = report
        rows.append((exp, mAP, mf1, mif1, ham, best, iters))

    hdr = (f"  {'Experiment':<28}{'mAP':>8}{'Macro-F1':>10}{'Micro-F1':>10}"
           f"{'Hamming':>9}{'best_mAP':>10}{'iters':>8}")
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for exp, mAP, mf1, mif1, ham, best, iters in rows:
        print(f"  {exp:<28}{mAP:>8.4f}{mf1:>10.4f}{mif1:>10.4f}{ham:>9.4f}"
              f"{best:>10.4f}{iters:>8}")
    print("=" * len(hdr))

    if a.out_csv:
        import csv
        with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["experiment", "mAP", "macro_F1", "micro_F1", "hamming", "best_mAP", "iters"])
            for exp, mAP, mf1, mif1, ham, best, iters in rows:
                w.writerow([exp, mAP, mf1, mif1, ham, best, iters])
        print(f"\n[summarize_results] CSV saved to {a.out_csv}")


if __name__ == "__main__":
    main()
